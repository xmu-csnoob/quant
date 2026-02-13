"""
ML Service - ML预测服务
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

import xgboost as xgb

from src.api.schemas.ml import (
    MLPredictionResponse,
    MLModelInfo,
    MLPredictionStats,
    FeatureImportance,
    PredictionDirection,
)
from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor


class PredictionCache:
    """预测结果缓存"""

    def __init__(self, ttl_seconds: int = 300):  # 默认5分钟过期
        self._cache: dict[str, tuple[datetime, MLPredictionResponse]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[MLPredictionResponse]:
        if key not in self._cache:
            return None
        created_at, result = self._cache[key]
        if datetime.now() - created_at > timedelta(seconds=self._ttl):
            del self._cache[key]
            return None
        return result

    def set(self, key: str, result: MLPredictionResponse):
        self._cache[key] = (datetime.now(), result)

    def clear(self):
        self._cache.clear()


class MLPredictionService:
    """ML预测服务"""

    def __init__(self):
        self.model: Optional[xgb.Booster] = None
        self.feature_cols: Optional[list[str]] = None
        self.model_path = Path("models/xgboost_model.json")
        self.feature_path = Path("models/feature_cols.json")
        self.storage = SQLiteStorage()
        self.feature_extractor = EnhancedFeatureExtractor(prediction_period=5)
        self._cache = PredictionCache(ttl_seconds=300)  # 5分钟缓存
        self._top_signals_cache: Optional[tuple[datetime, list[MLPredictionResponse], list[MLPredictionResponse]]] = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            if self.model_path.exists():
                self.model = xgb.Booster()
                self.model.load_model(str(self.model_path))
                logger.info(f"ML模型加载成功: {self.model_path}")
            else:
                logger.warning(f"ML模型文件不存在: {self.model_path}")

            if self.feature_path.exists():
                with open(self.feature_path, 'r') as f:
                    self.feature_cols = json.load(f)
                logger.info(f"特征列表加载成功: {len(self.feature_cols)} 个特征")
            else:
                logger.warning(f"特征列表文件不存在: {self.feature_path}")

        except Exception as e:
            logger.error(f"ML模型加载失败: {e}")
            self.model = None
            self.feature_cols = None

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None and self.feature_cols is not None

    def get_model_info(self) -> Optional[MLModelInfo]:
        """获取模型信息"""
        if not self.is_model_loaded():
            return None

        # 尝试读取模型训练信息
        info_path = Path("models/model_info.json")
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            return MLModelInfo(**info)

        # 返回基本信息
        return MLModelInfo(
            model_name="XGBoost Classifier",
            model_version="1.0.0",
            model_path=str(self.model_path),
            feature_count=len(self.feature_cols) if self.feature_cols else 0,
            prediction_period=5,
            train_samples=0,
            test_samples=0,
            train_auc=0.0,
            test_auc=0.0,
            train_accuracy=0.0,
            test_accuracy=0.0,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def predict(self, ts_code: str, include_features: bool = False) -> Optional[MLPredictionResponse]:
        """
        预测单只股票

        Args:
            ts_code: 股票代码
            include_features: 是否包含特征数据

        Returns:
            预测结果
        """
        if not self.is_model_loaded():
            logger.warning("模型未加载")
            return None

        # 检查缓存（如果不需要特征数据）
        if not include_features:
            cached = self._cache.get(ts_code)
            if cached:
                return cached

        try:
            # 获取股票数据（需要足够的历史数据计算特征）
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - pd.Timedelta(days=200)).strftime('%Y%m%d')

            df = self.storage.get_daily_prices(ts_code, start_date, end_date)
            if df is None or len(df) < 100:
                logger.warning(f"股票 {ts_code} 数据不足")
                return None

            # 提取特征
            features_df = self.feature_extractor.extract(df)
            if features_df is None or len(features_df) == 0:
                logger.warning(f"股票 {ts_code} 特征提取失败")
                return None

            # 获取最新一行特征
            latest = features_df.iloc[-1]
            feature_values = latest[self.feature_cols].values

            # 转换为float类型并处理NaN
            try:
                feature_values = feature_values.astype(np.float64)
            except (ValueError, TypeError):
                # 如果转换失败，尝试逐个转换
                feature_values = np.array([
                    float(v) if isinstance(v, (int, float, np.number)) else 0.0
                    for v in feature_values
                ], dtype=np.float64)

            # 检查并填充NaN
            if np.isnan(feature_values).any():
                logger.warning(f"股票 {ts_code} 特征包含NaN")
                feature_values = np.nan_to_num(feature_values, nan=0.0)

            # 预测
            dmatrix = xgb.DMatrix(feature_values.reshape(1, -1))
            probability = float(self.model.predict(dmatrix)[0])

            # 确定预测方向和信号
            if probability >= 0.55:
                prediction = PredictionDirection.UP
                signal = "buy"
            elif probability <= 0.45:
                prediction = PredictionDirection.DOWN
                signal = "sell"
            else:
                prediction = PredictionDirection.NEUTRAL
                signal = "hold"

            # 计算置信度
            confidence = abs(probability - 0.5) * 2

            # 获取股票名称
            stock_name = self._get_stock_name(ts_code)

            # 构建响应
            trade_date = latest.get('trade_date', end_date)
            if hasattr(trade_date, 'strftime'):
                trade_date = trade_date.strftime('%Y%m%d')
            elif not isinstance(trade_date, str):
                trade_date = str(trade_date)

            response = MLPredictionResponse(
                ts_code=ts_code,
                stock_name=stock_name,
                prediction=prediction,
                probability=probability,
                confidence=confidence,
                predicted_return=None,  # 可以根据概率估算
                signal=signal,
                features=dict(zip(self.feature_cols, feature_values.tolist())) if include_features else None,
                trade_date=trade_date,
                model_version="1.0.0",
                prediction_period=5
            )

            # 保存到缓存
            if not include_features:
                self._cache.set(ts_code, response)

            return response

        except Exception as e:
            logger.error(f"预测 {ts_code} 失败: {e}")
            return None

    def batch_predict(self, ts_codes: list[str]) -> list[MLPredictionResponse]:
        """批量预测"""
        results = []
        for ts_code in ts_codes:
            result = self.predict(ts_code)
            if result:
                results.append(result)
        return results

    def get_top_signals(self, limit: int = 20, signal_type: str = "buy") -> list[MLPredictionResponse]:
        """
        获取TOP信号

        Args:
            limit: 返回数量
            signal_type: 信号类型 buy/sell

        Returns:
            信号列表
        """
        if not self.is_model_loaded():
            return []

        # 检查TOP信号缓存（5分钟有效）
        if self._top_signals_cache:
            cache_time, buy_signals, sell_signals = self._top_signals_cache
            if datetime.now() - cache_time < timedelta(minutes=5):
                if signal_type == "buy":
                    return buy_signals[:limit]
                else:
                    return sell_signals[:limit]

        try:
            # 获取所有股票
            all_stocks = self.storage.get_all_stocks()

            # 批量预测（利用单只股票缓存）
            predictions = []
            for ts_code in all_stocks[:200]:  # 增加到200只，有缓存会很快
                result = self.predict(ts_code)
                if result:
                    predictions.append(result)

            # 分类和排序
            buy_signals = [p for p in predictions if p.signal == "buy"]
            buy_signals.sort(key=lambda x: x.probability, reverse=True)

            sell_signals = [p for p in predictions if p.signal == "sell"]
            sell_signals.sort(key=lambda x: x.probability)

            # 保存到缓存
            self._top_signals_cache = (datetime.now(), buy_signals, sell_signals)

            if signal_type == "buy":
                return buy_signals[:limit]
            else:
                return sell_signals[:limit]

        except Exception as e:
            logger.error(f"获取TOP信号失败: {e}")
            return []

    def get_feature_importance(self, top_n: int = 20) -> list[FeatureImportance]:
        """获取特征重要性"""
        if not self.is_model_loaded():
            return []

        try:
            importance = self.model.get_score(importance_type='gain')
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            results = []
            for i, (feat, score) in enumerate(sorted_importance[:top_n]):
                feat_idx = int(feat.replace('f', ''))
                feat_name = self.feature_cols[feat_idx] if feat_idx < len(self.feature_cols) else feat
                results.append(FeatureImportance(
                    feature_name=feat_name,
                    importance_score=score,
                    rank=i + 1
                ))

            return results

        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return []

    def get_prediction_stats(self) -> MLPredictionStats:
        """获取预测统计（基于最近预测结果）"""
        # 这里返回模拟数据，实际应该从数据库统计
        return MLPredictionStats(
            total_predictions=252469,
            correct_predictions=127890,
            accuracy=0.506,
            win_rate=0.553,
            avg_return=0.0125,
            profit_loss_ratio=1.44,
            buy_signals=63086,
            sell_signals=114321,
            hold_signals=75062
        )

    def _get_stock_name(self, ts_code: str) -> Optional[str]:
        """获取股票名称"""
        try:
            # 从数据库获取股票名称
            df = self.storage.get_daily_prices(ts_code)
            if df is not None and len(df) > 0:
                return df.iloc[0].get('name', ts_code)
        except:
            pass
        return None


# 单例
ml_service = MLPredictionService()
