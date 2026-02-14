"""
机器学习交易策略

使用训练好的模型进行交易信号预测
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

from src.strategies.base import BaseStrategy, Signal, SignalType
from src.utils.features.ml_features import MLFeatureExtractor


class MLStrategy(BaseStrategy):
    """
    机器学习交易策略

    工作流程：
    1. 训练阶段：使用历史数据训练模型
    2. 预测阶段：用模型预测未来收益率，生成信号
    """

    def __init__(
        self,
        model: object,
        feature_extractor,
        threshold: float = 0.02,
        prediction_period: int = 5,
        feature_cols: Optional[list] = None,
    ):
        """
        初始化

        Args:
            model: 训练好的机器学习模型（需有predict方法）
            feature_extractor: 特征提取器
            threshold: 买入阈值（预测收益率 > 此值时买入）
            prediction_period: 预测周期（天）
            feature_cols: 特征列名列表（可选，如不指定则使用所有f_开头的列）
        """
        super().__init__(name=f"ML_{model.__class__.__name__}")

        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.prediction_period = prediction_period
        self.feature_cols = feature_cols

        logger.info(
            f"ML Strategy initialized: {model.__class__.__name__}, "
            f"threshold={threshold:.2%}, features={len(feature_cols) if feature_cols else 'auto'}"
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        生成交易信号

        Args:
            df: 原始OHLCV数据

        Returns:
            信号列表
        """
        # 提取特征
        df = self.feature_extractor.extract(df)

        # 获取特征列
        if self.feature_cols:
            feature_cols = self.feature_cols
        else:
            feature_cols = [c for c in df.columns if c.startswith("f_")]

        # 过滤有效数据（特征无NaN）
        df_valid = df.dropna(subset=feature_cols).copy()

        if len(df_valid) == 0:
            logger.warning("没有有效数据用于预测")
            return []

        # 预测
        X = df_valid[feature_cols].values

        # 处理NaN（用0填充）
        import numpy as np
        X = np.nan_to_num(X, nan=0.0)

        # 根据模型类型选择预测方式
        try:
            import xgboost as xgb
            if isinstance(self.model, xgb.Booster):
                # XGBoost Booster需要DMatrix
                dmatrix = xgb.DMatrix(X)
                predictions = self.model.predict(dmatrix)
            else:
                # sklearn-style模型
                predictions = self.model.predict(X)
        except ImportError:
            # 如果没有xgboost，尝试sklearn方式
            predictions = self.model.predict(X)

        # 生成信号
        signals = []
        position = None

        for i, (idx, row) in enumerate(df_valid.iterrows()):
            pred_return = predictions[i]

            if position is None:
                # 无持仓，检查买入信号
                if pred_return > self.threshold:
                    signals.append(Signal(
                        date=row["trade_date"].strftime("%Y%m%d"),
                        signal_type=SignalType.BUY,
                        price=row["close"],
                        reason=f"预测收益率={pred_return:.2%} > 阈值={self.threshold:.2%}",
                        confidence=float(pred_return),
                    ))
                    position = "long"
            else:
                # 有持仓，检查卖出信号
                # 策略：持仓N天后卖出，或预测收益转负
                if pred_return < 0 or i >= len(df_valid) - 1:
                    signals.append(Signal(
                        date=row["trade_date"].strftime("%Y%m%d"),
                        signal_type=SignalType.SELL,
                        price=row["close"],
                        reason=f"预测收益率={pred_return:.2%} <= 0 或到期",
                        confidence=float(pred_return),
                    ))
                    position = None

        logger.info(f"生成 {len(signals)} 个ML信号")
        return signals

    def save_model(self, path: str):
        """保存模型"""
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"模型已保存: {path}")

    @classmethod
    def load_model(cls, path: str, feature_extractor: MLFeatureExtractor, **kwargs):
        """加载模型"""
        import joblib
        model = joblib.load(path)
        logger.info(f"模型已加载: {path}")
        return cls(model=model, feature_extractor=feature_extractor, **kwargs)
