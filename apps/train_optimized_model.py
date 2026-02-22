#!/usr/bin/env python3
"""
使用优化特征训练ML模型

移除了时间特征和低价值特征，专注于有预测价值的技术特征
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import xgboost as xgb
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.optimized_features import OptimizedFeatureExtractor


class OptimizedModelTrainer:
    """优化版模型训练器"""

    def __init__(self):
        self.storage = SQLiteStorage()
        self.feature_extractor = OptimizedFeatureExtractor(prediction_period=5)

    def prepare_training_data(
        self,
        start_date: str = "20210101",
        end_date: str = "20241231",
        min_samples: int = 60,
    ):
        """
        准备训练数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            min_samples: 最小样本数

        Returns:
            X, y, feature_cols
        """
        logger.info(f"准备训练数据: {start_date} ~ {end_date}")

        # 获取股票列表
        all_stocks = self.storage.get_all_stocks()
        logger.info(f"股票池: {len(all_stocks)} 只")

        X_list = []
        y_list = []

        for i, ts_code in enumerate(all_stocks):
            if (i + 1) % 500 == 0:
                logger.info(f"处理进度: {i+1}/{len(all_stocks)}")

            # 获取日线数据
            df = self.storage.get_daily_prices(ts_code, start_date, end_date)
            if df is None or len(df) < min_samples:
                continue

            # 提取特征
            df_features = self.feature_extractor.extract(df)

            # 获取特征列
            feature_cols = [c for c in df_features.columns if c.startswith("f_")]

            # 过滤有效数据
            df_valid = df_features.dropna(subset=feature_cols)

            if len(df_valid) < min_samples:
                continue

            # 计算标签：5日后的收益率
            df_valid = df_valid.copy()
            df_valid["future_return"] = df_valid["close"].pct_change(5).shift(-5)

            # 移除最后5行（无标签）
            df_valid = df_valid.dropna(subset=["future_return"])

            if len(df_valid) < min_samples:
                continue

            # 提取特征和标签
            X = df_valid[feature_cols].values
            y = df_valid["future_return"].values

            X_list.append(X)
            y_list.append(y)

        # 合并所有股票数据
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)

        # 处理异常值
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"训练数据: {X_all.shape[0]} 样本, {X_all.shape[1]} 特征")

        return X_all, y_all, feature_cols

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict = None,
    ):
        """
        训练XGBoost模型

        Args:
            X: 特征矩阵
            y: 标签
            params: 模型参数

        Returns:
            训练好的模型
        """
        if params is None:
            params = {
                "objective": "reg:squarederror",
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 300,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
            }

        logger.info(f"训练模型，参数: {params}")

        model = xgb.XGBRegressor(**params)
        model.fit(X, y)

        return model

    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray):
        """
        评估模型

        Args:
            model: 训练好的模型
            X: 特征矩阵
            y: 真实标签

        Returns:
            评估指标
        """
        y_pred = model.predict(X)

        # 基本指标
        mse = np.mean((y_pred - y) ** 2)
        mae = np.mean(np.abs(y_pred - y))
        correlation = np.corrcoef(y_pred, y)[0, 1]

        # 方向准确率
        direction_accuracy = np.mean(np.sign(y_pred) == np.sign(y))

        # IC (Information Coefficient)
        ic = np.corrcoef(y_pred, y)[0, 1]

        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "correlation": float(correlation),
            "direction_accuracy": float(direction_accuracy),
            "ic": float(ic),
        }

        logger.info(f"模型评估: MSE={mse:.6f}, IC={ic:.4f}, 方向准确率={direction_accuracy:.2%}")

        return metrics

    def save_model(self, model, feature_cols: list, output_dir: str = "models"):
        """
        保存模型

        Args:
            model: 训练好的模型
            feature_cols: 特征列名
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")

        # 保存模型
        model_path = output_path / f"xgboost_optimized_{timestamp}.json"
        model.save_model(str(model_path))
        logger.info(f"模型已保存: {model_path}")

        # 保存特征列
        feature_path = output_path / f"feature_cols_optimized_{timestamp}.json"
        with open(feature_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        logger.info(f"特征列已保存: {feature_path}")

        # 同时保存为默认名称（方便加载）
        model_path_default = output_path / "xgboost_optimized.json"
        model.save_model(str(model_path_default))

        feature_path_default = output_path / "feature_cols_optimized.json"
        with open(feature_path_default, "w") as f:
            json.dump(feature_cols, f, indent=2)

        return str(model_path), str(feature_path)

    def analyze_feature_importance(self, model, feature_cols: list, top_n: int = 20):
        """
        分析特征重要性

        Args:
            model: 训练好的模型
            feature_cols: 特征列名
            top_n: 显示前N个重要特征
        """
        importance = model.get_booster().get_score(importance_type="gain")

        df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in importance.items()
        ])
        df = df.sort_values("importance", ascending=False)

        logger.info(f"\n=== 特征重要性 Top {top_n} ===")
        for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
            # 解析特征索引
            if row["feature"].startswith("f"):
                idx = int(row["feature"][1:])
                if idx < len(feature_cols):
                    feature_name = feature_cols[idx]
                else:
                    feature_name = row["feature"]
            else:
                feature_name = row["feature"]

            logger.info(f"{i:>3}. {feature_name:<25} {row['importance']:>10.1f}")

        return df


def main():
    """主函数"""
    logger.info("=== 开始训练优化版ML模型 ===")

    trainer = OptimizedModelTrainer()

    # 准备数据
    X, y, feature_cols = trainer.prepare_training_data(
        start_date="20210101",
        end_date="20241231",
    )

    logger.info(f"特征数量: {len(feature_cols)}")
    logger.info(f"特征列表: {feature_cols}")

    # 训练模型
    model = trainer.train_model(X, y)

    # 评估模型
    metrics = trainer.evaluate_model(model, X, y)

    # 分析特征重要性
    trainer.analyze_feature_importance(model, feature_cols)

    # 保存模型
    model_path, feature_path = trainer.save_model(model, feature_cols)

    logger.info(f"\n=== 训练完成 ===")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"特征路径: {feature_path}")
    logger.info(f"评估指标: {metrics}")


if __name__ == "__main__":
    main()
