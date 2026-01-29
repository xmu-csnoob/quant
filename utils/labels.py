"""
标签生成器

为机器学习模型生成预测目标
"""

import pandas as pd
import numpy as np
from typing import Literal
from loguru import logger


class LabelGenerator:
    """
    标签生成器

    支持两种任务类型：
    1. regression: 回归任务，预测未来收益率（连续值）
    2. classification: 分类任务，预测涨跌方向（0/1）
    """

    def __init__(
        self,
        prediction_period: int = 5,
        task_type: Literal["regression", "classification"] = "regression",
        threshold: float = 0.02,
    ):
        """
        初始化

        Args:
            prediction_period: 预测未来N天的收益
            task_type: 任务类型，regression或classification
            threshold: 分类任务的涨跌阈值（如0.02表示2%涨幅为正类）
        """
        self.prediction_period = prediction_period
        self.task_type = task_type
        self.threshold = threshold

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成标签

        Args:
            df: 包含close列的DataFrame

        Returns:
            添加了label列的DataFrame
        """
        df = df.copy()

        # 计算未来收益率
        df["future_return"] = df["close"].pct_change(self.prediction_period).shift(-self.prediction_period)

        # 根据任务类型生成标签
        if self.task_type == "regression":
            df["label"] = df["future_return"]
        else:
            # 分类：未来收益率 > threshold 视为正类
            df["label"] = (df["future_return"] > self.threshold).astype(int)

        # 删除未来收益率列（可选保留用于分析）
        # df = df.drop(columns=["future_return"])

        logger.info(
            f"生成标签: {self.task_type}, "
            f"预测周期={self.prediction_period}天, "
            f"正样本比例={df['label'].mean():.2%}"
        )

        return df

    def get_label_distribution(self, df: pd.DataFrame) -> dict:
        """
        获取标签分布统计

        Args:
            df: 包含label列的DataFrame

        Returns:
            统计信息字典
        """
        if self.task_type == "regression":
            return {
                "mean": df["label"].mean(),
                "std": df["label"].std(),
                "min": df["label"].min(),
                "max": df["label"].max(),
                "median": df["label"].median(),
            }
        else:
            return {
                "positive_ratio": df["label"].mean(),
                "negative_ratio": 1 - df["label"].mean(),
                "positive_count": df["label"].sum(),
                "negative_count": len(df) - df["label"].sum(),
            }
