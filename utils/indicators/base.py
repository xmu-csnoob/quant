"""
技术指标基础模块

定义技术指标的抽象基类和通用接口
"""

from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
import numpy as np


class IndicatorError(Exception):
    """技术指标计算异常"""

    pass


class MaType(Enum):
    """移动平均线类型"""

    SMA = "SMA"  # 简单移动平均
    EMA = "EMA"  # 指数移动平均


class BaseIndicator(ABC):
    """
    技术指标抽象基类

    所有技术指标都应继承此类并实现 calculate 方法
    """

    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            df: 包含 OHLCV 数据的 DataFrame
                必须包含列: open, high, low, close, volume
            **kwargs: 指标参数

        Returns:
            包含指标值的 DataFrame，原始数据保持不变
        """
        pass

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        验证输入数据

        Args:
            df: 输入数据

        Raises:
            IndicatorError: 数据格式不正确
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise IndicatorError(
                f"缺少必要的数据列: {missing_columns}. "
                f"需要列: {required_columns}"
            )

        if len(df) == 0:
            raise IndicatorError("输入数据不能为空")

    def _handle_ema_first_value(self, data: pd.Series, period: int) -> pd.Series:
        """
        处理 EMA 第一个值

        EMA 的第一个值通常使用 SMA 作为初始值

        Args:
            data: 价格序列
            period: 周期

        Returns:
            处理后的 EMA 序列
        """
        # 第一个值用 SMA
        ema = pd.Series(index=data.index, dtype=float)
        ema.iloc[0] = data.iloc[:period].mean()

        # 后续值用 EMA 公式
        alpha = 2 / (period + 1)
        for i in range(1, len(data)):
            if pd.isna(data.iloc[i]):
                ema.iloc[i] = ema.iloc[i - 1]
            else:
                ema.iloc[i] = alpha * data.iloc[i] + (1 - alpha) * ema.iloc[i - 1]

        return ema
