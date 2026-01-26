"""
移动平均线指标 (MA / EMA)

核心思想：
- MA: 过滤噪音，提取趋势
- EMA: 对最新价格更敏感，反应更快

数学原理：
- SMA: 简单算术平均，所有价格权重相同
- EMA: 指数加权平均，近期价格权重更高

适用场景：
- 趋势市：有效
- 震荡市：失效（频繁假信号）
"""

import pandas as pd
import numpy as np
from .base import BaseIndicator, MaType, IndicatorError


class MA(BaseIndicator):
    """
    简单移动平均线 (Simple Moving Average)

    公式: SMA(n) = (P1 + P2 + ... + Pn) / n

    参数:
        period: 周期，常用 5, 10, 20, 60, 120, 250
    """

    def calculate(self, df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
        """
        计算简单移动平均线

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            period: MA 周期，默认 20
            column: 计算列，默认 close（收盘价）

        Returns:
            添加了 MA 列的 DataFrame
        """
        self._validate_data(df)

        if period <= 0:
            raise IndicatorError(f"MA 周期必须大于 0，当前值: {period}")

        result = df.copy()

        # 计算 MA
        ma_column = f"MA{period}"
        result[ma_column] = result[column].rolling(window=period, min_periods=1).mean()

        return result


class EMA(BaseIndicator):
    """
    指数移动平均线 (Exponential Moving Average)

    公式: EMA(t) = α × P(t) + (1-α) × EMA(t-1)
    其中: α = 2 / (n + 1)

    特点:
    - 近期价格权重更高
    - 反应比 SMA 更快
    - 但也更容易被噪音干扰

    参数:
        period: 周期，常用 12, 26
    """

    def calculate(
        self, df: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.DataFrame:
        """
        计算指数移动平均线

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            period: EMA 周期，默认 20
            column: 计算列，默认 close（收盘价）

        Returns:
            添加了 EMA 列的 DataFrame
        """
        self._validate_data(df)

        if period <= 0:
            raise IndicatorError(f"EMA 周期必须大于 0，当前值: {period}")

        result = df.copy()

        # 计算 EMA
        ema_column = f"EMA{period}"
        # pandas 的 ewm 函数直接实现了 EMA
        # alpha=2/(period+1) 是标准参数
        # adjust=False 使用递归公式（与标准EMA定义一致）
        result[ema_column] = (
            result[column]
            .ewm(alpha=2 / (period + 1), adjust=False)
            .mean()
        )

        return result


def calculate_ma_cross_signal(df: pd.DataFrame, fast_period: int = 5, slow_period: int = 20) -> pd.DataFrame:
    """
    计算 MA 金叉死叉信号

    金叉: 快线上穿慢线（买入信号）
    死叉: 快线下穿慢线（卖出信号）

    Args:
        df: 包含 OHLCV 数据的 DataFrame
        fast_period: 快线周期，默认 5
        slow_period: 慢线周期，默认 20

    Returns:
        添加了信号列的 DataFrame
        - MA{fast_period}: 快线
        - MA{slow_period}: 慢线
        - ma_signal: 信号 (1=金叉, -1=死叉, 0=无)
    """
    ma_indicator = MA()
    result = ma_indicator.calculate(df, period=fast_period)
    result = ma_indicator.calculate(result, period=slow_period)

    fast_col = f"MA{fast_period}"
    slow_col = f"MA{slow_period}"

    # 计算金叉死叉
    result["ma_signal"] = 0

    # 金叉：快线上穿慢线
    cross_above = (result[fast_col] > result[slow_col]) & (
        result[fast_col].shift(1) <= result[slow_col].shift(1)
    )
    result.loc[cross_above, "ma_signal"] = 1

    # 死叉：快线下穿慢线
    cross_below = (result[fast_col] < result[slow_col]) & (
        result[fast_col].shift(1) >= result[slow_col].shift(1)
    )
    result.loc[cross_below, "ma_signal"] = -1

    return result
