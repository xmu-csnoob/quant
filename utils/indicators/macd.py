"""
MACD 指标 (Moving Average Convergence Divergence)

核心思想：
- 基于双 EMA 系统的改进版
- 不仅判断趋势方向，还判断动能强弱
- 能提前预警趋势反转（背离信号）

数学原理：
- DIF = EMA(12) - EMA(26)  [快线，一阶导数]
- DEA = EMA(9) of DIF      [慢线，DIF 的平滑]
- MACD柱 = (DIF - DEA) × 2 [二阶导数，加速度]

适用场景：
- 趋势市：有效，能判断动能强弱
- 震荡市：失效，频繁假信号
"""

import pandas as pd
import numpy as np
from .base import BaseIndicator, IndicatorError


class MACD(BaseIndicator):
    """
    MACD 指标

    参数:
        fast_period: 快线周期，默认 12
        slow_period: 慢线周期，默认 26
        signal_period: 信号线周期，默认 9

    输出:
        DIF: 快线 (EMA12 - EMA26)
        DEA: 慢线 (DIF 的 EMA9)
        MACD: 柱状图 ((DIF - DEA) × 2)
    """

    def calculate(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        计算 MACD 指标

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            fast_period: 快线周期，默认 12
            slow_period: 慢线周期，默认 26
            signal_period: 信号线周期，默认 9
            column: 计算列，默认 close（收盘价）

        Returns:
            添加了 DIF, DEA, MACD 列的 DataFrame
        """
        self._validate_data(df)

        if fast_period >= slow_period:
            raise IndicatorError(
                f"快线周期 ({fast_period}) 必须小于慢线周期 ({slow_period})"
            )

        if signal_period <= 0:
            raise IndicatorError(f"信号线周期必须大于 0，当前值: {signal_period}")

        result = df.copy()

        # 计算 EMA(12) 和 EMA(26)
        ema_fast = result[column].ewm(alpha=2 / (fast_period + 1), adjust=False).mean()
        ema_slow = result[column].ewm(alpha=2 / (slow_period + 1), adjust=False).mean()

        # 计算 DIF
        result["DIF"] = ema_fast - ema_slow

        # 计算 DEA (DIF 的 EMA)
        result["DEA"] = result["DIF"].ewm(
            alpha=2 / (signal_period + 1), adjust=False
        ).mean()

        # 计算 MACD 柱状图
        result["MACD"] = (result["DIF"] - result["DEA"]) * 2

        return result


def calculate_macd_signal(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    计算 MACD 金叉死叉信号

    金叉: DIF 上穿 DEA（买入信号）
    死叉: DIF 下穿 DEA（卖出信号）

    Args:
        df: 包含 OHLCV 数据的 DataFrame
        fast_period: 快线周期，默认 12
        slow_period: 慢线周期，默认 26
        signal_period: 信号线周期，默认 9

    Returns:
        添加了信号列的 DataFrame
        - DIF: 快线
        - DEA: 慢线
        - MACD: 柱状图
        - macd_signal: 信号 (1=金叉, -1=死叉, 0=无)
        - macd_signal_strength: 信号强度 (zero_axis_above=零轴上金叉, zero_axis_below=零轴下金叉, none=无)
    """
    macd_indicator = MACD()
    result = macd_indicator.calculate(df, fast_period, slow_period, signal_period)

    # 计算金叉死叉
    result["macd_signal"] = 0
    result["macd_signal_strength"] = "none"

    # 金叉：DIF 上穿 DEA
    cross_above = (result["DIF"] > result["DEA"]) & (
        result["DIF"].shift(1) <= result["DEA"].shift(1)
    )
    result.loc[cross_above, "macd_signal"] = 1

    # 判断金叉位置
    zero_axis_above = cross_above & (result["DIF"] > 0)
    zero_axis_below = cross_above & (result["DIF"] <= 0)
    result.loc[zero_axis_above, "macd_signal_strength"] = "zero_axis_above"
    result.loc[zero_axis_below, "macd_signal_strength"] = "zero_axis_below"

    # 死叉：DIF 下穿 DEA
    cross_below = (result["DIF"] < result["DEA"]) & (
        result["DIF"].shift(1) >= result["DEA"].shift(1)
    )
    result.loc[cross_below, "macd_signal"] = -1

    return result


def detect_divergence(
    df: pd.DataFrame,
    column: str = "close",
    lookback: int = 20,
) -> pd.DataFrame:
    """
    检测 MACD 背离

    顶背离: 价格创新高，但 DIF 不创新高（卖出信号）
    底背离: 价格创新低，但 DIF 不创新低（买入信号）

    Args:
        df: 包含 MACD 数据的 DataFrame
        column: 价格列，默认 close
        lookback: 回溯周期，默认 20

    Returns:
        添加了背离标记的 DataFrame
        - bullish_divergence: 底背离 (True/False)
        - bearish_divergence: 顶背离 (True/False)
    """
    result = df.copy()

    if "DIF" not in result.columns:
        raise IndicatorError("输入数据必须包含 DIF 列，请先计算 MACD")

    result["bullish_divergence"] = False
    result["bearish_divergence"] = False

    # 检测底背离
    for i in range(lookback, len(result)):
        # 寻找价格的局部低点
        price_low_1 = result[column].iloc[i - lookback : i].min()
        price_low_2 = result[column].iloc[i : i + lookback].min() if i + lookback < len(result) else None

        if price_low_2 is not None and price_low_2 < price_low_1:
            # 价格创新低，检查 DIF 是否创新低
            dif_low_1 = result["DIF"].iloc[i - lookback : i].min()
            dif_low_2 = result["DIF"].iloc[i : i + lookback].min()

            if dif_low_2 > dif_low_1:
                # 底背离！
                result.loc[result.index[i : i + lookback], "bullish_divergence"] = True

    # 检测顶背离
    for i in range(lookback, len(result)):
        # 寻找价格的局部高点
        price_high_1 = result[column].iloc[i - lookback : i].max()
        price_high_2 = result[column].iloc[i : i + lookback].max() if i + lookback < len(result) else None

        if price_high_2 is not None and price_high_2 > price_high_1:
            # 价格创新高，检查 DIF 是否创新高
            dif_high_1 = result["DIF"].iloc[i - lookback : i].max()
            dif_high_2 = result["DIF"].iloc[i : i + lookback].max()

            if dif_high_2 < dif_high_1:
                # 顶背离！
                result.loc[result.index[i : i + lookback], "bearish_divergence"] = True

    return result
