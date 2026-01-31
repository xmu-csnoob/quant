"""
RSI 指标 (Relative Strength Index)

核心思想：
- 判断"涨太多了"还是"跌太多了"
- 基于均值回归假设：涨多了会跌，跌多了会涨
- 测量多空力量对比

数学原理：
- 上涨幅度 = max(今日收盘 - 昨日收盘, 0)
- 下跌幅度 = max(昨日收盘 - 今日收盘, 0)
- 平均涨幅 = EMA(上涨幅度, period)
- 平均跌幅 = EMA(下跌幅度, period)
- RS = 平均涨幅 / 平均跌幅
- RSI = 100 - (100 / (1 + RS))

参数:
    - period: 周期，默认 14
    - overbought: 超买线，默认 70
    - oversold: 超卖线，默认 30

适用场景：
    - 短期/中期投资：有效
    - 震荡市：有效
    - 长期投资：失效（钝化问题）
    - 趋势市：失效（长期钝化）

注意事项:
    - 强势股可以 RSI 长期 > 70（钝化）
    - 弱势股可以 RSI 长期 < 30（钝化）
    - 需要结合趋势指标（MA、MACD）使用
"""

import pandas as pd
import numpy as np
from .base import BaseIndicator, IndicatorError


class RSI(BaseIndicator):
    """
    RSI 相对强弱指标

    参数:
        period: 周期，默认 14
        overbought: 超买线，默认 70
        oversold: 超卖线，默认 30

    输出:
        RSI: RSI 值 (0-100)
        overbought: 是否超买
        oversold: 是否超卖
    """

    def calculate(
        self,
        df: pd.DataFrame,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        计算 RSI 指标

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            period: RSI 周期，默认 14
            overbought: 超买线，默认 70
            oversold: 超卖线，默认 30
            column: 计算列，默认 close（收盘价）

        Returns:
            添加了 RSI, overbought, oversold 列的 DataFrame
        """
        self._validate_data(df)

        if period <= 1:
            raise IndicatorError(f"RSI 周期必须大于 1，当前值: {period}")

        if overbought <= oversold:
            raise IndicatorError(
                f"超买线 ({overbought}) 必须大于超卖线 ({oversold})"
            )

        result = df.copy()

        # 计算价格变化
        delta = result[column].diff()

        # 分离涨跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 计算 EMA
        alpha = 2 / (period + 1)
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        # 计算 RS 和 RSI
        rs = avg_gain / avg_loss
        result["RSI"] = 100 - (100 / (1 + rs))

        # 标记超买超卖
        result["rsi_overbought"] = result["RSI"] > overbought
        result["rsi_oversold"] = result["RSI"] < oversold

        return result


def calculate_rsi_signal(
    df: pd.DataFrame,
    period: int = 14,
    overbought: float = 70,
    oversold: float = 30,
) -> pd.DataFrame:
    """
    计算 RSI 交易信号

    超买: RSI > 70，可能回调（卖出信号）
    超卖: RSI < 30，可能反弹（买入信号）

    Args:
        df: 包含 OHLCV 数据的 DataFrame
        period: RSI 周期，默认 14
        overbought: 超买线，默认 70
        oversold: 超卖线，默认 30

    Returns:
        添加了信号列的 DataFrame
        - RSI: RSI 值
        - rsi_overbought: 是否超买
        - rsi_oversold: 是否超卖
        - rsi_signal: 信号 (1=超卖, -1=超买, 0=无)
    """
    rsi_indicator = RSI()
    result = rsi_indicator.calculate(df, period, overbought, oversold)

    # 计算信号
    result["rsi_signal"] = 0
    result.loc[result["rsi_oversold"], "rsi_signal"] = 1  # 超卖 = 买入信号
    result.loc[result["rsi_overbought"], "rsi_signal"] = -1  # 超买 = 卖出信号

    return result


def detect_rsi_divergence(
    df: pd.DataFrame,
    column: str = "close",
    lookback: int = 20,
) -> pd.DataFrame:
    """
    检测 RSI 背离

    顶背离: 价格创新高，但 RSI 不创新高（卖出信号）
    底背离: 价格创新低，但 RSI 不创新低（买入信号）

    Args:
        df: 包含 RSI 数据的 DataFrame
        column: 价格列，默认 close
        lookback: 回溯周期，默认 20

    Returns:
        添加了背离标记的 DataFrame
        - rsi_bullish_divergence: 底背离 (True/False)
        - rsi_bearish_divergence: 顶背离 (True/False)
    """
    result = df.copy()

    if "RSI" not in result.columns:
        raise IndicatorError("输入数据必须包含 RSI 列，请先计算 RSI")

    result["rsi_bullish_divergence"] = False
    result["rsi_bearish_divergence"] = False

    # 检测底背离
    for i in range(lookback, len(result)):
        # 寻找价格的局部低点
        price_low_1 = result[column].iloc[i - lookback : i].min()
        price_low_2 = result[column].iloc[i : i + lookback].min() if i + lookback < len(result) else None

        if price_low_2 is not None and price_low_2 < price_low_1:
            # 价格创新低，检查 RSI 是否创新低
            rsi_low_1 = result["RSI"].iloc[i - lookback : i].min()
            rsi_low_2 = result["RSI"].iloc[i : i + lookback].min()

            if rsi_low_2 > rsi_low_1:
                # 底背离！
                result.loc[result.index[i : i + lookback], "rsi_bullish_divergence"] = True

    # 检测顶背离
    for i in range(lookback, len(result)):
        # 寻找价格的局部高点
        price_high_1 = result[column].iloc[i - lookback : i].max()
        price_high_2 = result[column].iloc[i : i + lookback].max() if i + lookback < len(result) else None

        if price_high_2 is not None and price_high_2 > price_high_1:
            # 价格创新高，检查 RSI 是否创新高
            rsi_high_1 = result["RSI"].iloc[i - lookback : i].max()
            rsi_high_2 = result["RSI"].iloc[i : i + lookback].max()

            if rsi_high_2 < rsi_high_1:
                # 顶背离！
                result.loc[result.index[i : i + lookback], "rsi_bearish_divergence"] = True

    return result
