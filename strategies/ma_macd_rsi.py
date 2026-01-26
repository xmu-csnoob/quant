"""
MA + MACD + RSI 组合策略

策略类型：趋势跟踪 (TREND_FOLLOWING)
资产类别：股票 (STOCK)
交易频率：日线 (DAILY)

策略思路：
1. 趋势判断：MA 多头排列确定趋势向上
2. 动能确认：MACD 金叉确认买入时机
3. 风险控制：RSI 判断是否超买超卖
4. 信号确认：多个指标同时发出信号时才交易
"""

import pandas as pd
import numpy as np
from .base import (
    BaseStrategy,
    Signal,
    SignalType,
    Position,
    PositionType,
    StrategyError,
    StrategyType,
    AssetClass,
    Frequency,
)
from utils.indicators import MA, MACD, RSI
from utils.indicators.ma import calculate_ma_cross_signal
from utils.indicators.macd import calculate_macd_signal, detect_divergence
from utils.indicators.rsi import calculate_rsi_signal, detect_rsi_divergence


class MaMacdRsiStrategy(BaseStrategy):
    """
    MA + MACD + RSI 组合策略

    策略元数据：
    - type: 趋势跟踪
    - asset_class: 股票
    - frequency: 日线

    参数:
        ma_fast: MA 快线周期，默认 5
        ma_slow: MA 慢线周期，默认 20
        ma_long: MA 长线周期，默认 60
        macd_fast: MACD 快线周期，默认 12
        macd_slow: MACD 慢线周期，默认 26
        macd_signal: MACD 信号线周期，默认 9
        rsi_period: RSI 周期，默认 14
        rsi_overbought: RSI 超买线，默认 75
        rsi_oversold: RSI 超卖线，默认 25
    """

    # 策略元数据
    strategy_type = StrategyType.TREND_FOLLOWING
    asset_class = AssetClass.STOCK
    frequency = Frequency.DAILY

    def __init__(
        self,
        ma_fast: int = 5,
        ma_slow: int = 20,
        ma_long: int = 60,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        rsi_overbought: float = 75,
        rsi_oversold: float = 25,
    ):
        super().__init__(name="MA_MACD_RSI")
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.ma_long = ma_long
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有需要的指标

        Args:
            df: 原始 OHLCV 数据

        Returns:
            添加了指标列的 DataFrame
        """
        result = df.copy()

        # 计算 MA
        ma_indicator = MA()
        result = ma_indicator.calculate(result, period=self.ma_fast)
        result = ma_indicator.calculate(result, period=self.ma_slow)
        result = ma_indicator.calculate(result, period=self.ma_long)

        # 计算 MA 交叉信号
        result = calculate_ma_cross_signal(result, self.ma_fast, self.ma_slow)

        # 计算 MACD
        result = calculate_macd_signal(
            result, self.macd_fast, self.macd_slow, self.macd_signal
        )

        # 检测 MACD 背离
        result = detect_divergence(result)

        # 计算 RSI
        result = calculate_rsi_signal(result, self.rsi_period, self.rsi_overbought, self.rsi_oversold)

        # 检测 RSI 背离
        result = detect_rsi_divergence(result)

        # 判断多头排列
        ma_fast_col = f"MA{self.ma_fast}"
        ma_slow_col = f"MA{self.ma_slow}"
        ma_long_col = f"MA{self.ma_long}"
        result["bullish_alignment"] = (
            (result[ma_fast_col] > result[ma_slow_col])
            & (result[ma_slow_col] > result[ma_long_col])
            & (result[ma_fast_col] > result[ma_fast_col].shift(1))
        )

        return result

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        生成交易信号

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            信号列表
        """
        # 计算指标
        result = self.calculate_indicators(df)

        signals = []
        position = None  # 当前持仓状态

        for i in range(max(self.ma_long, self.macd_slow, self.rsi_period), len(result)):
            row = result.iloc[i]
            date = row["trade_date"]
            close = row["close"]

            # 检查当前持仓
            if position is None:
                # 空仓，寻找买入机会
                if self._check_buy_conditions(row):
                    signal = Signal(
                        date=date,
                        signal_type=SignalType.BUY,
                        price=close,
                        reason=self._get_buy_reason(row),
                        confidence=self._calculate_buy_confidence(row),
                    )
                    signals.append(signal)
                    position = Position(
                        entry_date=date,
                        entry_price=close,
                        quantity=100,  # 固定买入100股
                        position_type=PositionType.LONG,
                    )

            else:
                # 有持仓，检查卖出条件
                if self._check_sell_conditions(row, position):
                    signal = Signal(
                        date=date,
                        signal_type=SignalType.SELL,
                        price=close,
                        reason=self._get_sell_reason(row),
                        confidence=self._calculate_sell_confidence(row),
                    )
                    signals.append(signal)
                    position = None

        self._signals = signals
        return signals

    def _check_buy_conditions(self, row: pd.Series) -> bool:
        """检查买入条件"""
        # 条件1：多头排列
        if not row.get("bullish_alignment", False):
            return False

        # 条件2：MACD 金叉（当日或前一日）
        if row["macd_signal"] != 1:
            return False

        # 条件3：RSI 不严重超买（放宽到85）
        if row["RSI"] > 85:
            return False

        # 暂时不考虑背离（背离检测算法需要改进）
        # 条件4：不在顶背离区域
        # if row.get("bearish_divergence", False) or row.get("rsi_bearish_divergence", False):
        #     return False

        return True

    def _check_sell_conditions(self, row: pd.Series, position: Position) -> bool:
        """检查卖出条件"""
        # 条件1：MA 死叉
        if row["ma_signal"] == -1:
            return True

        # 条件2：MACD 死叉
        if row["macd_signal"] == -1:
            return True

        # 条件3：MACD 顶背离
        if row.get("bearish_divergence", False):
            return True

        # 条件4：RSI 严重超买
        if row["RSI"] > 80:  # 硬阈值
            return True

        return False

    def _get_buy_reason(self, row: pd.Series) -> str:
        """获取买入原因"""
        reasons = []

        if row.get("bullish_alignment", False):
            reasons.append("多头排列")

        if row["macd_signal"] == 1:
            strength = row.get("macd_signal_strength", "")
            if strength == "zero_axis_above":
                reasons.append("零轴上金叉")
            else:
                reasons.append("MACD金叉")

        rsi = row.get("RSI", 50)
        if rsi < self.rsi_oversold:
            reasons.append(f"RSI超卖({rsi:.1f})")
        elif 30 < rsi < 70:
            reasons.append(f"RSI正常({rsi:.1f})")

        return ", ".join(reasons)

    def _get_sell_reason(self, row: pd.Series) -> str:
        """获取卖出原因"""
        if row["ma_signal"] == -1:
            return "MA死叉"

        if row["macd_signal"] == -1:
            return "MACD死叉"

        if row.get("bearish_divergence", False):
            return "MACD顶背离"

        if row["RSI"] > 80:
            return f"RSI严重超买({row['RSI']:.1f})"

        return "其他卖出条件"

    def _calculate_buy_confidence(self, row: pd.Series) -> float:
        """计算买入信号置信度"""
        confidence = 0.5  # 基础置信度

        # 多头排列 +0.2
        if row.get("bullish_alignment", False):
            confidence += 0.2

        # 零轴上金叉 +0.2
        if row.get("macd_signal_strength") == "zero_axis_above":
            confidence += 0.2

        # MACD 柱状图变长 +0.1
        if row.get("MACD", 0) > 0:
            confidence += 0.1

        return min(confidence, 1.0)

    def _calculate_sell_confidence(self, row: pd.Series) -> float:
        """计算卖出信号置信度"""
        confidence = 0.5  # 基础置信度

        # 多重卖出信号叠加
        if row["ma_signal"] == -1:
            confidence += 0.2
        if row["macd_signal"] == -1:
            confidence += 0.2
        if row.get("bearish_divergence", False):
            confidence += 0.3
        if row["RSI"] > 80:
            confidence += 0.1

        return min(confidence, 1.0)
