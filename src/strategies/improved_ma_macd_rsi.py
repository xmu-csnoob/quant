"""
改进版 MA + MACD + RSI 策略

改进点：
1. 区分短期回调与趋势反转
2. 添加止盈止损机制
3. 持仓时间管理
4. 信号质量评估
"""

import pandas as pd
import numpy as np
from src.strategies.base import (
    BaseStrategy,
    Signal,
    SignalType,
    Position,
    PositionType,
    StrategyType,
    AssetClass,
    Frequency,
)
from src.utils.indicators import MA, MACD, RSI
from src.utils.indicators.ma import calculate_ma_cross_signal
from src.utils.indicators.macd import calculate_macd_signal, detect_divergence
from src.utils.indicators.rsi import calculate_rsi_signal, detect_rsi_divergence


class ImprovedMaMacdRsiStrategy(BaseStrategy):
    """
    改进版 MA + MACD + RSI 策略

    策略元数据：
    - type: 趋势跟踪
    - asset_class: 股票
    - frequency: 日线

    改进点：
    1. 买入条件：区分"强金叉"和"弱金叉"
    2. 卖出条件：区分"短期回调"和"趋势反转"
    3. 风险管理：止盈止损、持仓时间
    4. 信号质量：只交易高质量信号
    """

    # 策略元数据
    strategy_type = StrategyType.TREND_FOLLOWING
    asset_class = AssetClass.STOCK
    frequency = Frequency.DAILY

    def __init__(
        self,
        # MA 参数
        ma_fast: int = 5,
        ma_slow: int = 20,
        ma_long: int = 60,
        # MACD 参数
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # RSI 参数
        rsi_period: int = 14,
        rsi_overbought: float = 75,
        rsi_oversold: float = 25,
        # 风险管理参数
        stop_loss_pct: float = 0.05,  # 止损 5%
        take_profit_pct: float = 0.20,  # 止盈 20%
        min_holding_days: int = 3,  # 最少持仓3天
        max_holding_days: int = 60,  # 最多持仓60天
        # 信号质量参数
        min_signal_strength: float = 0.6,  # 最低信号强度
    ):
        super().__init__(name="Improved_MA_MACD_RSI")
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.ma_long = ma_long
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_holding_days = min_holding_days
        self.max_holding_days = max_holding_days
        self.min_signal_strength = min_signal_strength

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有需要的指标"""
        result = df.copy()

        # 计算 MA
        ma_indicator = MA()
        result = ma_indicator.calculate(result, period=self.ma_fast)
        result = ma_indicator.calculate(result, period=self.ma_slow)
        result = ma_indicator.calculate(result, period=self.ma_long)

        # 计算长期趋势方向
        ma_long_col = f"MA{self.ma_long}"
        result["long_trend_up"] = result[ma_long_col] > result[ma_long_col].shift(1)

        # 计算 MA 交叉
        result = calculate_ma_cross_signal(result, self.ma_fast, self.ma_slow)

        # 计算 MACD
        result = calculate_macd_signal(result, self.macd_fast, self.macd_slow, self.macd_signal)

        # 计算 RSI
        result = calculate_rsi_signal(result, self.rsi_period, self.rsi_overbought, self.rsi_oversold)

        # 判断多头排列
        ma_fast_col = f"MA{self.ma_fast}"
        ma_slow_col = f"MA{self.ma_slow}"
        result["bullish_alignment"] = (
            (result[ma_fast_col] > result[ma_slow_col])
            & (result[ma_slow_col] > result[ma_long_col])
            & (result[ma_fast_col] > result[ma_fast_col].shift(1))
        )

        # 计算价格相对长期MA的位置
        result["price_above_long_ma"] = result["close"] > result[ma_long_col]

        return result

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """生成交易信号"""
        result = self.calculate_indicators(df)

        signals = []
        position = None
        position_entry_date = None
        position_entry_price = None
        position_stop_loss = None
        position_take_profit = None

        for i in range(max(self.ma_long, self.macd_slow, self.rsi_period), len(result)):
            row = result.iloc[i]
            date = row["trade_date"]
            close = row["close"]

            # 检查当前持仓
            if position is None:
                # 空仓，寻找买入机会
                signal, confidence = self._check_buy_signal(row, i, result)
                if signal:
                    # 计算止盈止损
                    stop_loss = close * (1 - self.stop_loss_pct)
                    take_profit = close * (1 + self.take_profit_pct)

                    sig = Signal(
                        date=date,
                        signal_type=SignalType.BUY,
                        price=close,
                        reason=self._get_buy_reason(row),
                        confidence=confidence,
                    )
                    signals.append(sig)
                    position = Position(
                        entry_date=date,
                        entry_price=close,
                        quantity=100,
                        position_type=PositionType.LONG,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                    position_entry_date = i
                    position_entry_price = close
                    position_stop_loss = stop_loss
                    position_take_profit = take_profit

            else:
                # 有持仓，检查卖出条件
                holding_days = i - position_entry_date
                should_sell, reason = self._check_sell_conditions(
                    row, position, holding_days, close, position_entry_price
                )

                if should_sell:
                    sig = Signal(
                        date=date,
                        signal_type=SignalType.SELL,
                        price=close,
                        reason=reason,
                        confidence=self._calculate_sell_confidence(row),
                    )
                    signals.append(sig)
                    position = None
                    position_entry_date = None
                    position_entry_price = None

        self._signals = signals
        return signals

    def _check_buy_signal(self, row: pd.Series, index: int, df: pd.DataFrame):
        """检查买入信号"""
        confidence = 0.0
        reasons = []

        # 条件1：多头排列
        if not row.get("bullish_alignment", False):
            return None, 0.0
        confidence += 0.3
        reasons.append("多头排列")

        # 条件2：MACD 金叉
        if row["macd_signal"] != 1:
            return None, 0.0

        # 检查金叉强度
        if row.get("macd_signal_strength") == "zero_axis_above":
            confidence += 0.3
            reasons.append("零轴上金叉")
        else:
            confidence += 0.1
            reasons.append("MACD金叉")

        # 条件3：价格在长期MA上方
        if row.get("price_above_long_ma", False):
            confidence += 0.2
            reasons.append("价格在长期均线上方")

        # 条件4：长期趋势向上
        if row.get("long_trend_up", False):
            confidence += 0.1
            reasons.append("长期趋势向上")

        # 条件5：RSI 不严重超买
        if row["RSI"] < 70:
            confidence += 0.1
            reasons.append(f"RSI正常({row['RSI']:.1f})")
        elif row["RSI"] < 85:
            # RSI偏高但可以接受
            pass
        else:
            # RSI严重超买，不交易
            return None, 0.0

        # 条件6：检查信号强度
        if confidence < self.min_signal_strength:
            return None, 0.0

        return True, confidence

    def _check_sell_conditions(
        self, row: pd.Series, position: Position, holding_days: int, current_price: float, entry_price: float
    ) -> tuple[bool, str]:
        """检查卖出条件"""
        # 条件1：止损（最高优先级）
        if current_price <= position.stop_loss:
            return True, f"止损触发 (价格{current_price:.2f} <= 止损{position.stop_loss:.2f})"

        # 条件2：止盈
        if current_price >= position.take_profit:
            return True, f"止盈触发 (价格{current_price:.2f} >= 止盈{position.take_profit:.2f})"

        # 条件3：最大持仓期
        if holding_days >= self.max_holding_days:
            return True, f"达到最大持仓期 ({holding_days}天)"

        # 条件4：明确趋势反转（而非短期回调）
        # 反转信号：长期趋势向下 + 价格跌破长期MA
        if not row.get("long_trend_up", False) and not row.get("price_above_long_ma", False):
            return True, "趋势反转（长期趋势向下且跌破长期均线）"

        # 条件5：深度死叉（而非浅层回调）
        ma_fast_col = f"MA{self.ma_fast}"
        ma_slow_col = f"MA{self.ma_slow}"
        ma_long_col = f"MA{self.ma_long}"

        # 计算死叉深度
        cross_depth = (row[ma_fast_col] - row[ma_slow_col]) / row[ma_slow_col]

        # 如果深度死叉（超过3%）且长期趋势向下
        if cross_depth < -0.03 and not row.get("long_trend_up", False):
            return True, f"深度死叉 ({cross_depth*100:.2f}%) 且长期趋势向下"

        # 条件6：MACD 顶背离
        if row.get("bearish_divergence", False):
            return True, "MACD 顶背离"

        # 条件7：最短持仓期保护（短期波动不卖出）
        if holding_days < self.min_holding_days:
            # 除非触发止损/止盈，否则持有
            return False, ""

        # 条件8：简单死叉（但长期趋势向上）
        # 这种情况下，可能是短期回调，持有
        return False, ""

    def _get_buy_reason(self, row: pd.Series) -> str:
        """获取买入原因"""
        reasons = []

        if row.get("bullish_alignment", False):
            reasons.append("多头排列")

        if row["macd_signal"] == 1:
            if row.get("macd_signal_strength") == "zero_axis_above":
                reasons.append("零轴上金叉")
            else:
                reasons.append("MACD金叉")

        if row.get("long_trend_up", False):
            reasons.append("长期趋势向上")

        if row.get("price_above_long_ma", False):
            reasons.append("价格在长期均线上方")

        return ", ".join(reasons)

    def _calculate_sell_confidence(self, row: pd.Series) -> float:
        """计算卖出信号置信度"""
        confidence = 0.5

        # 止损/止盈是高置信度信号
        # 但这个在 _check_sell_conditions 中已经处理

        # 趋势反转是高置信度
        if not row.get("long_trend_up", False):
            confidence += 0.3

        return min(confidence, 1.0)
