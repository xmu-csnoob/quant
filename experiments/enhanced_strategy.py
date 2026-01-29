"""
增强版技术指标策略

改进点：
1. 使用特征提取模块统一管理特征
2. 加入成交量确认（量价配合）
3. 加入波动率过滤（ATR）
4. 加入动量确认（ROC/MOM）
5. 动态仓位管理（根据信号强度调整仓位）
6. 更灵活的买卖条件
"""

import pandas as pd
import numpy as np
from strategies.base import (
    BaseStrategy,
    Signal,
    SignalType,
    Position,
    PositionType,
    StrategyType,
    AssetClass,
    Frequency,
)
from utils.features import FeatureBuilder, TechnicalFeatureExtractor


class EnhancedTechnicalStrategy(BaseStrategy):
    """
    增强版技术指标策略

    策略元数据：
    - type: 趋势跟踪
    - asset_class: 股票
    - frequency: 日线

    改进点：
    1. 使用44个技术指标特征
    2. 成交量确认（放量上涨）
    3. 波动率过滤（ATR正常范围）
    4. 动量确认（ROC向上）
    5. 多级信号强度（弱/中/强）
    6. 动态仓位管理
    """

    strategy_type = StrategyType.TREND_FOLLOWING
    asset_class = AssetClass.STOCK
    frequency = Frequency.DAILY

    def __init__(
        self,
        # 信号强度阈值
        min_signal_strength: float = 0.4,  # 降低阈值，增加交易次数
        # 成交量确认
        volume_confirmation: bool = True,
        min_volume_ratio: float = 0.8,  # 量比阈值
        # 波动率过滤
        volatility_filter: bool = True,
        max_atr_ratio: float = 0.08,  # ATR比率上限（8%）
        # 动量确认
        momentum_confirmation: bool = True,
        min_roc: float = -2.0,  # ROC最小值（-2%）
        # 仓位管理
        position_scaling: bool = True,
        base_quantity: int = 100,
    ):
        super().__init__(name="Enhanced_Technical")
        self.min_signal_strength = min_signal_strength
        self.volume_confirmation = volume_confirmation
        self.min_volume_ratio = min_volume_ratio
        self.volatility_filter = volatility_filter
        self.max_atr_ratio = max_atr_ratio
        self.momentum_confirmation = momentum_confirmation
        self.min_roc = min_roc
        self.position_scaling = position_scaling
        self.base_quantity = base_quantity

        # 创建特征构建器
        self.feature_builder = FeatureBuilder()
        self.feature_builder.add_extractor(TechnicalFeatureExtractor())

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征"""
        return self.feature_builder.build(df)

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """生成交易信号"""
        # 计算所有特征
        result = self.calculate_indicators(df)

        signals = []
        position = None

        # 从有完整特征数据的行开始
        start_idx = 60  # MA60需要60天数据

        for i in range(start_idx, len(result)):
            row = result.iloc[i]
            date = row["trade_date"]
            close = row["close"]

            if position is None:
                # 空仓，寻找买入机会
                signal, confidence, quantity = self._check_buy_signal(row, i, result)
                if signal:
                    sig = Signal(
                        date=date,
                        signal_type=SignalType.BUY,
                        price=close,
                        reason=self._get_buy_reason(row),
                        confidence=confidence,
                        quantity=quantity,  # 传递建议的仓位
                    )
                    signals.append(sig)
                    position = Position(
                        entry_date=date,
                        entry_price=close,
                        quantity=quantity,
                        position_type=PositionType.LONG,
                    )
            else:
                # 有持仓，检查卖出条件
                should_sell, reason = self._check_sell_conditions(row, position)
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

        self._signals = signals
        return signals

    def _check_buy_signal(self, row: pd.Series, index: int, df: pd.DataFrame):
        """检查买入信号"""
        score = 0.0
        max_score = 0.0
        reasons = []

        # 1. 趋势判断（权重：40%）
        max_score += 0.4

        # 多头排列（强信号）
        if row.get("bullish_alignment", 0) == 1:
            score += 0.25
            reasons.append("多头排列")

        # MA5 > MA20（短中期向上）
        ma5_above_ma20 = row.get("MA5", 0) > row.get("MA20", 0)
        if ma5_above_ma20:
            score += 0.10
            reasons.append("MA5>MA20")

        # MA20斜率向上（趋势向上）
        if row.get("MA20_slope", 0) > 0:
            score += 0.05
            reasons.append("MA20斜率向上")

        # 2. 动能确认（权重：25%）
        max_score += 0.25

        # MACD在零轴上方或金叉
        dif = row.get("DIF", 0)
        dea = row.get("DEA", 0)
        macd_above_zero = dif > 0

        if macd_above_zero:
            score += 0.15
            reasons.append("MACD零轴上方")

        # DIF上穿DEA（金叉）
        if index > 0:
            prev_dif = df.iloc[index - 1].get("DIF", 0)
            prev_dea = df.iloc[index - 1].get("DEA", 0)
            golden_cross = (dif > dea) and (prev_dif <= prev_dea)
            if golden_cross:
                score += 0.10
                reasons.append("MACD金叉")

        # 3. 振荡指标（权重：15%）
        max_score += 0.15

        rsi = row.get("RSI", 50)
        # RSI在合理区间（30-70）
        if 30 < rsi < 70:
            score += 0.10
            reasons.append(f"RSI正常({rsi:.1f})")
        elif 20 < rsi <= 30:
            # RSI偏低但不是超卖，可能是买入机会
            score += 0.05
            reasons.append(f"RSI偏低({rsi:.1f})")

        # KDJ金叉
        k = row.get("K", 50)
        d = row.get("D", 50)
        if index > 0:
            prev_k = df.iloc[index - 1].get("K", 50)
            prev_d = df.iloc[index - 1].get("D", 50)
            kdj_golden = (k > d) and (prev_k <= prev_d)
            if kdj_golden:
                score += 0.05
                reasons.append("KDJ金叉")

        # 4. 成交量确认（权重：10%）
        max_score += 0.10

        if self.volume_confirmation:
            volume_ratio = row.get("volume_ratio", 1)
            if volume_ratio > self.min_volume_ratio:
                score += 0.10
                reasons.append(f"放量(量比{volume_ratio:.2f})")

        # 5. 波动率过滤（权重：5%）
        max_score += 0.05

        if self.volatility_filter:
            atr_ratio = row.get("ATR_ratio", 0)
            # ATR在合理范围（不要太大也不要太小）
            if 0.02 < atr_ratio < self.max_atr_ratio:
                score += 0.05
                reasons.append(f"波动率正常({atr_ratio*100:.2f}%)")

        # 6. 动量确认（权重：5%）
        max_score += 0.05

        if self.momentum_confirmation:
            roc = row.get("ROC_10", 0)
            if roc > self.min_roc:
                score += 0.05
                reasons.append(f"动量向上(ROC{roc:.2f})")

        # 计算置信度
        confidence = score / max_score if max_score > 0 else 0

        # 检查是否达到最低阈值
        if confidence >= self.min_signal_strength:
            # 计算仓位（动态）
            quantity = self._calculate_quantity(confidence)
            return True, confidence, quantity

        return False, 0, 0

    def _check_sell_conditions(self, row: pd.Series, position: Position) -> tuple[bool, str]:
        """检查卖出条件"""
        reasons = []
        sell_signals = 0

        # 1. 死叉信号
        dif = row.get("DIF", 0)
        dea = row.get("DEA", 0)
        if dif < dea:
            sell_signals += 1
            reasons.append("MACD死叉")

        # 2. MA死叉
        ma5 = row.get("MA5", 0)
        ma20 = row.get("MA20", 0)
        if ma5 < ma20:
            sell_signals += 1
            reasons.append("MA5<MA20")

        # 3. 空头排列
        if row.get("bearish_alignment", 0) == 1:
            sell_signals += 1
            reasons.append("空头排列")

        # 4. RSI严重超买
        rsi = row.get("RSI", 50)
        if rsi > 80:
            sell_signals += 1
            reasons.append(f"RSI严重超买({rsi:.1f})")

        # 5. KDJ高位死叉
        k = row.get("K", 50)
        d = row.get("D", 50)
        j = row.get("J", 50)
        if k < d and j > 100:
            sell_signals += 1
            reasons.append("KDJ高位死叉")

        # 6. 缩量上涨（量价背离）
        volume_ratio = row.get("volume_ratio", 1)
        if volume_ratio < 0.5:
            sell_signals += 1
            reasons.append("缩量")

        # 至少2个卖出信号才卖出
        if sell_signals >= 2:
            return True, ", ".join(reasons[:3])

        return False, ""

    def _calculate_quantity(self, confidence: float) -> int:
        """根据信号强度动态计算仓位"""
        if not self.position_scaling:
            return self.base_quantity

        # 置信度越高，仓位越大
        # confidence范围：0.4-1.0
        # base_quantity * (1 + (confidence - 0.5))
        multiplier = 1 + (confidence - 0.5)
        multiplier = max(0.5, min(2.0, multiplier))  # 限制在0.5-2倍

        return int(self.base_quantity * multiplier)

    def _get_buy_reason(self, row: pd.Series) -> str:
        """获取买入原因"""
        reasons = []

        if row.get("bullish_alignment", 0) == 1:
            reasons.append("多头排列")

        dif = row.get("DIF", 0)
        if dif > 0:
            reasons.append("MACD零轴上方")

        rsi = row.get("RSI", 50)
        if 30 < rsi < 70:
            reasons.append(f"RSI正常({rsi:.1f})")

        volume_ratio = row.get("volume_ratio", 1)
        if volume_ratio > 1:
            reasons.append(f"放量(量比{volume_ratio:.2f})")

        return ", ".join(reasons)

    def _calculate_sell_confidence(self, row: pd.Series) -> float:
        """计算卖出信号置信度"""
        confidence = 0.5

        # 多个卖出信号叠加
        if row.get("bearish_alignment", 0) == 1:
            confidence += 0.2

        rsi = row.get("RSI", 50)
        if rsi > 80:
            confidence += 0.1

        return min(confidence, 1.0)
