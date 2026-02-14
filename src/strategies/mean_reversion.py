"""
均值回归策略

原理：
- 价格会回归到均值
- 当价格过度偏离均值时，反向交易
- 适用于震荡市和优质蓝筹股

买入信号：
1. 价格跌破布林带下轨
2. RSI < 30（超卖）
3. 价格远低于MA（如 < MA - 2*ATR）

卖出信号：
1. 价格涨回MA附近
2. RSI > 50（回归正常）
"""

import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy, Signal, SignalType
from src.utils.features.technical import TechnicalFeatureExtractor
from src.utils.features.builder import FeatureBuilder
from loguru import logger


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略

    特点：
    - 逢低买入，逢高卖出
    - 适用于震荡市
    - 对优质蓝筹股效果更好
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        ma_period: int = 60,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        """
        初始化均值回归策略

        Args:
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            rsi_period: RSI周期
            rsi_oversold: RSI超卖阈值（买入）
            rsi_overbought: RSI超买阈值（卖出）
            ma_period: 均线周期
            atr_period: ATR周期
            atr_multiplier: ATR倍数（用于判断偏离程度）
        """
        super().__init__(name="Mean_Reversion")

        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

        # 构建特征
        self.feature_builder = FeatureBuilder()
        self.feature_builder.add_extractor(
            TechnicalFeatureExtractor(
                ma_periods=[ma_period],
                bb_period=bb_period,
                bb_std=bb_std,
                rsi_period=rsi_period,
                atr_period=atr_period,
            )
        )

    @staticmethod
    def _format_date(trade_date) -> str:
        """格式化日期为YYYYMMDD字符串"""
        if hasattr(trade_date, 'strftime'):
            return trade_date.strftime("%Y%m%d")
        else:
            return str(trade_date).replace('-', '')[:8]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        生成均值回归信号

        买入条件（满足任一）：
        1. 价格 < 布林带下轨
        2. RSI < 30
        3. 价格 < MA - 2*ATR

        卖出条件：
        1. 价格 > MA（回归均值）
        2. 或 RSI > 50
        """
        # 计算指标
        df = self.calculate_indicators(df)

        signals = []
        position = None  # 当前持仓

        for i in range(self.ma_period, len(df)):
            row = df.iloc[i]
            date = self._format_date(row["trade_date"])
            price = row["close"]

            # 检查持仓状态
            if position is None:
                # 当前无持仓，寻找买入机会
                signal = self._check_buy_signal(row, i, df)
                if signal:
                    signals.append(signal)
                    position = "long"
            else:
                # 当前有持仓，寻找卖出机会
                signal = self._check_sell_signal(row, i, df)
                if signal:
                    signals.append(signal)
                    position = None

        logger.info(f"Generated {len(signals)} signals for {len(df)} bars")
        return signals

    def _check_buy_signal(self, row: pd.Series, i: int, df: pd.DataFrame) -> Signal | None:
        """检查买入信号"""
        reasons = []
        score = 0

        # 1. 布林带下轨
        if "BB_lower" in df.columns and not pd.isna(row["BB_lower"]):
            if row["close"] < row["BB_lower"]:
                reasons.append(f"价格{row['close']:.2f} < 布林下轨{row['BB_lower']:.2f}")
                score += 0.4

        # 2. RSI超卖
        if "RSI" in df.columns and not pd.isna(row["RSI"]):
            if row["RSI"] < self.rsi_oversold:
                reasons.append(f"RSI={row['RSI']:.1f} < {self.rsi_oversold}（超卖）")
                score += 0.35

        # 3. 价格远低于MA
        ma_col = f"MA{self.ma_period}"
        if ma_col in df.columns and "ATR" in df.columns:
            if not pd.isna(row[ma_col]) and not pd.isna(row["ATR"]):
                threshold = row[ma_col] - self.atr_multiplier * row["ATR"]
                if row["close"] < threshold:
                    reasons.append(f"价格{row['close']:.2f} < MA-{self.atr_multiplier}*ATR={threshold:.2f}")
                    score += 0.25

        # 至少满足一个条件，且分数足够
        if score >= 0.35:
            return Signal(
                date=self._format_date(row["trade_date"]),
                signal_type=SignalType.BUY,
                price=row["close"],
                reason="; ".join(reasons),
                confidence=score,
            )

        return None

    def _check_sell_signal(self, row: pd.Series, i: int, df: pd.DataFrame) -> Signal | None:
        """检查卖出信号"""
        reasons = []
        score = 0

        # 1. 价格回归MA
        ma_col = f"MA{self.ma_period}"
        if ma_col in df.columns and not pd.isna(row[ma_col]):
            if row["close"] >= row[ma_col]:
                reasons.append(f"价格{row['close']:.2f} >= MA{row[ma_col]:.2f}（回归均值）")
                score += 0.6

        # 2. RSI回归正常
        if "RSI" in df.columns and not pd.isna(row["RSI"]):
            if row["RSI"] > 50:
                reasons.append(f"RSI={row['RSI']:.1f} > 50（回归正常）")
                score += 0.4

        # 至少满足一个条件
        if score >= 0.4:
            return Signal(
                date=self._format_date(row["trade_date"]),
                signal_type=SignalType.SELL,
                price=row["close"],
                reason="; ".join(reasons),
                confidence=score,
            )

        return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所需指标"""
        return self.feature_builder.build(df)
