"""
动态策略选择器

根据市场环境自动选择最佳策略
"""

import pandas as pd
from typing import Dict, Optional
from loguru import logger

from strategies.base import BaseStrategy, Signal, SignalType
from strategies.ma_macd_rsi import MaMacdRsiStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from utils.market_regime import MarketRegimeDetector, MarketRegime


class DynamicStrategySelector(BaseStrategy):
    """
    动态策略选择器

    根据市场环境自动切换策略：
    - 牛市：趋势跟踪策略
    - 熊市：均值回归策略
    - 震荡市：均值回归策略
    - 高波动：空仓观望
    """

    def __init__(
        self,
        strategy_map: Dict[MarketRegime, BaseStrategy] = None,
        default_strategy: BaseStrategy = None,
        confidence_threshold: float = 0.5,
    ):
        """
        初始化

        Args:
            strategy_map: 市场环境 -> 策略的映射
            default_strategy: 默认策略（当置信度不够时使用）
            confidence_threshold: 置信度阈值
        """
        super().__init__(name="Dynamic_Strategy_Selector")

        # 默认策略映射
        if strategy_map is None:
            strategy_map = {
                MarketRegime.BULL: MaMacdRsiStrategy(),  # 牛市用趋势跟踪
                MarketRegime.BEAR: MeanReversionStrategy(),  # 熊市用均值回归
                MarketRegime.SIDEWAYS: MeanReversionStrategy(),  # 震荡用均值回归
                MarketRegime.VOLATILE: None,  # 高波动空仓
            }

        self.strategy_map = strategy_map
        self.default_strategy = default_strategy or MaMacdRsiStrategy()
        self.confidence_threshold = confidence_threshold

        # 市场环境识别器
        self.detector = MarketRegimeDetector()

        logger.info(
            f"DynamicStrategy initialized: "
            f"conf_threshold={confidence_threshold}"
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        生成动态选择的策略信号

        Args:
            df: OHLCV数据

        Returns:
            信号列表
        """
        # 获取市场环境历史
        df_regime = self.detector.get_regime_history(df)

        signals = []
        position = None

        # 遍历每个交易日
        for i in range(60, len(df_regime)):  # 前60天数据不足
            row = df_regime.iloc[i]
            date = row["trade_date"].strftime("%Y%m%d")
            regime_str = row["market_regime"]
            confidence = row["regime_confidence"]

            # 解析市场环境
            try:
                regime = MarketRegime(regime_str)
            except ValueError:
                regime = MarketRegime.SIDEWAYS

            # 选择策略
            if confidence < self.confidence_threshold:
                # 置信度不够，使用默认策略
                strategy = self.default_strategy
            else:
                # 根据市场环境选择策略
                strategy = self.strategy_map.get(
                    regime,
                    self.default_strategy
                )

            # 高波动环境下，如果有持仓则平仓
            if regime == MarketRegime.VOLATILE:
                if position is not None:
                    signals.append(Signal(
                        date=date,
                        signal_type=SignalType.SELL,
                        price=row["close"],
                        reason=f"市场环境转为{regime.value}，平仓观望",
                        confidence=confidence,
                    ))
                    position = None
                continue

            # 对于非ML策略，需要根据历史数据生成信号
            # 这里我们简化处理：使用当前日期的市场环境做决策
            if position is None:
                # 当前无持仓，根据环境决定是否买入
                should_buy = self._should_buy(regime, row, confidence)

                if should_buy:
                    signals.append(Signal(
                        date=date,
                        signal_type=SignalType.BUY,
                        price=row["close"],
                        reason=f"市场环境：{regime.value}（置信度{confidence:.1%}）",
                        confidence=confidence,
                    ))
                    position = "long"
            else:
                # 当前有持仓，根据环境决定是否卖出
                should_sell = self._should_sell(regime, row, confidence)

                if should_sell:
                    signals.append(Signal(
                        date=date,
                        signal_type=SignalType.SELL,
                        price=row["close"],
                        reason=f"市场环境变化：{regime.value}",
                        confidence=confidence,
                    ))
                    position = None

        logger.info(f"Dynamic selector generated {len(signals)} signals")
        return signals

    def _should_buy(self, regime: MarketRegime, row: pd.Series, confidence: float) -> bool:
        """判断是否应该买入"""
        # 牛市且置信度高
        if regime == MarketRegime.BULL and confidence > 0.6:
            return True

        # 震荡市且价格偏低（根据RSI等）
        if regime == MarketRegime.SIDEWAYS:
            # 这里简化处理，实际可以用更多指标
            return False

        return False

    def _should_sell(self, regime: MarketRegime, row: pd.Series, confidence: float) -> bool:
        """判断是否应该卖出"""
        # 从牛市转为非牛市
        if regime != MarketRegime.BULL:
            return True

        return False


class AdaptiveDynamicStrategy(BaseStrategy):
    """
    自适应动态策略

    特点：
    - 实时跟踪市场环境变化
    - 使用多种策略的组合信号
    - 根据市场状态调整仓位
    """

    def __init__(
        self,
        strategies: Dict[str, BaseStrategy],
        rebalance_freq: int = 20,
    ):
        """
        初始化

        Args:
            strategies: 可用策略字典 {"trend": ..., "mean_reversion": ..., "ml": ...}
            rebalance_freq: 重新评估频率（天）
        """
        super().__init__(name="Adaptive_Dynamic")

        self.strategies = strategies
        self.rebalance_freq = rebalance_freq
        self.detector = MarketRegimeDetector()

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        生成自适应动态信号

        策略：
        1. 检测市场环境
        2. 根据环境选择最合适的策略组合
        3. 定期重新评估
        """
        df_regime = self.detector.get_regime_history(df)

        signals = []
        position = None
        last_rebalance = 0

        for i in range(60, len(df_regime)):
            row = df_regime.iloc[i]
            date = row["trade_date"].strftime("%Y%m%d")
            regime_str = row["market_regime"]
            confidence = row["regime_confidence"]

            try:
                regime = MarketRegime(regime_str)
            except ValueError:
                regime = MarketRegime.SIDEWAYS

            # 定期重新评估仓位
            days_since_rebalance = i - last_rebalance
            needs_rebalance = days_since_rebalance >= self.rebalance_freq

            if position is None:
                # 当前无持仓，决定是否买入
                if self._should_enter(regime, confidence):
                    signals.append(Signal(
                        date=date,
                        signal_type=SignalType.BUY,
                        price=row["close"],
                        reason=f"进入市场（环境：{regime.value}，置信度：{confidence:.1%}）",
                        confidence=confidence,
                    ))
                    position = "long"
                    last_rebalance = i
            else:
                # 当前有持仓
                should_exit = (
                    needs_rebalance and self._should_exit(regime, confidence)
                ) or (
                    regime == MarketRegime.VOLATILE and confidence > 0.7
                )

                if should_exit:
                    signals.append(Signal(
                        date=date,
                        signal_type=SignalType.SELL,
                        price=row["close"],
                        reason=f"退出市场（环境：{regime.value}）",
                        confidence=confidence,
                    ))
                    position = None
                    last_rebalance = i

        logger.info(f"Adaptive dynamic generated {len(signals)} signals")
        return signals

    def _should_enter(self, regime: MarketRegime, confidence: float) -> bool:
        """判断是否应该进入市场"""
        # 牛市且置信度高
        if regime == MarketRegime.BULL and confidence > 0.5:
            return True

        # 震荡市且置信度极高
        if regime == MarketRegime.SIDEWAYS and confidence > 0.7:
            return True

        return False

    def _should_exit(self, regime: MarketRegime, confidence: float) -> bool:
        """判断是否应该退出市场"""
        # 熊市或高波动
        if regime in [MarketRegime.BEAR, MarketRegime.VOLATILE]:
            return confidence > 0.5

        return False
