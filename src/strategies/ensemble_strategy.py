"""
多策略组合

通过组合多个策略来提高稳定性和收益
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger

from src.strategies.base import BaseStrategy, Signal, SignalType
from src.strategies.ma_macd_rsi import MaMacdRsiStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.ml_strategy import MLStrategy


class EnsembleStrategy(BaseStrategy):
    """
    多策略组合

    组合方式：
    1. 投票机制 - 多数策略同意时才交易
    2. 加权机制 - 根据策略历史表现加权
    3. 市场环境切换 - 根据市场状态选择策略
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        method: str = "voting",
        min_agree: int = 2,
        weights: Dict[str, float] = None,
    ):
        """
        初始化组合策略

        Args:
            strategies: 策略列表
            method: 组合方法
                - voting: 投票机制（至少min_agree个策略同意）
                - weighted: 加权平均（需要提供weights）
                - best_performer: 只使用表现最好的策略
            min_agree: 投票机制下最少同意票数
            weights: 各策略权重 {"TrendFollowing": 0.3, "MeanReversion": 0.3, "ML": 0.4}
        """
        names = [s.name for s in strategies]
        super().__init__(name=f"Ensemble_{'_'.join(names)}")

        self.strategies = strategies
        self.method = method
        self.min_agree = min_agree
        self.weights = weights or {s.name: 1.0/len(strategies) for s in strategies}

        logger.info(
            f"Ensemble initialized: {len(strategies)} strategies, "
            f"method={method}"
        )

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        生成组合信号

        Args:
            df: OHLCV数据

        Returns:
            组合后的信号列表
        """
        # 获取各策略的信号
        all_signals = {}
        for strategy in self.strategies:
            signals = strategy.generate_signals(df)
            all_signals[strategy.name] = signals

        # 按日期汇总信号
        date_signals = {}  # {date: {strategy: signal}}

        for strat_name, signals in all_signals.items():
            for sig in signals:
                if sig.date not in date_signals:
                    date_signals[sig.date] = {}
                date_signals[sig.date][strat_name] = sig

        # 根据方法生成组合信号
        if self.method == "voting":
            return self._voting_ensemble(date_signals)
        elif self.method == "weighted":
            return self._weighted_ensemble(date_signals)
        else:
            logger.warning(f"Unknown method: {self.method}, using voting")
            return self._voting_ensemble(date_signals)

    def _voting_ensemble(self, date_signals: Dict) -> List[Signal]:
        """
        投票机制

        买入条件：至少min_agree个策略建议买入
        卖出条件：至少min_agree个策略建议卖出
        """
        combined = []
        position = None

        # 将日期字符串转换为可排序的格式
        sorted_dates = sorted(date_signals.keys(), key=lambda x: pd.to_datetime(x, format="%Y%m%d"))

        for date in sorted_dates:
            signals = date_signals[date]

            buy_votes = sum(1 for s in signals.values() if s.signal_type == SignalType.BUY)
            sell_votes = sum(1 for s in signals.values() if s.signal_type == SignalType.SELL)

            # 汇总理由
            buy_reasons = [s.reason for s in signals.values() if s.signal_type == SignalType.BUY]
            sell_reasons = [s.reason for s in signals.values() if s.signal_type == SignalType.SELL]

            # 使用第一个买入信号的price
            buy_prices = [s.price for s in signals.values() if s.signal_type == SignalType.BUY]
            sell_prices = [s.price for s in signals.values() if s.signal_type == SignalType.SELL]

            # 投票决策
            if position is None:
                # 当前无持仓
                if buy_votes >= self.min_agree:
                    combined.append(Signal(
                        date=date,
                        signal_type=SignalType.BUY,
                        price=buy_prices[0] if buy_prices else 0,
                        reason=f"[投票 {buy_votes}/{len(self.strategies)}] " + "; ".join(buy_reasons),
                        confidence=buy_votes / len(self.strategies),
                    ))
                    position = "long"
            else:
                # 当前有持仓
                if sell_votes >= self.min_agree:
                    combined.append(Signal(
                        date=date,
                        signal_type=SignalType.SELL,
                        price=sell_prices[0] if sell_prices else 0,
                        reason=f"[投票 {sell_votes}/{len(self.strategies)}] " + "; ".join(sell_reasons),
                        confidence=sell_votes / len(self.strategies),
                    ))
                    position = None

        logger.info(f"Ensemble (voting): 生成 {len(combined)} 个信号")
        return combined

    def _weighted_ensemble(self, date_signals: Dict) -> List[Signal]:
        """
        加权机制

        计算各策略的加权置信度，超过阈值时交易
        """
        combined = []
        position = None

        buy_threshold = 0.5  # 加权买入阈值
        sell_threshold = 0.3  # 加权卖出阈值

        # 将日期字符串转换为可排序的格式
        sorted_dates = sorted(date_signals.keys(), key=lambda x: pd.to_datetime(x, format="%Y%m%d"))

        for date in sorted_dates:
            signals = date_signals[date]

            # 计算加权分数
            buy_score = 0
            sell_score = 0

            for s in signals.values():
                weight = self.weights.get(s.name, 0)
                confidence = s.confidence if s.confidence else 0.5

                if s.signal_type == SignalType.BUY:
                    buy_score += weight * confidence
                elif s.signal_type == SignalType.SELL:
                    sell_score += weight * confidence

            # 汇总信息
            buy_signals = [s for s in signals.values() if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in signals.values() if s.signal_type == SignalType.SELL]

            if position is None:
                if buy_score >= buy_threshold and buy_signals:
                    combined.append(Signal(
                        date=date,
                        signal_type=SignalType.BUY,
                        price=buy_signals[0].price,
                        reason=f"[加权买入 {buy_score:.2f}] " + buy_signals[0].reason,
                        confidence=buy_score,
                    ))
                    position = "long"
            else:
                if sell_score >= sell_threshold and sell_signals:
                    combined.append(Signal(
                        date=date,
                        signal_type=SignalType.SELL,
                        price=sell_signals[0].price,
                        reason=f"[加权卖出 {sell_score:.2f}] " + sell_signals[0].reason,
                        confidence=sell_score,
                    ))
                    position = None

        logger.info(f"Ensemble (weighted): 生成 {len(combined)} 个信号")
        return combined


def create_default_ensemble(model=None, feature_extractor=None):
    """
    创建默认的组合策略

    组合：
    1. 趋势跟踪（牛市有效）
    2. 均值回归（熊市/震荡有效）
    3. ML模型（复杂模式）
    """
    strategies = [
        MaMacdRsiStrategy(),
        MeanReversionStrategy(),
    ]

    # 如果提供了ML模型，添加到组合
    if model and feature_extractor:
        ml_strategy = MLStrategy(
            model=model,
            feature_extractor=feature_extractor,
            threshold=0.01,
        )
        strategies.append(ml_strategy)

    # 创建组合（投票机制，至少2票同意）
    ensemble = EnsembleStrategy(
        strategies=strategies,
        method="voting",
        min_agree=2,
    )

    return ensemble
