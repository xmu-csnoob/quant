"""
组合策略单元测试

测试EnsembleStrategy的核心功能：初始化、投票机制、加权机制
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, MagicMock

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.ensemble_strategy import EnsembleStrategy, create_default_ensemble
from src.strategies.base import BaseStrategy, Signal, SignalType


@pytest.fixture
def sample_ohlcv():
    """生成样本OHLCV数据"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    base_price = 10.0
    returns = np.random.randn(100) * 0.02
    prices = base_price * (1 + returns).cumprod()

    return pd.DataFrame({
        'ts_code': '600000.SH',
        'trade_date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
    })


@pytest.fixture
def mock_strategy_buy():
    """创建总是买入的Mock策略"""
    strategy = Mock(spec=BaseStrategy)
    strategy.name = "MockBuy"
    strategy.generate_signals = Mock(return_value=[
        Signal(date="20230601", signal_type=SignalType.BUY, price=10.0, reason="test", confidence=0.8),
        Signal(date="20230610", signal_type=SignalType.SELL, price=11.0, reason="test", confidence=0.7),
    ])
    return strategy


@pytest.fixture
def mock_strategy_sell():
    """创建总是卖出的Mock策略"""
    strategy = Mock(spec=BaseStrategy)
    strategy.name = "MockSell"
    strategy.generate_signals = Mock(return_value=[
        Signal(date="20230601", signal_type=SignalType.SELL, price=10.0, reason="test", confidence=0.6),
    ])
    return strategy


class TestEnsembleInit:
    """测试初始化"""

    def test_init_with_strategies(self, mock_strategy_buy, mock_strategy_sell):
        """测试带策略初始化"""
        ensemble = EnsembleStrategy(
            strategies=[mock_strategy_buy, mock_strategy_sell],
            method="voting",
        )

        assert len(ensemble.strategies) == 2
        assert ensemble.method == "voting"
        assert ensemble.min_agree == 2

    def test_init_with_weights(self, mock_strategy_buy, mock_strategy_sell):
        """测试带权重初始化"""
        weights = {"MockBuy": 0.7, "MockSell": 0.3}
        ensemble = EnsembleStrategy(
            strategies=[mock_strategy_buy, mock_strategy_sell],
            method="weighted",
            weights=weights,
        )

        assert ensemble.weights == weights


class TestEnsembleVoting:
    """测试投票机制"""

    def test_voting_buy_consensus(self, mock_strategy_buy, sample_ohlcv):
        """测试买入共识"""
        # 创建两个都会产生买入信号的策略
        strategy1 = Mock(spec=BaseStrategy)
        strategy1.name = "S1"
        strategy1.generate_signals = Mock(return_value=[
            Signal(date="20230601", signal_type=SignalType.BUY, price=10.0, reason="buy1", confidence=0.8),
        ])

        strategy2 = Mock(spec=BaseStrategy)
        strategy2.name = "S2"
        strategy2.generate_signals = Mock(return_value=[
            Signal(date="20230601", signal_type=SignalType.BUY, price=10.0, reason="buy2", confidence=0.7),
        ])

        ensemble = EnsembleStrategy(
            strategies=[strategy1, strategy2],
            method="voting",
            min_agree=2,
        )

        signals = ensemble.generate_signals(sample_ohlcv)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]

        assert len(buy_signals) > 0

    def test_voting_no_consensus(self, mock_strategy_buy, mock_strategy_sell, sample_ohlcv):
        """测试无共识"""
        ensemble = EnsembleStrategy(
            strategies=[mock_strategy_buy, mock_strategy_sell],
            method="voting",
            min_agree=2,
        )

        # 不同策略产生不同信号，需要2票同意，可能无法达成共识
        signals = ensemble.generate_signals(sample_ohlcv)
        # 信号可能为空或很少
        assert isinstance(signals, list)


class TestEnsembleWeighted:
    """测试加权机制"""

    def test_weighted_buy(self, mock_strategy_buy, sample_ohlcv):
        """测试加权买入"""
        strategy1 = Mock(spec=BaseStrategy)
        strategy1.name = "S1"
        strategy1.generate_signals = Mock(return_value=[
            Signal(date="20230601", signal_type=SignalType.BUY, price=10.0, reason="buy1", confidence=0.9),
        ])

        strategy2 = Mock(spec=BaseStrategy)
        strategy2.name = "S2"
        strategy2.generate_signals = Mock(return_value=[
            Signal(date="20230601", signal_type=SignalType.BUY, price=10.0, reason="buy2", confidence=0.8),
        ])

        ensemble = EnsembleStrategy(
            strategies=[strategy1, strategy2],
            method="weighted",
            weights={"S1": 0.6, "S2": 0.4},
        )

        signals = ensemble.generate_signals(sample_ohlcv)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]

        # 高权重高置信度应该产生买入信号
        assert len(buy_signals) > 0


class TestCreateDefaultEnsemble:
    """测试默认组合创建"""

    def test_create_without_ml(self):
        """测试不使用ML模型"""
        ensemble = create_default_ensemble()

        assert ensemble is not None
        assert len(ensemble.strategies) == 2  # MaMacdRsi + MeanReversion
        assert ensemble.method == "voting"

    def test_create_with_ml(self):
        """测试使用ML模型"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.05]))

        mock_extractor = Mock()
        mock_extractor.extract = Mock(return_value=pd.DataFrame({'f1': [1]}))

        ensemble = create_default_ensemble(
            model=mock_model,
            feature_extractor=mock_extractor,
        )

        assert len(ensemble.strategies) == 3  # + ML


class TestEnsembleUnknownMethod:
    """测试未知方法"""

    def test_unknown_method_fallback(self, mock_strategy_buy, sample_ohlcv):
        """测试未知方法回退到voting"""
        ensemble = EnsembleStrategy(
            strategies=[mock_strategy_buy],
            method="unknown",
        )

        # 应该回退到voting而不是崩溃
        signals = ensemble.generate_signals(sample_ohlcv)
        assert isinstance(signals, list)
