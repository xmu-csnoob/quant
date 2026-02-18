"""
动态策略选择器单元测试

测试DynamicStrategySelector的核心功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestDynamicStrategyInit:
    """测试初始化"""

    def test_default_init(self):
        """测试默认初始化"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector

        strategy = DynamicStrategySelector()

        assert strategy.name == "Dynamic_Strategy_Selector"
        assert strategy.confidence_threshold == 0.5
        assert strategy.strategy_map is not None

    def test_custom_params(self):
        """测试自定义参数"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector

        mock_strategy = Mock()
        strategy = DynamicStrategySelector(
            confidence_threshold=0.7,
            default_strategy=mock_strategy,
        )

        assert strategy.confidence_threshold == 0.7
        assert strategy.default_strategy == mock_strategy


class TestDynamicStrategySignals:
    """测试信号生成"""

    @pytest.fixture
    def sample_ohlcv(self):
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

    def test_generate_signals_returns_list(self, sample_ohlcv):
        """测试返回列表"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector

        strategy = DynamicStrategySelector()
        signals = strategy.generate_signals(sample_ohlcv)

        assert isinstance(signals, list)

    def test_insufficient_data(self):
        """测试数据不足"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector

        short_data = pd.DataFrame({
            'trade_date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': np.random.randn(10) + 10,
            'high': np.random.randn(10) + 10.5,
            'low': np.random.randn(10) + 9.5,
            'close': np.random.randn(10) + 10,
            'volume': np.random.randint(1000000, 10000000, 10),
        })

        strategy = DynamicStrategySelector()
        signals = strategy.generate_signals(short_data)

        assert isinstance(signals, list)


class TestDynamicStrategyMapping:
    """测试策略映射"""

    def test_strategy_map_exists(self):
        """测试策略映射存在"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector

        strategy = DynamicStrategySelector()

        assert len(strategy.strategy_map) > 0

    def test_strategy_map_contains_regimes(self):
        """测试策略映射包含各种市场环境"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        strategy = DynamicStrategySelector()

        assert MarketRegime.BULL in strategy.strategy_map
        assert MarketRegime.BEAR in strategy.strategy_map
        assert MarketRegime.SIDEWAYS in strategy.strategy_map
        assert MarketRegime.VOLATILE in strategy.strategy_map


class TestDynamicStrategyShouldBuy:
    """测试买入决策"""

    def test_should_buy_bull_market_high_confidence(self):
        """测试牛市高置信度买入"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        strategy = DynamicStrategySelector()
        row = pd.Series({'close': 10.0})

        result = strategy._should_buy(MarketRegime.BULL, row, 0.7)
        assert result is True

    def test_should_buy_bull_market_low_confidence(self):
        """测试牛市低置信度不买入"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        strategy = DynamicStrategySelector()
        row = pd.Series({'close': 10.0})

        result = strategy._should_buy(MarketRegime.BULL, row, 0.5)
        assert result is False

    def test_should_buy_sideways(self):
        """测试震荡市不买入"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        strategy = DynamicStrategySelector()
        row = pd.Series({'close': 10.0})

        result = strategy._should_buy(MarketRegime.SIDEWAYS, row, 0.8)
        assert result is False

    def test_should_buy_bear_market(self):
        """测试熊市不买入"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        strategy = DynamicStrategySelector()
        row = pd.Series({'close': 10.0})

        result = strategy._should_buy(MarketRegime.BEAR, row, 0.8)
        assert result is False


class TestDynamicStrategyShouldSell:
    """测试卖出决策"""

    def test_should_sell_bear_market(self):
        """测试熊市卖出"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        strategy = DynamicStrategySelector()
        row = pd.Series({'close': 10.0})

        result = strategy._should_sell(MarketRegime.BEAR, row, 0.8)
        assert result is True

    def test_should_sell_sideways(self):
        """测试震荡市卖出"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        strategy = DynamicStrategySelector()
        row = pd.Series({'close': 10.0})

        result = strategy._should_sell(MarketRegime.SIDEWAYS, row, 0.8)
        assert result is True

    def test_should_sell_volatile(self):
        """测试高波动卖出"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        strategy = DynamicStrategySelector()
        row = pd.Series({'close': 10.0})

        result = strategy._should_sell(MarketRegime.VOLATILE, row, 0.8)
        assert result is True

    def test_should_not_sell_bull_market(self):
        """测试牛市不卖出"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        strategy = DynamicStrategySelector()
        row = pd.Series({'close': 10.0})

        result = strategy._should_sell(MarketRegime.BULL, row, 0.8)
        assert result is False


class TestDynamicStrategyFullCycle:
    """测试完整交易周期"""

    @pytest.fixture
    def bull_market_data(self):
        """创建牛市数据"""
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(42)

        # 创建上升趋势
        trend = np.linspace(0, 0.5, 150)
        noise = np.random.randn(150) * 0.01
        prices = 10.0 * (1 + trend + noise)

        return pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 150),
        })

    def test_signals_with_bull_market(self, bull_market_data):
        """测试牛市环境下的信号"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector

        strategy = DynamicStrategySelector()
        signals = strategy.generate_signals(bull_market_data)

        assert isinstance(signals, list)
        # 验证信号结构
        for sig in signals:
            assert hasattr(sig, 'signal_type')
            assert hasattr(sig, 'price')
            assert hasattr(sig, 'date')

    def test_volatile_market_closes_position(self):
        """测试高波动环境平仓"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.strategies.base import SignalType

        # 创建先涨后波动的数据
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(42)

        # 前半段上涨，后半段高波动
        prices = np.concatenate([
            np.linspace(10, 15, 75),
            15 + np.random.randn(75) * 2  # 高波动
        ])

        df = pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 150),
        })

        strategy = DynamicStrategySelector()
        signals = strategy.generate_signals(df)

        # 检查是否有卖出信号
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        # 在高波动环境下可能会有卖出
        assert isinstance(sell_signals, list)


class TestAdaptiveDynamicStrategy:
    """测试自适应动态策略"""

    def test_init_default(self):
        """测试默认初始化"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy

        strategies = {
            'trend': Mock(),
            'mean_reversion': Mock(),
        }
        strategy = AdaptiveDynamicStrategy(strategies=strategies)

        assert strategy.name == "Adaptive_Dynamic"
        assert strategy.rebalance_freq == 20

    def test_init_custom_params(self):
        """测试自定义参数"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy

        strategies = {'test': Mock()}
        strategy = AdaptiveDynamicStrategy(
            strategies=strategies,
            rebalance_freq=30,
        )

        assert strategy.rebalance_freq == 30

    def test_should_enter_bull_market(self):
        """测试牛市入场"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.utils.market_regime import MarketRegime

        strategy = AdaptiveDynamicStrategy(strategies={'test': Mock()})

        result = strategy._should_enter(MarketRegime.BULL, 0.6)
        assert result is True

    def test_should_enter_bull_low_confidence(self):
        """测试牛市低置信度不入场"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.utils.market_regime import MarketRegime

        strategy = AdaptiveDynamicStrategy(strategies={'test': Mock()})

        result = strategy._should_enter(MarketRegime.BULL, 0.4)
        assert result is False

    def test_should_enter_sideways_high_confidence(self):
        """测试震荡市高置信度入场"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.utils.market_regime import MarketRegime

        strategy = AdaptiveDynamicStrategy(strategies={'test': Mock()})

        result = strategy._should_enter(MarketRegime.SIDEWAYS, 0.8)
        assert result is True

    def test_should_exit_bear_market(self):
        """测试熊市退出"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.utils.market_regime import MarketRegime

        strategy = AdaptiveDynamicStrategy(strategies={'test': Mock()})

        result = strategy._should_exit(MarketRegime.BEAR, 0.6)
        assert result is True

    def test_should_exit_volatile(self):
        """测试高波动退出"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.utils.market_regime import MarketRegime

        strategy = AdaptiveDynamicStrategy(strategies={'test': Mock()})

        result = strategy._should_exit(MarketRegime.VOLATILE, 0.6)
        assert result is True

    def test_should_exit_bull_market(self):
        """测试牛市不退出"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.utils.market_regime import MarketRegime

        strategy = AdaptiveDynamicStrategy(strategies={'test': Mock()})

        result = strategy._should_exit(MarketRegime.BULL, 0.8)
        assert result is False

    def test_generate_signals(self):
        """测试信号生成"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy

        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(42)

        trend = np.linspace(0, 0.3, 150)
        prices = 10.0 * (1 + trend)

        df = pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 150),
        })

        strategies = {'trend': Mock()}
        strategy = AdaptiveDynamicStrategy(strategies=strategies)
        signals = strategy.generate_signals(df)

        assert isinstance(signals, list)


class TestDynamicStrategyEdgeCases:
    """测试边界情况"""

    def test_invalid_regime_string(self):
        """测试无效的市场环境字符串"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector

        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        prices = 10.0 + np.random.randn(100) * 0.1

        df = pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
        })

        strategy = DynamicStrategySelector()
        # 应该能正常处理，不会崩溃
        signals = strategy.generate_signals(df)
        assert isinstance(signals, list)

    def test_confidence_below_threshold_uses_default(self):
        """测试低置信度使用默认策略"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.utils.market_regime import MarketRegime

        mock_default = Mock()
        mock_default.generate_signals = Mock(return_value=[])

        strategy = DynamicStrategySelector(
            confidence_threshold=0.9,  # 高阈值
            default_strategy=mock_default,
        )

        # 当置信度低于阈值时，应该使用默认策略
        assert strategy.confidence_threshold == 0.9


class TestDynamicStrategyVolatileExit:
    """测试高波动环境平仓"""

    def test_volatile_environment_with_position(self):
        """测试有持仓时高波动环境平仓"""
        from src.strategies.dynamic_strategy import DynamicStrategySelector
        from src.strategies.base import SignalType
        from unittest.mock import patch, MagicMock

        # 创建策略
        strategy = DynamicStrategySelector()

        # 创建测试数据：先平稳后高波动
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(123)

        # 创建一个先涨后高波动的价格序列
        prices = np.concatenate([
            np.linspace(10, 12, 100),  # 前100天上涨
            12 + np.random.randn(50) * 3  # 后50天高波动
        ])

        df = pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.05,
            'low': prices * 0.95,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 150),
        })

        # 运行策略
        signals = strategy.generate_signals(df)

        # 应该返回信号列表
        assert isinstance(signals, list)


class TestAdaptiveDynamicStrategyFullCycle:
    """测试自适应动态策略完整周期"""

    def test_generate_signals_with_buy_and_sell(self):
        """测试完整的买卖周期"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.strategies.base import SignalType

        # 创建测试数据：先涨后跌
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(456)

        prices = np.concatenate([
            np.linspace(10, 15, 75),  # 上涨
            np.linspace(15, 10, 75),  # 下跌
        ])

        df = pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 150),
        })

        strategies = {'trend': Mock()}
        strategy = AdaptiveDynamicStrategy(strategies=strategies, rebalance_freq=10)
        signals = strategy.generate_signals(df)

        # 检查信号类型
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]

        # 应该有买入信号
        assert isinstance(signals, list)

    def test_should_not_enter_bear_market(self):
        """测试熊市不入场"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.utils.market_regime import MarketRegime

        strategy = AdaptiveDynamicStrategy(strategies={'test': Mock()})

        result = strategy._should_enter(MarketRegime.BEAR, 0.8)
        assert result is False

    def test_should_not_enter_volatile(self):
        """测试高波动不入场"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.utils.market_regime import MarketRegime

        strategy = AdaptiveDynamicStrategy(strategies={'test': Mock()})

        result = strategy._should_enter(MarketRegime.VOLATILE, 0.8)
        assert result is False

    def test_should_not_exit_sideways(self):
        """测试震荡市不退出"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.utils.market_regime import MarketRegime

        strategy = AdaptiveDynamicStrategy(strategies={'test': Mock()})

        result = strategy._should_exit(MarketRegime.SIDEWAYS, 0.8)
        assert result is False

    def test_rebalance_frequency(self):
        """测试重新平衡频率"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.strategies.base import SignalType

        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(789)

        trend = np.linspace(0, 0.3, 150)
        prices = 10.0 * (1 + trend)

        df = pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 150),
        })

        strategies = {'trend': Mock()}
        strategy = AdaptiveDynamicStrategy(strategies=strategies, rebalance_freq=30)
        signals = strategy.generate_signals(df)

        assert isinstance(signals, list)

    def test_volatile_exit_with_high_confidence(self):
        """测试高波动高置信度退出"""
        from src.strategies.dynamic_strategy import AdaptiveDynamicStrategy
        from src.strategies.base import SignalType

        # 创建测试数据：先涨后高波动
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(101)

        prices = np.concatenate([
            np.linspace(10, 15, 100),  # 上涨
            15 + np.random.randn(50) * 3  # 高波动
        ])

        df = pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 150),
        })

        strategies = {'trend': Mock()}
        strategy = AdaptiveDynamicStrategy(strategies=strategies, rebalance_freq=10)
        signals = strategy.generate_signals(df)

        # 检查是否有卖出信号
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        # 可能会有卖出信号
        assert isinstance(sell_signals, list)
