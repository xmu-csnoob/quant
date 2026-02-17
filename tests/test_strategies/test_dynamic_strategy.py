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
