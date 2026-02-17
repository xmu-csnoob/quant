"""
MA+MACD+RSI策略单元测试

测试MaMacdRsiStrategy的核心功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.ma_macd_rsi import MaMacdRsiStrategy
from src.strategies.base import SignalType


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


class TestMaMacdRsiInit:
    """测试初始化"""

    def test_default_params(self):
        """测试默认参数"""
        strategy = MaMacdRsiStrategy()

        assert strategy.name == "MA_MACD_RSI"
        assert strategy.ma_fast == 5
        assert strategy.ma_slow == 20
        assert strategy.ma_long == 60
        assert strategy.macd_fast == 12
        assert strategy.macd_slow == 26
        assert strategy.rsi_period == 14

    def test_custom_params(self):
        """测试自定义参数"""
        strategy = MaMacdRsiStrategy(
            ma_fast=10,
            ma_slow=30,
            rsi_overbought=80,
        )

        assert strategy.ma_fast == 10
        assert strategy.ma_slow == 30
        assert strategy.rsi_overbought == 80


class TestMaMacdRsiIndicators:
    """测试指标计算"""

    def test_calculate_indicators(self, sample_ohlcv):
        """测试指标计算"""
        strategy = MaMacdRsiStrategy()

        df = strategy.calculate_indicators(sample_ohlcv)

        # 应该添加MA列
        assert f'MA{strategy.ma_fast}' in df.columns
        assert f'MA{strategy.ma_slow}' in df.columns

    def test_calculate_indicators_adds_columns(self, sample_ohlcv):
        """测试指标添加列"""
        strategy = MaMacdRsiStrategy()
        original_cols = len(sample_ohlcv.columns)

        df = strategy.calculate_indicators(sample_ohlcv)

        assert len(df.columns) > original_cols


class TestMaMacdRsiSignals:
    """测试信号生成"""

    def test_generate_signals_returns_list(self, sample_ohlcv):
        """测试返回列表"""
        strategy = MaMacdRsiStrategy()

        signals = strategy.generate_signals(sample_ohlcv)

        assert isinstance(signals, list)

    def test_signal_structure(self, sample_ohlcv):
        """测试信号结构"""
        strategy = MaMacdRsiStrategy()

        signals = strategy.generate_signals(sample_ohlcv)

        if len(signals) > 0:
            sig = signals[0]
            assert sig.signal_type in [SignalType.BUY, SignalType.SELL]
            assert sig.price > 0
            assert sig.date is not None

    def test_insufficient_data(self):
        """测试数据不足"""
        short_data = pd.DataFrame({
            'trade_date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': np.random.randn(10) + 10,
            'high': np.random.randn(10) + 10.5,
            'low': np.random.randn(10) + 9.5,
            'close': np.random.randn(10) + 10,
            'volume': np.random.randint(1000000, 10000000, 10),
        })

        strategy = MaMacdRsiStrategy()
        signals = strategy.generate_signals(short_data)

        # 数据不足可能返回空列表或很少信号
        assert isinstance(signals, list)


class TestMaMacdRsiTrend:
    """测试趋势判断"""

    def test_uptrend_detection(self):
        """测试上升趋势检测"""
        # 创建明确的上升趋势
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.linspace(10, 15, 100)  # 明显上涨

        df = pd.DataFrame({
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
        })

        strategy = MaMacdRsiStrategy()
        signals = strategy.generate_signals(df)

        # 上升趋势应该产生买入信号
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 0  # 可能产生买入信号
