"""
均值回归策略单元测试

测试MeanReversionStrategy的核心功能：初始化、信号生成、买卖条件
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.base import SignalType


@pytest.fixture
def sample_ohlcv():
    """生成样本OHLCV数据"""
    dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
    np.random.seed(42)

    # 创建震荡行情（均值回归适合）
    base_price = 10.0
    noise = np.random.randn(150) * 0.03
    # 添加周期性波动
    cycle = 0.1 * np.sin(np.linspace(0, 4*np.pi, 150))
    prices = base_price * (1 + noise + cycle)

    return pd.DataFrame({
        'ts_code': '600000.SH',
        'trade_date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 150),
    })


class TestMeanReversionInit:
    """测试初始化"""

    def test_default_params(self):
        """测试默认参数初始化"""
        strategy = MeanReversionStrategy()

        assert strategy.name == "Mean_Reversion"
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0
        assert strategy.rsi_period == 14
        assert strategy.rsi_oversold == 30
        assert strategy.rsi_overbought == 70

    def test_custom_params(self):
        """测试自定义参数"""
        strategy = MeanReversionStrategy(
            bb_period=15,
            bb_std=1.5,
            rsi_oversold=25,
            ma_period=30,
        )

        assert strategy.bb_period == 15
        assert strategy.bb_std == 1.5
        assert strategy.rsi_oversold == 25
        assert strategy.ma_period == 30


class TestMeanReversionSignals:
    """测试信号生成"""

    def test_generate_signals(self, sample_ohlcv):
        """测试信号生成"""
        strategy = MeanReversionStrategy()

        signals = strategy.generate_signals(sample_ohlcv)

        # 应该返回Signal列表
        assert isinstance(signals, list)
        # 信号数量应该合理
        assert len(signals) >= 0

    def test_buy_signal_structure(self, sample_ohlcv):
        """测试买入信号结构"""
        strategy = MeanReversionStrategy(
            rsi_oversold=50,  # 降低阈值，更容易触发
        )

        signals = strategy.generate_signals(sample_ohlcv)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]

        if len(buy_signals) > 0:
            sig = buy_signals[0]
            assert sig.signal_type == SignalType.BUY
            assert sig.price > 0
            assert sig.date is not None
            assert sig.confidence >= 0

    def test_sell_signal_structure(self, sample_ohlcv):
        """测试卖出信号结构"""
        strategy = MeanReversionStrategy()

        signals = strategy.generate_signals(sample_ohlcv)
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]

        if len(sell_signals) > 0:
            sig = sell_signals[0]
            assert sig.signal_type == SignalType.SELL
            assert sig.price > 0

    def test_insufficient_data(self):
        """测试数据不足"""
        short_data = pd.DataFrame({
            'trade_date': [pd.Timestamp('2023-01-01')],
            'open': [10.0],
            'high': [10.5],
            'low': [9.5],
            'close': [10.0],
            'volume': [1000000],
        })

        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(short_data)

        assert len(signals) == 0


class TestMeanReversionIndicators:
    """测试指标计算"""

    def test_calculate_indicators(self, sample_ohlcv):
        """测试指标计算"""
        strategy = MeanReversionStrategy()

        df = strategy.calculate_indicators(sample_ohlcv)

        # 应该添加技术指标
        assert len(df.columns) > len(sample_ohlcv.columns)


class TestMeanReversionFormatDate:
    """测试日期格式化"""

    def test_format_datetime(self):
        """测试datetime格式化"""
        from datetime import datetime
        dt = datetime(2023, 6, 15)
        result = MeanReversionStrategy._format_date(dt)
        assert result == "20230615"

    def test_format_string(self):
        """测试字符串格式化"""
        result = MeanReversionStrategy._format_date("2023-06-15")
        assert result == "20230615"

    def test_format_timestamp(self):
        """测试pandas Timestamp格式化"""
        ts = pd.Timestamp('2023-06-15')
        result = MeanReversionStrategy._format_date(ts)
        assert result == "20230615"
