"""
改进版MA+MACD+RSI策略单元测试

测试ImprovedMaMacdRsiStrategy的核心功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.improved_ma_macd_rsi import ImprovedMaMacdRsiStrategy
from src.strategies.base import SignalType


@pytest.fixture
def sample_ohlcv():
    """生成样本OHLCV数据"""
    dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
    np.random.seed(42)

    base_price = 10.0
    returns = np.random.randn(150) * 0.02
    prices = base_price * (1 + returns).cumprod()

    return pd.DataFrame({
        'ts_code': '600000.SH',
        'trade_date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 150),
    })


class TestImprovedMaMacdRsiInit:
    """测试初始化"""

    def test_default_params(self):
        """测试默认参数"""
        strategy = ImprovedMaMacdRsiStrategy()

        assert strategy.name == "Improved_MA_MACD_RSI"
        assert strategy.ma_fast == 5
        assert strategy.ma_slow == 20
        assert strategy.ma_long == 60
        assert strategy.stop_loss_pct == 0.05
        assert strategy.take_profit_pct == 0.20

    def test_custom_params(self):
        """测试自定义参数"""
        strategy = ImprovedMaMacdRsiStrategy(
            ma_fast=10,
            stop_loss_pct=0.08,
            take_profit_pct=0.15,
            min_holding_days=5,
            max_holding_days=30,
        )

        assert strategy.ma_fast == 10
        assert strategy.stop_loss_pct == 0.08
        assert strategy.take_profit_pct == 0.15
        assert strategy.min_holding_days == 5
        assert strategy.max_holding_days == 30


class TestImprovedMaMacdRsiIndicators:
    """测试指标计算"""

    def test_calculate_indicators(self, sample_ohlcv):
        """测试指标计算"""
        strategy = ImprovedMaMacdRsiStrategy()

        df = strategy.calculate_indicators(sample_ohlcv)

        # 应该添加MA列
        assert f'MA{strategy.ma_fast}' in df.columns
        assert f'MA{strategy.ma_slow}' in df.columns
        assert f'MA{strategy.ma_long}' in df.columns

    def test_calculate_indicators_adds_columns(self, sample_ohlcv):
        """测试指标添加列"""
        strategy = ImprovedMaMacdRsiStrategy()
        original_cols = len(sample_ohlcv.columns)

        df = strategy.calculate_indicators(sample_ohlcv)

        assert len(df.columns) > original_cols


class TestImprovedMaMacdRsiSignals:
    """测试信号生成"""

    def test_generate_signals_returns_list(self, sample_ohlcv):
        """测试返回列表"""
        strategy = ImprovedMaMacdRsiStrategy()

        signals = strategy.generate_signals(sample_ohlcv)

        assert isinstance(signals, list)

    def test_signal_structure(self, sample_ohlcv):
        """测试信号结构"""
        strategy = ImprovedMaMacdRsiStrategy()

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

        strategy = ImprovedMaMacdRsiStrategy()
        signals = strategy.generate_signals(short_data)

        assert isinstance(signals, list)


class TestImprovedMaMacdRsiRiskManagement:
    """测试风险管理参数"""

    def test_stop_loss_config(self):
        """测试止损配置"""
        strategy = ImprovedMaMacdRsiStrategy(stop_loss_pct=0.08)

        assert strategy.stop_loss_pct == 0.08

    def test_take_profit_config(self):
        """测试止盈配置"""
        strategy = ImprovedMaMacdRsiStrategy(take_profit_pct=0.25)

        assert strategy.take_profit_pct == 0.25

    def test_holding_period_config(self):
        """测试持仓周期配置"""
        strategy = ImprovedMaMacdRsiStrategy(
            min_holding_days=5,
            max_holding_days=30,
        )

        assert strategy.min_holding_days == 5
        assert strategy.max_holding_days == 30


class TestImprovedMaMacdRsiSignalQuality:
    """测试信号质量"""

    def test_min_signal_strength(self):
        """测试最低信号强度"""
        strategy = ImprovedMaMacdRsiStrategy(min_signal_strength=0.7)

        assert strategy.min_signal_strength == 0.7

    def test_high_signal_strength(self):
        """测试高信号强度要求"""
        strategy = ImprovedMaMacdRsiStrategy(min_signal_strength=0.9)

        # 高信号强度可能导致信号减少
        assert strategy.min_signal_strength == 0.9
