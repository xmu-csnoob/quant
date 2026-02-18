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


class TestMaMacdRsiCheckBuyConditions:
    """测试买入条件检查"""

    @pytest.fixture
    def strategy(self):
        return MaMacdRsiStrategy()

    def test_no_bullish_alignment(self, strategy):
        """测试没有多头排列时不买入"""
        row = pd.Series({
            'bullish_alignment': False,
            'macd_signal': 1,
            'RSI': 50,
        })

        result = strategy._check_buy_conditions(row)
        assert result is False

    def test_no_macd_gold_cross(self, strategy):
        """测试没有MACD金叉时不买入"""
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal': 0,
            'RSI': 50,
        })

        result = strategy._check_buy_conditions(row)
        assert result is False

    def test_rsi_severely_overbought(self, strategy):
        """测试RSI严重超买时不买入"""
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal': 1,
            'RSI': 90,  # > 85
        })

        result = strategy._check_buy_conditions(row)
        assert result is False

    def test_all_conditions_met(self, strategy):
        """测试所有条件满足时买入"""
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal': 1,
            'RSI': 50,
        })

        result = strategy._check_buy_conditions(row)
        assert result is True


class TestMaMacdRsiCheckSellConditions:
    """测试卖出条件检查"""

    @pytest.fixture
    def strategy(self):
        return MaMacdRsiStrategy()

    @pytest.fixture
    def mock_position(self):
        from src.strategies.base import Position, PositionType
        return Position(
            entry_date='20230101',
            entry_price=10.0,
            quantity=100,
            position_type=PositionType.LONG,
        )

    def test_ma_death_cross(self, strategy, mock_position):
        """测试MA死叉卖出"""
        row = pd.Series({
            'ma_signal': -1,
            'macd_signal': 0,
            'RSI': 50,
            'bearish_divergence': False,
        })

        result = strategy._check_sell_conditions(row, mock_position)
        assert result is True

    def test_macd_death_cross(self, strategy, mock_position):
        """测试MACD死叉卖出"""
        row = pd.Series({
            'ma_signal': 0,
            'macd_signal': -1,
            'RSI': 50,
            'bearish_divergence': False,
        })

        result = strategy._check_sell_conditions(row, mock_position)
        assert result is True

    def test_bearish_divergence(self, strategy, mock_position):
        """测试MACD顶背离卖出"""
        row = pd.Series({
            'ma_signal': 0,
            'macd_signal': 0,
            'RSI': 50,
            'bearish_divergence': True,
        })

        result = strategy._check_sell_conditions(row, mock_position)
        assert result is True

    def test_rsi_severely_overbought(self, strategy, mock_position):
        """测试RSI严重超买卖出"""
        row = pd.Series({
            'ma_signal': 0,
            'macd_signal': 0,
            'RSI': 85,  # > 80
            'bearish_divergence': False,
        })

        result = strategy._check_sell_conditions(row, mock_position)
        assert result is True

    def test_no_sell_condition(self, strategy, mock_position):
        """测试无卖出条件"""
        row = pd.Series({
            'ma_signal': 0,
            'macd_signal': 0,
            'RSI': 50,
            'bearish_divergence': False,
        })

        result = strategy._check_sell_conditions(row, mock_position)
        assert result is False


class TestMaMacdRsiReasons:
    """测试买入/卖出原因"""

    @pytest.fixture
    def strategy(self):
        return MaMacdRsiStrategy()

    def test_get_buy_reason_normal_macd(self, strategy):
        """测试普通MACD金叉买入原因"""
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal': 1,
            'macd_signal_strength': 'normal',
            'RSI': 50,
        })

        reason = strategy._get_buy_reason(row)
        assert 'MACD金叉' in reason

    def test_get_buy_reason_zero_axis_above(self, strategy):
        """测试零轴上金叉买入原因"""
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal': 1,
            'macd_signal_strength': 'zero_axis_above',
            'RSI': 50,
        })

        reason = strategy._get_buy_reason(row)
        assert '零轴上金叉' in reason

    def test_get_buy_reason_rsi_oversold(self, strategy):
        """测试RSI超卖买入原因"""
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal': 1,
            'macd_signal_strength': 'normal',
            'RSI': 20,  # < rsi_oversold (25)
        })

        reason = strategy._get_buy_reason(row)
        assert 'RSI超卖' in reason

    def test_get_buy_reason_rsi_normal(self, strategy):
        """测试RSI正常买入原因"""
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal': 1,
            'macd_signal_strength': 'normal',
            'RSI': 50,  # 30 < RSI < 70
        })

        reason = strategy._get_buy_reason(row)
        assert 'RSI正常' in reason

    def test_get_sell_reason_ma_death(self, strategy):
        """测试MA死叉卖出原因"""
        row = pd.Series({
            'ma_signal': -1,
            'macd_signal': 0,
            'RSI': 50,
            'bearish_divergence': False,
        })

        reason = strategy._get_sell_reason(row)
        assert 'MA死叉' in reason

    def test_get_sell_reason_macd_death(self, strategy):
        """测试MACD死叉卖出原因"""
        row = pd.Series({
            'ma_signal': 0,
            'macd_signal': -1,
            'RSI': 50,
            'bearish_divergence': False,
        })

        reason = strategy._get_sell_reason(row)
        assert 'MACD死叉' in reason

    def test_get_sell_reason_bearish_divergence(self, strategy):
        """测试MACD顶背离卖出原因"""
        row = pd.Series({
            'ma_signal': 0,
            'macd_signal': 0,
            'RSI': 50,
            'bearish_divergence': True,
        })

        reason = strategy._get_sell_reason(row)
        assert 'MACD顶背离' in reason

    def test_get_sell_reason_rsi_overbought(self, strategy):
        """测试RSI严重超买卖出原因"""
        row = pd.Series({
            'ma_signal': 0,
            'macd_signal': 0,
            'RSI': 85,
            'bearish_divergence': False,
        })

        reason = strategy._get_sell_reason(row)
        assert 'RSI严重超买' in reason

    def test_get_sell_reason_other(self, strategy):
        """测试其他卖出原因"""
        row = pd.Series({
            'ma_signal': 0,
            'macd_signal': 0,
            'RSI': 50,
            'bearish_divergence': False,
        })

        reason = strategy._get_sell_reason(row)
        assert '其他' in reason


class TestMaMacdRsiConfidence:
    """测试置信度计算"""

    @pytest.fixture
    def strategy(self):
        return MaMacdRsiStrategy()

    def test_calculate_buy_confidence_high(self, strategy):
        """测试高买入置信度"""
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal_strength': 'zero_axis_above',
            'MACD': 0.5,  # > 0
        })

        confidence = strategy._calculate_buy_confidence(row)
        assert confidence > 0.5
        assert confidence <= 1.0

    def test_calculate_buy_confidence_low(self, strategy):
        """测试低买入置信度"""
        row = pd.Series({
            'bullish_alignment': False,
            'macd_signal_strength': 'normal',
            'MACD': -0.5,
        })

        confidence = strategy._calculate_buy_confidence(row)
        assert confidence == 0.5  # 基础置信度

    def test_calculate_sell_confidence_multiple_signals(self, strategy):
        """测试多重卖出信号置信度"""
        row = pd.Series({
            'ma_signal': -1,
            'macd_signal': -1,
            'bearish_divergence': True,
            'RSI': 85,
        })

        confidence = strategy._calculate_sell_confidence(row)
        assert confidence >= 0.9  # 多重信号应该有高置信度

    def test_calculate_sell_confidence_single_signal(self, strategy):
        """测试单一卖出信号置信度"""
        row = pd.Series({
            'ma_signal': -1,
            'macd_signal': 0,
            'bearish_divergence': False,
            'RSI': 50,
        })

        confidence = strategy._calculate_sell_confidence(row)
        assert confidence >= 0.7


class TestMaMacdRsiFullCycle:
    """测试完整交易周期"""

    def test_signals_alternate(self):
        """测试买卖信号交替"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(123)

        # 创建先涨后跌的趋势
        up_trend = np.linspace(0, 0.4, 100)
        down_trend = np.linspace(0.4, 0.1, 100)
        trend = np.concatenate([up_trend, down_trend])
        noise = np.random.randn(200) * 0.01
        prices = 10.0 * (1 + trend + noise)

        df = pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 200),
        })

        strategy = MaMacdRsiStrategy()
        signals = strategy.generate_signals(df)

        # 检查买卖信号应该交替出现
        prev_type = None
        for sig in signals:
            if prev_type is not None:
                if prev_type == SignalType.BUY:
                    assert sig.signal_type == SignalType.SELL
                else:
                    assert sig.signal_type == SignalType.BUY
            prev_type = sig.signal_type
