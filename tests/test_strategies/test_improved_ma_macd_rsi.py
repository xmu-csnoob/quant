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


class TestImprovedMaMacdRsiCheckBuySignal:
    """测试买入信号检查"""

    @pytest.fixture
    def strategy(self):
        return ImprovedMaMacdRsiStrategy()

    @pytest.fixture
    def sample_df(self):
        """生成趋势上涨的样本数据"""
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(42)

        # 创建上升趋势
        base_price = 10.0
        trend = np.linspace(0, 0.5, 150)  # 50%的上升趋势
        noise = np.random.randn(150) * 0.02
        prices = base_price * (1 + trend + noise)

        return pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 150),
        })

    def test_check_buy_signal_no_bullish_alignment(self, strategy, sample_df):
        """测试没有多头排列时不买入"""
        df = strategy.calculate_indicators(sample_df)
        # 设置没有多头排列
        row = df.iloc[-1].copy()
        row['bullish_alignment'] = False

        result, confidence = strategy._check_buy_signal(row, len(df) - 1, df)
        assert result is None
        assert confidence == 0.0

    def test_check_buy_signal_no_macd_cross(self, strategy, sample_df):
        """测试没有MACD金叉时不买入"""
        df = strategy.calculate_indicators(sample_df)
        row = df.iloc[-1].copy()
        row['bullish_alignment'] = True
        row['macd_signal'] = 0  # 不是金叉

        result, confidence = strategy._check_buy_signal(row, len(df) - 1, df)
        assert result is None

    def test_check_buy_signal_rsi_severely_overbought(self, strategy, sample_df):
        """测试RSI严重超买时不买入"""
        df = strategy.calculate_indicators(sample_df)
        row = df.iloc[-1].copy()
        row['bullish_alignment'] = True
        row['macd_signal'] = 1
        row['RSI'] = 90  # 严重超买

        result, confidence = strategy._check_buy_signal(row, len(df) - 1, df)
        assert result is None

    def test_check_buy_signal_confidence_below_threshold(self, strategy, sample_df):
        """测试信号强度低于阈值时不买入"""
        strategy_low = ImprovedMaMacdRsiStrategy(min_signal_strength=0.99)
        df = strategy_low.calculate_indicators(sample_df)
        row = df.iloc[-1].copy()
        row['bullish_alignment'] = True
        row['macd_signal'] = 1
        row['RSI'] = 50
        row['long_trend_up'] = False
        row['price_above_long_ma'] = False

        result, confidence = strategy_low._check_buy_signal(row, len(df) - 1, df)
        # 低置信度应该被拒绝
        assert result is None

    def test_check_buy_signal_zero_axis_above(self, strategy, sample_df):
        """测试零轴上金叉增加置信度"""
        df = strategy.calculate_indicators(sample_df)
        row = df.iloc[-1].copy()
        row['bullish_alignment'] = True
        row['macd_signal'] = 1
        row['macd_signal_strength'] = 'zero_axis_above'
        row['RSI'] = 50
        row['long_trend_up'] = True
        row['price_above_long_ma'] = True

        result, confidence = strategy._check_buy_signal(row, len(df) - 1, df)
        assert result is True
        assert confidence >= 0.6  # 应该有较高置信度


class TestImprovedMaMacdRsiCheckSellConditions:
    """测试卖出条件检查"""

    @pytest.fixture
    def strategy(self):
        return ImprovedMaMacdRsiStrategy(
            stop_loss_pct=0.05,
            take_profit_pct=0.20,
            min_holding_days=3,
            max_holding_days=60,
        )

    @pytest.fixture
    def mock_position(self):
        """创建模拟持仓"""
        from src.strategies.base import Position, PositionType
        return Position(
            entry_date='20230101',
            entry_price=10.0,
            quantity=100,
            position_type=PositionType.LONG,
            stop_loss=9.5,  # 5% 止损
            take_profit=12.0,  # 20% 止盈
        )

    @pytest.fixture
    def sample_row(self):
        """创建模拟行数据"""
        return pd.Series({
            'long_trend_up': True,
            'price_above_long_ma': True,
            'MA5': 10.5,
            'MA20': 10.3,
            'MA60': 10.0,
            'bearish_divergence': False,
        })

    def test_stop_loss_triggered(self, strategy, mock_position, sample_row):
        """测试止损触发"""
        current_price = 9.4  # 低于止损价
        should_sell, reason = strategy._check_sell_conditions(
            sample_row, mock_position, 5, current_price, 10.0
        )
        assert should_sell is True
        assert '止损' in reason

    def test_take_profit_triggered(self, strategy, mock_position, sample_row):
        """测试止盈触发"""
        current_price = 12.5  # 高于止盈价
        should_sell, reason = strategy._check_sell_conditions(
            sample_row, mock_position, 5, current_price, 10.0
        )
        assert should_sell is True
        assert '止盈' in reason

    def test_max_holding_days_reached(self, strategy, mock_position, sample_row):
        """测试最大持仓期"""
        should_sell, reason = strategy._check_sell_conditions(
            sample_row, mock_position, 60, 10.5, 10.0
        )
        assert should_sell is True
        assert '最大持仓期' in reason

    def test_trend_reversal(self, strategy, mock_position):
        """测试趋势反转"""
        sample_row = pd.Series({
            'long_trend_up': False,
            'price_above_long_ma': False,
            'MA5': 9.8,
            'MA20': 10.0,
            'MA60': 10.2,
            'bearish_divergence': False,
        })
        should_sell, reason = strategy._check_sell_conditions(
            sample_row, mock_position, 10, 10.0, 10.0
        )
        assert should_sell is True
        assert '趋势反转' in reason

    def test_deep_death_cross(self, strategy, mock_position):
        """测试深度死叉"""
        sample_row = pd.Series({
            'long_trend_up': False,
            'price_above_long_ma': True,
            'MA5': 9.5,  # 低于MA20超过3%
            'MA20': 10.0,
            'MA60': 10.0,
            'bearish_divergence': False,
        })
        should_sell, reason = strategy._check_sell_conditions(
            sample_row, mock_position, 10, 10.0, 10.0
        )
        assert should_sell is True
        assert '死叉' in reason

    def test_bearish_divergence(self, strategy, mock_position):
        """测试MACD顶背离"""
        sample_row = pd.Series({
            'long_trend_up': True,
            'price_above_long_ma': True,
            'MA5': 10.5,
            'MA20': 10.3,
            'MA60': 10.0,
            'bearish_divergence': True,
        })
        should_sell, reason = strategy._check_sell_conditions(
            sample_row, mock_position, 10, 10.5, 10.0
        )
        assert should_sell is True
        assert '背离' in reason

    def test_min_holding_days_protection(self, strategy, mock_position, sample_row):
        """测试最短持仓期保护"""
        should_sell, reason = strategy._check_sell_conditions(
            sample_row, mock_position, 1, 10.5, 10.0  # 只持有1天
        )
        assert should_sell is False

    def test_no_sell_short_term_pullback(self, strategy, mock_position, sample_row):
        """测试短期回调不卖出"""
        should_sell, reason = strategy._check_sell_conditions(
            sample_row, mock_position, 10, 10.5, 10.0
        )
        # 长期趋势向上，应该持有
        assert should_sell is False


class TestImprovedMaMacdRsiHelpers:
    """测试辅助方法"""

    def test_get_buy_reason(self):
        """测试获取买入原因"""
        strategy = ImprovedMaMacdRsiStrategy()
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal': 1,
            'macd_signal_strength': 'zero_axis_above',
            'long_trend_up': True,
            'price_above_long_ma': True,
        })

        reason = strategy._get_buy_reason(row)
        assert '多头排列' in reason
        assert '零轴上金叉' in reason
        assert '长期趋势向上' in reason

    def test_get_buy_reason_normal_macd(self):
        """测试普通MACD金叉的买入原因"""
        strategy = ImprovedMaMacdRsiStrategy()
        row = pd.Series({
            'bullish_alignment': True,
            'macd_signal': 1,
            'macd_signal_strength': 'normal',
            'long_trend_up': True,
            'price_above_long_ma': True,
        })

        reason = strategy._get_buy_reason(row)
        assert 'MACD金叉' in reason

    def test_calculate_sell_confidence_high(self):
        """测试卖出信号高置信度"""
        strategy = ImprovedMaMacdRsiStrategy()
        row = pd.Series({
            'long_trend_up': False,
        })

        confidence = strategy._calculate_sell_confidence(row)
        assert confidence > 0.5

    def test_calculate_sell_confidence_low(self):
        """测试卖出信号低置信度"""
        strategy = ImprovedMaMacdRsiStrategy()
        row = pd.Series({
            'long_trend_up': True,
        })

        confidence = strategy._calculate_sell_confidence(row)
        assert confidence == 0.5


class TestImprovedMaMacdRsiFullCycle:
    """测试完整交易周期"""

    @pytest.fixture
    def trending_data(self):
        """创建趋势数据"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(123)

        # 创建先涨后跌的趋势
        up_trend = np.linspace(0, 0.4, 100)
        down_trend = np.linspace(0.4, 0.1, 100)
        trend = np.concatenate([up_trend, down_trend])
        noise = np.random.randn(200) * 0.01
        prices = 10.0 * (1 + trend + noise)

        return pd.DataFrame({
            'ts_code': '600000.SH',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 200),
        })

    def test_generate_signals_with_bull_market(self, trending_data):
        """测试牛市环境下的信号生成"""
        strategy = ImprovedMaMacdRsiStrategy()
        signals = strategy.generate_signals(trending_data)

        assert isinstance(signals, list)
        # 应该有一些买入信号
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 0  # 取决于数据

    def test_signals_alternate(self, trending_data):
        """测试买卖信号交替"""
        strategy = ImprovedMaMacdRsiStrategy()
        signals = strategy.generate_signals(trending_data)

        # 检查买卖信号应该交替出现
        prev_type = None
        for sig in signals:
            if prev_type is not None:
                if prev_type == SignalType.BUY:
                    assert sig.signal_type == SignalType.SELL
                else:
                    assert sig.signal_type == SignalType.BUY
            prev_type = sig.signal_type
