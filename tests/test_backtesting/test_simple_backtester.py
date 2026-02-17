"""
回测引擎单元测试

测试SimpleBacktester的核心功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import date, datetime
from unittest.mock import Mock

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting.simple_backtester import (
    parse_date,
    get_next_trading_day,
    can_sell_after_t1,
    SimpleBacktester,
)
from src.strategies.base import Signal, SignalType


class TestDateUtils:
    """测试日期工具函数"""

    def test_parse_date_string_with_dash(self):
        """测试解析YYYY-MM-DD格式"""
        result = parse_date("2023-06-15")
        assert result == date(2023, 6, 15)

    def test_parse_date_string_no_dash(self):
        """测试解析YYYYMMDD格式"""
        result = parse_date("20230615")
        assert result == date(2023, 6, 15)

    def test_parse_date_object(self):
        """测试解析date对象"""
        d = date(2023, 6, 15)
        result = parse_date(d)
        assert result == d

    def test_get_next_trading_day_weekday(self):
        """测试工作日的下一交易日"""
        # 周一的下一交易日是周二
        monday = date(2023, 6, 12)  # Monday
        result = get_next_trading_day(monday)
        assert result.weekday() == 1  # Tuesday

    def test_get_next_trading_day_friday(self):
        """测试周五的下一交易日是周一"""
        friday = date(2023, 6, 16)  # Friday
        result = get_next_trading_day(friday)
        assert result.weekday() == 0  # Monday

    def test_can_sell_after_t1_same_day(self):
        """测试当天买入不能卖出"""
        buy_date = date(2023, 6, 12)
        sell_date = date(2023, 6, 12)
        assert can_sell_after_t1(buy_date, sell_date) is False

    def test_can_sell_after_t1_next_day(self):
        """测试次日可以卖出"""
        buy_date = date(2023, 6, 12)  # Monday
        sell_date = date(2023, 6, 13)  # Tuesday
        assert can_sell_after_t1(buy_date, sell_date) is True


class TestSimpleBacktesterInit:
    """测试回测器初始化"""

    def test_default_init(self):
        """测试默认初始化"""
        backtester = SimpleBacktester()

        assert backtester.initial_capital > 0

    def test_custom_init(self):
        """测试自定义初始化"""
        backtester = SimpleBacktester(initial_capital=500000)

        assert backtester.initial_capital == 500000


class TestSimpleBacktesterRun:
    """测试回测运行"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        return pd.DataFrame({
            'trade_date': dates,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100),
        })

    @pytest.fixture
    def mock_strategy(self):
        """创建Mock策略"""
        strategy = Mock()
        strategy.name = "MockStrategy"
        strategy.generate_signals = Mock(return_value=[
            Signal(date="20230201", signal_type=SignalType.BUY, price=100.0, reason="test"),
            Signal(date="20230210", signal_type=SignalType.SELL, price=105.0, reason="test"),
        ])
        return strategy

    def test_run_backtest(self, sample_data, mock_strategy):
        """测试运行回测"""
        backtester = SimpleBacktester(initial_capital=1000000)

        # run(strategy, df) - 注意参数顺序
        result = backtester.run(mock_strategy, sample_data)

        # 应该返回BacktestResult
        assert result is not None

    def test_run_with_empty_signals(self, sample_data):
        """测试空信号"""
        strategy = Mock()
        strategy.name = "EmptyStrategy"
        strategy.generate_signals = Mock(return_value=[])

        backtester = SimpleBacktester()
        result = backtester.run(strategy, sample_data)

        assert result is not None


class TestSimpleBacktesterStats:
    """测试回测统计"""

    def test_backtester_creation(self):
        """测试回测器创建"""
        backtester = SimpleBacktester()

        # 验证基本属性
        assert backtester.initial_capital > 0
