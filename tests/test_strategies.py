"""
策略层单元测试

测试各种策略的信号生成
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


class TestMaMacdRsiStrategy:
    """MA+MACD+RSI策略测试"""

    def test_signal_generation(self):
        """测试信号生成"""
        print("\n测试MA+MACD+RSI策略信号生成")
        from src.strategies import MaMacdRsiStrategy
        from src.data.fetchers.mock import MockDataFetcher
        from src.data.storage.storage import DataStorage
        from src.data.api.data_manager import DataManager

        fetcher = MockDataFetcher(scenario="bull")
        storage = DataStorage()
        manager = DataManager(fetcher=fetcher, storage=storage)
        df = manager.get_daily_price("600000.SH", "20230101", "20231231")

        strategy = MaMacdRsiStrategy()
        signals = strategy.generate_signals(df)

        # 信号数量可能为0，这是正常的
        print(f"  ✅ 生成信号: {len(signals)}个")

        # 验证信号格式
        if signals:
            signal = signals[0]
            assert hasattr(signal, 'date')
            assert hasattr(signal, 'signal_type')
            assert hasattr(signal, 'price')
            print("  ✅ 信号格式正确")

    def test_strategy_with_bear_market(self):
        """测试熊市场景"""
        print("\n测试熊市场景")
        from src.strategies import MaMacdRsiStrategy
        from src.data.fetchers.mock import MockDataFetcher
        from src.data.storage.storage import DataStorage
        from src.data.api.data_manager import DataManager

        fetcher = MockDataFetcher(scenario="bear")
        storage = DataStorage()
        manager = DataManager(fetcher=fetcher, storage=storage)
        df = manager.get_daily_price("600000.SH", "20230101", "20231231")

        strategy = MaMacdRsiStrategy()
        signals = strategy.generate_signals(df)

        print(f"  ✅ 熊市信号: {len(signals)}个")


class TestStrategyBase:
    """策略基类测试"""

    def test_signal_types(self):
        """测试信号类型"""
        print("\n测试信号类型")
        from src.strategies.base import SignalType, Signal

        # 测试买入信号
        buy_signal = Signal(
            date="20230101",
            signal_type=SignalType.BUY,
            price=10.0,
            reason="测试买入",
        )
        assert buy_signal.signal_type == SignalType.BUY
        print("  ✅ 买入信号创建成功")

        # 测试卖出信号
        sell_signal = Signal(
            date="20230102",
            signal_type=SignalType.SELL,
            price=11.0,
            reason="测试卖出",
        )
        assert sell_signal.signal_type == SignalType.SELL
        print("  ✅ 卖出信号创建成功")


class TestBacktestScenarios:
    """回测场景测试"""

    def test_bull_scenario(self):
        """测试牛市回测"""
        print("\n测试牛市回测")
        from src.strategies import MaMacdRsiStrategy
        from src.backtesting.simple_backtester import SimpleBacktester
        from src.data.fetchers.mock import MockDataFetcher
        from src.data.storage.storage import DataStorage
        from src.data.api.data_manager import DataManager

        fetcher = MockDataFetcher(scenario="bull")
        storage = DataStorage()
        manager = DataManager(fetcher=fetcher, storage=storage)
        df = manager.get_daily_price("600000.SH", "20230101", "20231231")

        strategy = MaMacdRsiStrategy()
        backtester = SimpleBacktester(initial_capital=100000)
        result = backtester.run(strategy, df)

        assert result is not None
        print(f"  ✅ 牛市回测完成: 收益率{result.total_return*100:.2f}%")

    def test_bear_scenario(self):
        """测试熊市回测"""
        print("\n测试熊市回测")
        from src.strategies import MaMacdRsiStrategy
        from src.backtesting.simple_backtester import SimpleBacktester
        from src.data.fetchers.mock import MockDataFetcher
        from src.data.storage.storage import DataStorage
        from src.data.api.data_manager import DataManager

        fetcher = MockDataFetcher(scenario="bear")
        storage = DataStorage()
        manager = DataManager(fetcher=fetcher, storage=storage)
        df = manager.get_daily_price("600000.SH", "20230101", "20231231")

        strategy = MaMacdRsiStrategy()
        backtester = SimpleBacktester(initial_capital=100000)
        result = backtester.run(strategy, df)

        assert result is not None
        print(f"  ✅ 熊市回测完成: 收益率{result.total_return*100:.2f}%")

    def test_sideways_scenario(self):
        """测试震荡市回测"""
        print("\n测试震荡市回测")
        from src.strategies import MaMacdRsiStrategy
        from src.backtesting.simple_backtester import SimpleBacktester
        from src.data.fetchers.mock import MockDataFetcher
        from src.data.storage.storage import DataStorage
        from src.data.api.data_manager import DataManager

        fetcher = MockDataFetcher(scenario="sideways")
        storage = DataStorage()
        manager = DataManager(fetcher=fetcher, storage=storage)
        df = manager.get_daily_price("600000.SH", "20230101", "20231231")

        strategy = MaMacdRsiStrategy()
        backtester = SimpleBacktester(initial_capital=100000)
        result = backtester.run(strategy, df)

        assert result is not None
        print(f"  ✅ 震荡市回测完成: 收益率{result.total_return*100:.2f}%")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("策略层单元测试")
    print("=" * 60)

    # 策略测试
    ts = TestMaMacdRsiStrategy()
    ts.test_signal_generation()
    ts.test_strategy_with_bear_market()

    # 基类测试
    tsb = TestStrategyBase()
    tsb.test_signal_types()

    # 回测场景测试
    tbs = TestBacktestScenarios()
    tbs.test_bull_scenario()
    tbs.test_bear_scenario()
    tbs.test_sideways_scenario()

    print("\n" + "=" * 60)
    print("所有策略测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
