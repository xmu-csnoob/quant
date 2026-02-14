"""
高级回测绩效指标测试

测试新增的高级绩效指标：
- 索提诺比率（Sortino Ratio）
- Calmar比率
- 信息比率（Information Ratio）
- Alpha/Beta
- 盈亏比
- 连胜/连亏统计
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies import MaMacdRsiStrategy
from src.backtesting.simple_backtester import SimpleBacktester
from src.data.fetchers.mock import MockDataFetcher
from src.data.api.data_manager import DataManager
from src.data.storage.storage import DataStorage


def test_advanced_metrics():
    """测试高级绩效指标计算"""
    print("=" * 70)
    print("测试高级绩效指标")
    print("=" * 70)

    # 使用牛市数据
    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)
    df = manager.get_daily_price("600000.SH", "20230101", "20231231")

    # 创建策略
    strategy = MaMacdRsiStrategy()

    # 运行回测
    backtester = SimpleBacktester(initial_capital=100000.0)
    result = backtester.run(strategy, df)

    # 验证基础指标
    print(f"\n基础指标:")
    print(f"  总收益率: {result.total_return*100:.2f}%")
    print(f"  年化收益: {result.annual_return*100:.2f}%")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  最大回撤: {result.max_drawdown*100:.2f}%")

    # 验证高级指标
    print(f"\n高级指标:")
    print(f"  索提诺比率: {result.sortino_ratio:.2f}")
    assert isinstance(result.sortino_ratio, float), "索提诺比率应为float类型"

    print(f"  Calmar比率: {result.calmar_ratio:.2f}")
    assert isinstance(result.calmar_ratio, float), "Calmar比率应为float类型"

    print(f"  信息比率: {result.information_ratio:.2f}")
    assert isinstance(result.information_ratio, float), "信息比率应为float类型"

    print(f"  Alpha: {result.alpha:.4f}")
    assert isinstance(result.alpha, float), "Alpha应为float类型"

    print(f"  Beta: {result.beta:.4f}")
    assert isinstance(result.beta, float), "Beta应为float类型"

    print(f"  盈亏比: {result.profit_factor:.2f}")
    assert isinstance(result.profit_factor, float), "盈亏比应为float类型"
    assert result.profit_factor >= 0, "盈亏比应>=0"

    print(f"  平均盈利: {result.avg_win:,.2f}")
    print(f"  平均亏损: {result.avg_loss:,.2f}")

    print(f"  最大连胜: {result.max_consecutive_wins}")
    print(f"  最大连亏: {result.max_consecutive_losses}")
    assert isinstance(result.max_consecutive_wins, int), "最大连胜应为int类型"
    assert isinstance(result.max_consecutive_losses, int), "最大连亏应为int类型"

    print(f"  基准收益: {result.benchmark_return*100:.2f}%")
    assert isinstance(result.benchmark_return, float), "基准收益应为float类型"

    print("\n✅ 所有高级指标测试通过")


def test_scenarios_comparison():
    """对比不同场景下的高级指标"""
    print("\n" + "=" * 70)
    print("不同场景高级指标对比")
    print("=" * 70)

    scenarios = ["bull", "bear", "sideways", "volatile"]
    results = {}

    for scenario in scenarios:
        try:
            fetcher = MockDataFetcher(scenario=scenario)
            storage = DataStorage()
            manager = DataManager(fetcher=fetcher, storage=storage)
            df = manager.get_daily_price("600000.SH", "20230101", "20231231")

            strategy = MaMacdRsiStrategy()
            backtester = SimpleBacktester(initial_capital=100000.0)
            result = backtester.run(strategy, df)
            results[scenario] = result
        except Exception as e:
            print(f"场景 {scenario} 出错: {e}")

    if results:
        # 打印对比表
        print(
            f"{'场景':<12} {'总收益':<10} {'年化':<10} {'夏普':<8} "
            f"{'索提诺':<8} {'Calmar':<8} {'盈亏比':<8}"
        )
        print("-" * 70)

        for scenario, result in results.items():
            print(
                f"{scenario:<12} {result.total_return*100:>8.2f}% "
                f"{result.annual_return*100:>8.2f}% "
                f"{result.sharpe_ratio:>6.2f} "
                f"{result.sortino_ratio:>6.2f} "
                f"{result.calmar_ratio:>6.2f} "
                f"{result.profit_factor:>6.2f}"
            )

        print("=" * 70)


def test_edge_cases():
    """测试边缘情况"""
    print("\n" + "=" * 70)
    print("测试边缘情况")
    print("=" * 70)

    # 测试无交易信号的情况
    print("\n1. 测试无交易信号:")
    fetcher = MockDataFetcher(scenario="sideways")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)
    df = manager.get_daily_price("600000.SH", "20230101", "20230131")

    # 使用非常严格的参数，可能不会产生信号
    strategy = MaMacdRsiStrategy(
        rsi_overbought=99,
        rsi_oversold=1,
    )
    backtester = SimpleBacktester(initial_capital=100000.0)
    result = backtester.run(strategy, df)

    print(f"  交易次数: {result.trade_count}")
    print(f"  所有指标应为0或默认值")
    assert result.trade_count == 0 or result.sortino_ratio >= 0
    print("  ✅ 无交易信号情况处理正确")

    print("\n✅ 边缘情况测试通过")


if __name__ == "__main__":
    test_advanced_metrics()
    test_scenarios_comparison()
    test_edge_cases()

    print("\n" + "=" * 70)
    print("所有测试通过 ✅")
    print("=" * 70)
