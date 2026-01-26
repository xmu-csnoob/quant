"""
策略回测测试脚本

测试 MA + MACD + RSI 组合策略的回测效果
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from strategies import MaMacdRsiStrategy
from backtesting.simple_backtester import SimpleBacktester
from data.fetchers.mock import MockDataFetcher
from data.api.data_manager import DataManager
from data.storage.storage import DataStorage


def run_backtest(scenario: str = "bull"):
    """
    运行回测

    Args:
        scenario: 市场场景 (bull/bear/sideways/volatile)
    """
    print("=" * 70)
    print(f"回测场景: {scenario.upper()}")
    print("=" * 70)

    # 获取数据
    fetcher = MockDataFetcher(scenario=scenario)
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)
    df = manager.get_daily_price("600000.SH", "20230101", "20231231")

    print(f"数据范围: {df.iloc[0]['trade_date']} ~ {df.iloc[-1]['trade_date']}")
    print(f"数据量: {len(df)} 条")
    print(f"起始价格: {df.iloc[0]['close']:.2f}")
    print(f"结束价格: {df.iloc[-1]['close']:.2f}")
    print(f"买入持有收益率: {(df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100:.2f}%")
    print()

    # 创建策略
    strategy = MaMacdRsiStrategy(
        ma_fast=5,
        ma_slow=20,
        ma_long=60,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rsi_period=14,
        rsi_overbought=75,
        rsi_oversold=25,
    )

    # 运行回测
    backtester = SimpleBacktester(initial_capital=100000.0)
    result = backtester.run(strategy, df)

    # 打印结果
    result.print_summary()
    result.print_trades(limit=20)

    return result


def compare_scenarios():
    """对比不同市场场景下的表现"""
    scenarios = ["bull", "bear", "sideways", "volatile"]

    print("\n" + "=" * 70)
    print("不同市场场景对比")
    print("=" * 70)

    results = {}

    for scenario in scenarios:
        try:
            result = run_backtest(scenario)
            results[scenario] = result
            print("\n")
        except Exception as e:
            print(f"\n场景 {scenario} 出错: {e}")
            import traceback

            traceback.print_exc()

    # 对比表
    if results:
        print("\n" + "=" * 70)
        print("场景对比汇总")
        print("=" * 70)
        print(
            f"{'场景':<12} {'收益率':<10} {'交易次数':<8} {'胜率':<8} {'最大回撤':<10} {'夏普比率':<10}"
        )
        print("-" * 70)

        for scenario, result in results.items():
            print(
                f"{scenario:<12} {result.total_return*100:>8.2f}% "
                f"{result.trade_count:>6d} "
                f"{result.win_rate*100:>6.2f}% "
                f"{result.max_drawdown*100:>8.2f}% "
                f"{result.sharpe_ratio:>8.2f}"
            )

        print("=" * 70)

        # 分析
        print("\n策略分析:")
        print("1. 牛市场景: 策略应该能捕捉上涨趋势")
        print("2. 熊市场景: 策略应该能避开下跌")
        print("3. 震荡场景: 策略可能会有频繁交易")
        print("4. 波动场景: 测试策略的稳定性")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="策略回测")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["bull", "bear", "sideways", "volatile", "all"],
        help="市场场景",
    )

    args = parser.parse_args()

    if args.scenario == "all":
        compare_scenarios()
    else:
        run_backtest(args.scenario)
