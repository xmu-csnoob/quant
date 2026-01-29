"""
策略对比实验

对比原始策略和增强策略的表现
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from backtesting.simple_backtester import SimpleBacktester
from backtesting.data_split import split_data
from strategies.ma_macd_rsi import MaMacdRsiStrategy
from experiments.enhanced_strategy import EnhancedTechnicalStrategy
from data.fetchers.mock import MockDataFetcher
from data.storage.storage import DataStorage
from data.api.data_manager import DataManager


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def compare_strategies():
    """对比两个策略"""
    print_section("策略对比实验：原始策略 vs 增强策略")

    # 获取数据
    print("\n步骤1：获取数据")
    print("-" * 70)

    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)

    df = manager.get_daily_price("600000.SH", "20230101", "20231231")
    print(f"获取数据: {len(df)} 条")
    print(f"日期范围: {df.iloc[0]['trade_date']} ~ {df.iloc[-1]['trade_date']}")

    # 划分数据集
    print("\n步骤2：划分数据集")
    print("-" * 70)

    data_split = split_data(
        df,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )

    # 创建策略
    print("\n步骤3：创建策略")
    print("-" * 70)

    original_strategy = MaMacdRsiStrategy()
    enhanced_strategy = EnhancedTechnicalStrategy(
        min_signal_strength=0.4,  # 适中阈值
        volume_confirmation=True,
        volatility_filter=True,
        momentum_confirmation=True,
        position_scaling=True,
    )

    print("原始策略: MA + MACD + RSI")
    print("增强策略: 44个特征 + 成交量确认 + 波动率过滤 + 动量确认")

    # 回测 - 训练集
    print("\n步骤4：回测 - 训练集")
    print("-" * 70)

    backtester = SimpleBacktester(initial_capital=100000.0)

    print("\n原始策略（训练集）：")
    result_original_train = backtester.run(MaMacdRsiStrategy(), data_split.train)
    result_original_train.print_summary()

    print("\n增强策略（训练集）：")
    result_enhanced_train = backtester.run(
        EnhancedTechnicalStrategy(
            min_signal_strength=0.4,
            volume_confirmation=True,
            volatility_filter=True,
            momentum_confirmation=True,
            position_scaling=True,
        ),
        data_split.train,
    )
    result_enhanced_train.print_summary()

    # 回测 - 验证集
    print("\n步骤5：回测 - 验证集")
    print("-" * 70)

    print("\n原始策略（验证集）：")
    result_original_val = backtester.run(MaMacdRsiStrategy(), data_split.val)
    result_original_val.print_summary()

    print("\n增强策略（验证集）：")
    result_enhanced_val = backtester.run(
        EnhancedTechnicalStrategy(
            min_signal_strength=0.4,
            volume_confirmation=True,
            volatility_filter=True,
            momentum_confirmation=True,
            position_scaling=True,
        ),
        data_split.val,
    )
    result_enhanced_val.print_summary()

    # 回测 - 测试集
    print("\n步骤6：回测 - 测试集（最终验证）")
    print("-" * 70)

    print("\n原始策略（测试集）：")
    result_original_test = backtester.run(MaMacdRsiStrategy(), data_split.test)
    result_original_test.print_summary()
    result_original_test.print_trades(limit=10)

    print("\n增强策略（测试集）：")
    result_enhanced_test = backtester.run(
        EnhancedTechnicalStrategy(
            min_signal_strength=0.4,
            volume_confirmation=True,
            volatility_filter=True,
            momentum_confirmation=True,
            position_scaling=True,
        ),
        data_split.test,
    )
    result_enhanced_test.print_summary()
    result_enhanced_test.print_trades(limit=10)

    # 对比总结
    print_section("对比总结")

    print("\n训练集对比：")
    print(f"  原始策略: {result_original_train.total_return*100:.2f}% ({result_original_train.trade_count}笔交易)")
    print(f"  增强策略: {result_enhanced_train.total_return*100:.2f}% ({result_enhanced_train.trade_count}笔交易)")
    print(f"  差异: {(result_enhanced_train.total_return - result_original_train.total_return)*100:+.2f}%")

    print("\n验证集对比：")
    print(f"  原始策略: {result_original_val.total_return*100:.2f}% ({result_original_val.trade_count}笔交易)")
    print(f"  增强策略: {result_enhanced_val.total_return*100:.2f}% ({result_enhanced_val.trade_count}笔交易)")
    print(f"  差异: {(result_enhanced_val.total_return - result_original_val.total_return)*100:+.2f}%")

    print("\n测试集对比：")
    print(f"  原始策略: {result_original_test.total_return*100:.2f}% ({result_original_test.trade_count}笔交易)")
    print(f"  增强策略: {result_enhanced_test.total_return*100:.2f}% ({result_enhanced_test.trade_count}笔交易)")
    print(f"  差异: {(result_enhanced_test.total_return - result_original_test.total_return)*100:+.2f}%")

    # 买入持有基准
    print("\n买入持有基准：")
    buy_hold_train = (data_split.train.iloc[-1]["close"] / data_split.train.iloc[0]["close"] - 1) * 100
    buy_hold_val = (data_split.val.iloc[-1]["close"] / data_split.val.iloc[0]["close"] - 1) * 100
    buy_hold_test = (data_split.test.iloc[-1]["close"] / data_split.test.iloc[0]["close"] - 1) * 100

    print(f"  训练集: {buy_hold_train:.2f}%")
    print(f"  验证集: {buy_hold_val:.2f}%")
    print(f"  测试集: {buy_hold_test:.2f}%")

    # 关键发现
    print_section("关键发现")

    print("\n交易次数对比：")
    print(f"  原始策略训练集: {result_original_train.trade_count}笔")
    print(f"  增强策略训练集: {result_enhanced_train.trade_count}笔")
    print(f"  增加: {result_enhanced_train.trade_count - result_original_train.trade_count}笔")

    print("\n胜率对比：")
    print(f"  原始策略: {result_original_train.win_rate*100:.1f}%")
    print(f"  增强策略: {result_enhanced_train.win_rate*100:.1f}%")

    # 分析
    print("\n分析：")

    if result_enhanced_train.trade_count > result_original_train.trade_count:
        print("✓ 增强策略交易次数更多（条件更灵活）")

    if result_enhanced_test.total_return > result_original_test.total_return:
        print("✓ 增强策略在测试集表现更好")
    else:
        print("✗ 增强策略在测试集表现更差（可能过拟合）")

    if result_enhanced_test.total_return > 0:
        print("✓ 增强策略在测试集盈利")
    else:
        print("✗ 增强策略在测试集亏损")

    print_section("结论")

    print("\n改进效果评估：")
    print("1. 交易次数：原策略 → 增强策略")
    print(f"   {result_original_test.trade_count}笔 → {result_enhanced_test.trade_count}笔")

    print("\n2. 收益率：原策略 → 增强策略")
    print(f"   {result_original_test.total_return*100:.2f}% → {result_enhanced_test.total_return*100:.2f}%")

    print("\n3. 与买入持有对比：")
    print(f"   买入持有: {buy_hold_test:.2f}%")
    print(f"   原始策略: {result_original_test.total_return*100:.2f}%")
    print(f"   增强策略: {result_enhanced_test.total_return*100:.2f}%")

    print("\n建议：")
    if result_enhanced_test.total_return > buy_hold_test / 100:
        print("✓ 增强策略跑赢了买入持有！")
    else:
        print("✗ 仍需改进，未能跑赢买入持有")


if __name__ == "__main__":
    compare_strategies()
