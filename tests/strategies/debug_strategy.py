"""
调试策略信号生成

检查为什么没有产生交易信号
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from strategies import MaMacdRsiStrategy
from data.fetchers.mock import MockDataFetcher
from data.api.data_manager import DataManager
from data.storage.storage import DataStorage


def debug_signals():
    """调试信号生成"""
    # 获取数据
    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)
    df = manager.get_daily_price("600000.SH", "20230101", "20231231")

    # 创建策略
    strategy = MaMacdRsiStrategy()

    # 计算指标
    result = strategy.calculate_indicators(df)

    # 检查关键指标
    print("=" * 80)
    print("指标检查")
    print("=" * 80)

    # 检查多头排列
    ma_fast_col = f"MA{strategy.ma_fast}"
    ma_slow_col = f"MA{strategy.ma_slow}"
    ma_long_col = f"MA{strategy.ma_long}"

    result["bullish_alignment"] = (
        (result[ma_fast_col] > result[ma_slow_col])
        & (result[ma_slow_col] > result[ma_long_col])
        & (result[ma_fast_col] > result[ma_fast_col].shift(1))
    )

    print(f"\n多头排列次数: {result['bullish_alignment'].sum()}")
    print(f"MA5 > MA20 > MA60 的天数: {result['bullish_alignment'].sum()}")

    # 检查 MACD 金叉
    print(f"\nMACD 金叉次数: {(result['macd_signal'] == 1).sum()}")
    print(f"MACD 零轴上金叉次数: {(result['macd_signal_strength'] == 'zero_axis_above').sum()}")

    # 检查 RSI
    print(f"\nRSI 超买次数: {(result['rsi_overbought']).sum()}")
    print(f"RSI 超卖次数: {(result['rsi_oversold']).sum()}")
    print(f"RSI < {strategy.rsi_overbought} 次数: {(result['RSI'] < strategy.rsi_overbought).sum()}")

    # 检查背离
    print(f"\nMACD 顶背离次数: {result['bearish_divergence'].sum()}")
    print(f"MACD 底背离次数: {result['bullish_divergence'].sum()}")
    print(f"RSI 顶背离次数: {result['rsi_bearish_divergence'].sum()}")
    print(f"RSI 底背离次数: {result['rsi_bullish_divergence'].sum()}")

    # 检查同时满足条件的情况
    buy_conditions = (
        result["bullish_alignment"]
        & (result["macd_signal"] == 1)
        & (result["RSI"] < strategy.rsi_overbought)
        & (~result["bearish_divergence"])
        & (~result["rsi_bearish_divergence"])
    )

    print(f"\n同时满足买入条件的天数: {buy_conditions.sum()}")

    # 显示满足条件的天
    if buy_conditions.sum() > 0:
        print("\n满足买入条件的天:")
        cols = [
            "trade_date",
            "close",
            ma_fast_col,
            ma_slow_col,
            ma_long_col,
            "DIF",
            "DEA",
            "MACD",
            "RSI",
            "bullish_alignment",
            "macd_signal",
            "macd_signal_strength",
        ]
        print(result.loc[buy_conditions, cols].tail(10))
    else:
        print("\n没有一天满足所有买入条件！")

        # 让我们看看为什么
        print("\n分析各个条件:")
        for i in range(max(strategy.ma_long, 100), min(len(result), 110)):
            row = result.iloc[i]
            print(f"\n{row['trade_date']}:")
            print(f"  价格: {row['close']:.2f}")
            print(f"  MA5: {row[ma_fast_col]:.2f}, MA20: {row[ma_slow_col]:.2f}, MA60: {row[ma_long_col]:.2f}")
            print(f"  多头排列: {row['bullish_alignment']}")
            print(f"  MACD 金叉: {row['macd_signal'] == 1}, 强度: {row['macd_signal_strength']}")
            print(f"  RSI: {row['RSI']:.2f}, 超买线: {strategy.rsi_overbought}")
            print(f"  RSI < {strategy.rsi_overbought}: {row['RSI'] < strategy.rsi_overbought}")
            print(f"  MACD 顶背离: {row['bearish_divergence']}")
            print(f"  RSI 顶背离: {row['rsi_bearish_divergence']}")

            # 检查买入条件
            checks = []
            if not row["bullish_alignment"]:
                checks.append("❌ 多头排列不满足")
            else:
                checks.append("✓ 多头排列满足")

            if row["macd_signal"] != 1:
                checks.append("❌ MACD 金叉不满足")
            else:
                checks.append("✓ MACD 金叉满足")

            if row["RSI"] >= strategy.rsi_overbought:
                checks.append(f"❌ RSI 超买 ({row['RSI']:.2f} >= {strategy.rsi_overbought})")
            else:
                checks.append(f"✓ RSI 正常 ({row['RSI']:.2f} < {strategy.rsi_overbought})")

            if row["bearish_divergence"]:
                checks.append("❌ MACD 顶背离")
            else:
                checks.append("✓ 无 MACD 顶背离")

            if row["rsi_bearish_divergence"]:
                checks.append("❌ RSI 顶背离")
            else:
                checks.append("✓ 无 RSI 顶背离")

            for check in checks:
                print(f"    {check}")


if __name__ == "__main__":
    debug_signals()
