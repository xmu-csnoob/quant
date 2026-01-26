"""
技术指标测试脚本

测试 MA、MACD、RSI 三个指标的计算和信号生成
"""

import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.indicators import MA, EMA, MACD, RSI
from utils.indicators.ma import calculate_ma_cross_signal
from utils.indicators.macd import calculate_macd_signal, detect_divergence
from utils.indicators.rsi import calculate_rsi_signal, detect_rsi_divergence
from data.fetchers.mock import MockDataFetcher
from data.api.data_manager import DataManager
from data.storage.storage import DataStorage


def create_sample_data():
    """创建示例数据用于测试"""
    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)

    # 获取一只股票的数据
    df = manager.get_daily_price("600000.SH", "20230101", "20231231")
    return df


def test_ma():
    """测试 MA 指标"""
    print("=" * 60)
    print("测试 MA 指标")
    print("=" * 60)

    df = create_sample_data()

    # 计算 MA5, MA20, MA60
    ma_indicator = MA()
    df_ma5 = ma_indicator.calculate(df, period=5)
    df_ma20 = ma_indicator.calculate(df, period=20)
    df_ma60 = ma_indicator.calculate(df, period=60)

    # 合并结果
    result = df.copy()
    result["MA5"] = df_ma5["MA5"]
    result["MA20"] = df_ma20["MA20"]
    result["MA60"] = df_ma60["MA60"]

    # 计算金叉死叉
    result = calculate_ma_cross_signal(result, fast_period=5, slow_period=20)

    # 显示最近10天
    print("\n最近10天的数据:")
    print(result[["trade_date", "close", "MA5", "MA20", "MA60", "ma_signal"]].tail(10))

    # 统计信号
    print("\n信号统计:")
    print(f"金叉次数: {(result['ma_signal'] == 1).sum()}")
    print(f"死叉次数: {(result['ma_signal'] == -1).sum()}")

    return result


def test_ema():
    """测试 EMA 指标"""
    print("\n" + "=" * 60)
    print("测试 EMA 指标")
    print("=" * 60)

    df = create_sample_data()

    # 计算 EMA12, EMA26
    ema_indicator = EMA()
    df_ema12 = ema_indicator.calculate(df, period=12)
    df_ema26 = ema_indicator.calculate(df, period=26)

    # 合并结果
    result = df.copy()
    result["EMA12"] = df_ema12["EMA12"]
    result["EMA26"] = df_ema26["EMA26"]

    # 显示最近10天
    print("\n最近10天的数据:")
    print(result[["trade_date", "close", "EMA12", "EMA26"]].tail(10))

    return result


def test_macd():
    """测试 MACD 指标"""
    print("\n" + "=" * 60)
    print("测试 MACD 指标")
    print("=" * 60)

    df = create_sample_data()

    # 计算 MACD
    macd_indicator = MACD()
    result = macd_indicator.calculate(df)

    # 计算信号
    result = calculate_macd_signal(result)

    # 检测背离
    result = detect_divergence(result)

    # 显示最近10天
    print("\n最近10天的数据:")
    cols = ["trade_date", "close", "DIF", "DEA", "MACD", "macd_signal", "macd_signal_strength"]
    print(result[cols].tail(10))

    # 统计信号
    print("\n信号统计:")
    print(f"金叉次数: {(result['macd_signal'] == 1).sum()}")
    print(f"死叉次数: {(result['macd_signal'] == -1).sum()}")
    print(f"零轴上金叉: {(result['macd_signal_strength'] == 'zero_axis_above').sum()}")
    print(f"零轴下金叉: {(result['macd_signal_strength'] == 'zero_axis_below').sum()}")
    print(f"顶背离次数: {result['bearish_divergence'].sum()}")
    print(f"底背离次数: {result['bullish_divergence'].sum()}")

    return result


def test_rsi():
    """测试 RSI 指标"""
    print("\n" + "=" * 60)
    print("测试 RSI 指标")
    print("=" * 60)

    df = create_sample_data()

    # 计算 RSI
    rsi_indicator = RSI()
    result = rsi_indicator.calculate(df)

    # 计算信号
    result = calculate_rsi_signal(result)

    # 检测背离
    result = detect_rsi_divergence(result)

    # 显示最近10天
    print("\n最近10天的数据:")
    cols = ["trade_date", "close", "RSI", "rsi_overbought", "rsi_oversold", "rsi_signal"]
    print(result[cols].tail(10))

    # 统计信号
    print("\n信号统计:")
    print(f"超买次数: {result['rsi_overbought'].sum()}")
    print(f"超卖次数: {result['rsi_oversold'].sum()}")
    print(f"顶背离次数: {result['rsi_bearish_divergence'].sum()}")
    print(f"底背离次数: {result['rsi_bullish_divergence'].sum()}")

    return result


def test_combined():
    """测试组合指标"""
    print("\n" + "=" * 60)
    print("测试组合指标")
    print("=" * 60)

    df = create_sample_data()

    # 计算所有指标
    result = df.copy()

    # MA
    result = calculate_ma_cross_signal(result, fast_period=5, slow_period=20)

    # MACD
    result = calculate_macd_signal(result)
    result = detect_divergence(result)

    # RSI
    result = calculate_rsi_signal(result)
    result = detect_rsi_divergence(result)

    # 显示最近10天
    print("\n最近10天的组合信号:")
    cols = [
        "trade_date",
        "close",
        "ma_signal",
        "macd_signal",
        "rsi_signal",
        "bearish_divergence",
        "bullish_divergence",
    ]
    print(result[cols].tail(10))

    # 分析信号一致性
    print("\n信号分析:")

    # 多头信号（金叉）
    ma_bullish = result["ma_signal"] == 1
    macd_bullish = result["macd_signal"] == 1
    rsi_bullish = result["rsi_signal"] == 1

    print(f"\n多头信号:")
    print(f"  MA金叉: {ma_bullish.sum()}次")
    print(f"  MACD金叉: {macd_bullish.sum()}次")
    print(f"  RSI超卖: {rsi_bullish.sum()}次")

    # 共同信号
    both_ma_macd = ma_bullish & macd_bullish
    print(f"\n  MA+MACD同时金叉: {both_ma_macd.sum()}次")

    # 背离信号
    print(f"\n背离信号:")
    print(f"  MACD顶背离: {result['bearish_divergence'].sum()}次")
    print(f"  MACD底背离: {result['bullish_divergence'].sum()}次")
    print(f"  RSI顶背离: {result['rsi_bearish_divergence'].sum()}次")
    print(f"  RSI底背离: {result['rsi_bullish_divergence'].sum()}次")

    return result


if __name__ == "__main__":
    print("开始测试技术指标...")

    try:
        test_ma()
        test_ema()
        test_macd()
        test_rsi()
        test_combined()

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback

        traceback.print_exc()
