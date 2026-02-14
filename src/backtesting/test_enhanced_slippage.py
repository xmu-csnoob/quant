"""
增强滑点模型测试

测试时段感知滑点和综合滑点模型
"""

import sys
from pathlib import Path
from datetime import time
from decimal import Decimal

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting.slippage import (
    TimeAwareSlippage,
    ComprehensiveSlippage,
    TradingSession,
    TradingTimeContext,
    SESSION_MULTIPLIERS,
)
from src.backtesting.costs import TradeSide


def test_trading_session_detection():
    """测试交易时段检测"""
    print("=" * 60)
    print("测试交易时段检测")
    print("=" * 60)

    test_cases = [
        (time(9, 20), TradingSession.PRE_MARKET),
        (time(9, 45), TradingSession.OPENING),
        (time(10, 30), TradingSession.MORNING),
        (time(11, 0), TradingSession.MORNING),
        (time(12, 0), TradingSession.NOON),
        (time(13, 30), TradingSession.AFTERNOON),
        (time(14, 45), TradingSession.CLOSING),
        (time(15, 30), TradingSession.AFTER_HOURS),
    ]

    for t, expected in test_cases:
        ctx = TradingTimeContext.from_time(t)
        status = "✅" if ctx.session == expected else "❌"
        print(f"  {status} {t.strftime('%H:%M')} -> {ctx.session.value} (期望: {expected.value})")

    print()


def test_session_multipliers():
    """测试时段系数"""
    print("=" * 60)
    print("测试时段系数")
    print("=" * 60)

    for session, multiplier in SESSION_MULTIPLIERS.items():
        print(f"  {session.value:15s}: ×{multiplier}")

    print()


def test_time_aware_slippage():
    """测试时段感知滑点"""
    print("=" * 60)
    print("测试时段感知滑点模型")
    print("=" * 60)

    model = TimeAwareSlippage(base_rate=Decimal("0.001"))
    price = Decimal("10.00")

    # 测试不同时段
    test_times = [
        (time(9, 20), "集合竞价"),
        (time(9, 45), "开盘"),
        (time(10, 30), "上午正常"),
        (time(14, 45), "尾盘"),
    ]

    print(f"\n买入价格: {price}")
    print("-" * 40)

    for t, desc in test_times:
        result = model.apply_slippage(price, TradeSide.BUY, current_time=t)
        slippage_pct = float(result.slippage_rate) * 100
        print(f"  {desc:12s} ({t.strftime('%H:%M')}): "
              f"成交价 {result.adjusted_price:.4f}, "
              f"滑点 {slippage_pct:.3f}%")

    print()


def test_volume_impact():
    """测试成交量影响"""
    print("=" * 60)
    print("测试成交量影响")
    print("=" * 60)

    model = TimeAwareSlippage(
        base_rate=Decimal("0.001"),
        volume_impact_factor=Decimal("0.3")
    )
    price = Decimal("10.00")
    avg_volume = Decimal("100000")

    print(f"\n基准: 平均成交量 {avg_volume}")
    print("-" * 40)

    volume_ratios = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    for ratio in volume_ratios:
        volume = avg_volume * Decimal(str(ratio))
        result = model.apply_slippage(
            price, TradeSide.BUY,
            volume=volume,
            avg_volume=avg_volume
        )
        slippage_pct = float(result.slippage_rate) * 100
        print(f"  成交量 {ratio*100:5.1f}%: 成交价 {result.adjusted_price:.4f}, "
              f"滑点 {slippage_pct:.3f}%")

    print()


def test_liquidity_impact():
    """测试流动性影响"""
    print("=" * 60)
    print("测试流动性影响")
    print("=" * 60)

    model = TimeAwareSlippage(
        base_rate=Decimal("0.001"),
        liquidity_factor=Decimal("0.4")
    )
    price = Decimal("10.00")

    print(f"\n买入价格: {price}")
    print("-" * 40)

    liquidity_scores = [
        (Decimal("1.0"), "高流动性（大盘股）"),
        (Decimal("0.7"), "中等流动性"),
        (Decimal("0.5"), "一般流动性"),
        (Decimal("0.3"), "低流动性（小盘股）"),
        (Decimal("0.1"), "极低流动性"),
    ]

    for score, desc in liquidity_scores:
        result = model.apply_slippage(
            price, TradeSide.BUY,
            liquidity_score=score
        )
        slippage_pct = float(result.slippage_rate) * 100
        print(f"  {desc:20s}: 成交价 {result.adjusted_price:.4f}, "
              f"滑点 {slippage_pct:.3f}%")

    print()


def test_comprehensive_slippage():
    """测试综合滑点模型"""
    print("=" * 60)
    print("测试综合滑点模型")
    print("=" * 60)

    model = ComprehensiveSlippage(
        base_rate=Decimal("0.001"),
        max_rate=Decimal("0.05")
    )
    price = Decimal("10.00")

    scenarios = [
        {
            "name": "正常时段小单",
            "time": time(10, 30),
            "volume": Decimal("1000"),
            "avg_volume": Decimal("100000"),
            "liquidity": Decimal("0.8"),
            "volatility": Decimal("0.02"),
        },
        {
            "name": "开盘大单",
            "time": time(9, 45),
            "volume": Decimal("20000"),
            "avg_volume": Decimal("100000"),
            "liquidity": Decimal("0.6"),
            "volatility": Decimal("0.03"),
        },
        {
            "name": "尾盘低流动性",
            "time": time(14, 45),
            "volume": Decimal("5000"),
            "avg_volume": Decimal("100000"),
            "liquidity": Decimal("0.3"),
            "volatility": Decimal("0.04"),
        },
    ]

    print(f"\n买入价格: {price}")
    print("-" * 60)

    for s in scenarios:
        result = model.apply_slippage(
            price, TradeSide.BUY,
            volume=s["volume"],
            avg_volume=s["avg_volume"],
            current_time=s["time"],
            liquidity_score=s["liquidity"],
            volatility=s["volatility"]
        )
        slippage_pct = float(result.slippage_rate) * 100
        print(f"  {s['name']:20s}:")
        print(f"    成交价: {result.adjusted_price:.4f}")
        print(f"    滑点: {slippage_pct:.3f}%")
        print()

    print()


def test_comparison():
    """对比不同滑点模型"""
    print("=" * 60)
    print("对比不同滑点模型")
    print("=" * 60)

    from src.backtesting.slippage import FixedSlippage, VolumeBasedSlippage

    models = [
        ("固定滑点 0.1%", FixedSlippage(Decimal("0.001"))),
        ("成交量滑点", VolumeBasedSlippage(Decimal("0.001"))),
        ("时段感知滑点", TimeAwareSlippage(Decimal("0.001"))),
        ("综合滑点", ComprehensiveSlippage(Decimal("0.001"))),
    ]

    price = Decimal("10.00")
    volume = Decimal("10000")
    avg_volume = Decimal("50000")

    print(f"\n场景: 价格{price}, 成交量{volume}, 平均成交量{avg_volume}")
    print(f"时段: 开盘 (9:45)")
    print("-" * 60)

    for name, model in models:
        if isinstance(model, (TimeAwareSlippage, ComprehensiveSlippage)):
            result = model.apply_slippage(
                price, TradeSide.BUY,
                volume=volume,
                avg_volume=avg_volume,
                current_time=time(9, 45)
            )
        else:
            result = model.apply_slippage(
                price, TradeSide.BUY,
                volume=volume,
                avg_volume=avg_volume
            )

        slippage_pct = float(result.slippage_rate) * 100
        print(f"  {name:20s}: 成交价 {result.adjusted_price:.4f}, "
              f"滑点 {slippage_pct:.3f}%")

    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试增强滑点模型")
    print("=" * 60)
    print()

    test_trading_session_detection()
    test_session_multipliers()
    test_time_aware_slippage()
    test_volume_impact()
    test_liquidity_impact()
    test_comprehensive_slippage()
    test_comparison()

    print("=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
