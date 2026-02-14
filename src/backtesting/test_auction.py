"""
集合竞价模拟测试

测试集合竞价撮合逻辑
"""

import sys
from pathlib import Path
from datetime import time
from decimal import Decimal
import random

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting.auction import (
    CallAuction,
    CallAuctionSimulator,
    AuctionOrder,
    AuctionPhase,
    OrderSide,
)


def test_auction_phase():
    """测试竞价阶段判断"""
    print("=" * 60)
    print("测试竞价阶段判断")
    print("=" * 60)

    auction = CallAuction()

    test_times = [
        (time(9, 10), AuctionPhase.PRE_OPEN, "盘前"),
        (time(9, 15), AuctionPhase.FREE_CANCEL, "可撤单"),
        (time(9, 17), AuctionPhase.FREE_CANCEL, "可撤单"),
        (time(9, 20), AuctionPhase.NO_CANCEL, "不可撤单"),
        (time(9, 23), AuctionPhase.NO_CANCEL, "不可撤单"),
        (time(9, 25), AuctionPhase.MATCHING, "撮合"),
        (time(9, 30), AuctionPhase.CONTINUOUS, "连续竞价"),
        (time(10, 0), AuctionPhase.CONTINUOUS, "连续竞价"),
    ]

    for t, expected, desc in test_times:
        phase = auction.get_phase(t)
        status = "✅" if phase == expected else "❌"
        print(f"  {status} {t.strftime('%H:%M')} -> {phase.value:12s} ({desc})")

    print()


def test_basic_auction():
    """测试基本集合竞价"""
    print("=" * 60)
    print("测试基本集合竞价")
    print("=" * 60)

    auction = CallAuction()

    # 添加订单
    # 买单
    auction.add_order(AuctionOrder("B1", "600519.SH", OrderSide.BUY, Decimal("1800"), 100))
    auction.add_order(AuctionOrder("B2", "600519.SH", OrderSide.BUY, Decimal("1799"), 200))
    auction.add_order(AuctionOrder("B3", "600519.SH", OrderSide.BUY, Decimal("1798"), 150))

    # 卖单
    auction.add_order(AuctionOrder("S1", "600519.SH", OrderSide.SELL, Decimal("1800"), 150))
    auction.add_order(AuctionOrder("S2", "600519.SH", OrderSide.SELL, Decimal("1801"), 100))
    auction.add_order(AuctionOrder("S3", "600519.SH", OrderSide.SELL, Decimal("1799"), 200))

    # 执行竞价
    result = auction.execute(
        symbol="600519.SH",
        prev_close=Decimal("1795"),
        limit_up=Decimal("1974.5"),
        limit_down=Decimal("1615.5"),
    )

    print(f"\n竞价结果:")
    print(f"  开盘价: {result.open_price}")
    print(f"  成交量: {result.volume}")
    print(f"  成交额: {result.turnover}")
    print(f"  成功: {'✅' if result.success else '❌'}")

    print(f"\n  撮合明细:")
    for m in result.matches[:5]:  # 只显示前5条
        print(f"    {m.buy_order_id} <-> {m.sell_order_id}: {m.quantity}@{m.price}")

    print()


def test_cancel_order():
    """测试撤单功能"""
    print("=" * 60)
    print("测试撤单功能")
    print("=" * 60)

    auction = CallAuction()

    # 添加订单
    auction.add_order(AuctionOrder("B1", "600519.SH", OrderSide.BUY, Decimal("1800"), 100))

    # 9:15 可撤单
    can_cancel = auction.cancel_order("B1", "600519.SH", time(9, 15))
    print(f"  9:15 撤单: {'✅ 成功' if can_cancel else '❌ 失败'}")

    # 重新添加
    auction.add_order(AuctionOrder("B2", "600519.SH", OrderSide.BUY, Decimal("1800"), 100))

    # 9:22 不可撤单
    can_cancel = auction.cancel_order("B2", "600519.SH", time(9, 22))
    print(f"  9:22 撤单: {'⚠️ 成功（不应该）' if can_cancel else '✅ 被拒绝'}")

    print()


def test_simulated_auction():
    """测试模拟集合竞价"""
    print("=" * 60)
    print("测试模拟集合竞价（生成随机订单）")
    print("=" * 60)

    random.seed(42)  # 固定种子以便复现

    simulator = CallAuctionSimulator()

    # 模拟不同股票的开盘竞价
    stocks = [
        ("600519.SH", Decimal("1800"), "主板大盘股"),
        ("300750.SZ", Decimal("200"), "创业板"),
        ("688981.SH", Decimal("100"), "科创板"),
    ]

    for symbol, prev_close, desc in stocks:
        # 计算涨跌停
        limit_up = prev_close * Decimal("1.1")
        limit_down = prev_close * Decimal("0.9")

        result = simulator.simulate_open(
            symbol=symbol,
            prev_close=prev_close,
            limit_up=limit_up,
            limit_down=limit_down,
            avg_volume=500000,
            volatility=Decimal("0.03"),
        )

        change_pct = ((result.open_price or prev_close) / prev_close - 1) * 100

        print(f"\n  {symbol} ({desc}):")
        print(f"    前收盘: {prev_close}")
        print(f"    开盘价: {result.open_price}")
        print(f"    涨跌幅: {change_pct:+.2f}%")
        print(f"    成交量: {result.volume:,}")
        print(f"    成交额: {result.turnover:,.0f}")

    print()


def test_price_discovery():
    """测试价格发现机制"""
    print("=" * 60)
    print("测试价格发现机制（成交量最大化原则）")
    print("=" * 60)

    auction = CallAuction()

    # 构造一个场景：在1800元价位成交量最大
    # 买盘集中在1795-1810
    for i, price in enumerate([1810, 1805, 1800, 1795, 1790]):
        auction.add_order(AuctionOrder(
            f"B{i}", "600519.SH", OrderSide.BUY,
            Decimal(str(price)), 100
        ))

    # 卖盘集中在1795-1810
    for i, price in enumerate([1795, 1800, 1800, 1805, 1810]):
        auction.add_order(AuctionOrder(
            f"S{i}", "600519.SH", OrderSide.SELL,
            Decimal(str(price)), 100
        ))

    result = auction.execute(
        symbol="600519.SH",
        prev_close=Decimal("1800"),
    )

    print(f"\n订单分布:")
    print(f"  买盘: 1810, 1805, 1800, 1795, 1790 各100股")
    print(f"  卖盘: 1795, 1800, 1800, 1805, 1810 各100股")

    print(f"\n理论分析:")
    print(f"  1795: 买400股 vs 卖100股 -> 成交100股")
    print(f"  1800: 买300股 vs 卖300股 -> 成交300股 ← 最大")
    print(f"  1805: 买200股 vs 卖400股 -> 成交200股")
    print(f"  1810: 买100股 vs 卖500股 -> 成交100股")

    print(f"\n实际结果:")
    print(f"  开盘价: {result.open_price}")
    print(f"  成交量: {result.volume}")

    is_correct = result.open_price == Decimal("1800") and result.volume == 300
    print(f"  验证: {'✅ 正确' if is_correct else '❌ 错误'}")

    print()


def test_no_match():
    """测试无法成交的情况"""
    print("=" * 60)
    print("测试无法成交的情况")
    print("=" * 60)

    auction = CallAuction()

    # 买盘价格低于卖盘价格，无法成交
    auction.add_order(AuctionOrder("B1", "600519.SH", OrderSide.BUY, Decimal("1790"), 100))
    auction.add_order(AuctionOrder("S1", "600519.SH", OrderSide.SELL, Decimal("1810"), 100))

    result = auction.execute(
        symbol="600519.SH",
        prev_close=Decimal("1800"),
    )

    print(f"\n买盘最高价: 1790")
    print(f"卖盘最低价: 1810")
    print(f"结果: 无法匹配")
    print(f"  开盘价: {result.open_price}")
    print(f"  成交量: {result.volume}")
    print(f"  成功: {'❌' if result.success else '✅ 未成交（符合预期）'}")

    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始集合竞价模拟测试")
    print("=" * 60)
    print()

    test_auction_phase()
    test_basic_auction()
    test_cancel_order()
    test_price_discovery()
    test_no_match()
    test_simulated_auction()

    print("=" * 60)
    print("所有测试完成 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
