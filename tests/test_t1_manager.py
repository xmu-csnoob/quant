"""
T+1持仓管理测试
"""

import sys
from pathlib import Path
from datetime import date

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.t1_manager import T1Manager, PositionLot


def test_position_lot():
    """测试持仓批次"""
    print("=" * 60)
    print("测试持仓批次")
    print("=" * 60)

    lot = PositionLot(
        code="600519.SH",
        shares=1000,
        cost_price=100.0,
        buy_date=date(2024, 1, 8),
        available_date=date(2024, 1, 9)
    )

    assert lot.code == "600519.SH"
    assert lot.shares == 1000
    print("  ✅ 持仓批次创建正确")


def test_t1_manager_buy():
    """测试T+1管理器买入"""
    print("\n" + "=" * 60)
    print("测试T+1管理器买入功能")
    print("=" * 60)

    manager = T1Manager()

    # 买入
    manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)
    manager.record_buy("600519.SH", 500, date(2024, 1, 9), 102.0)

    total = manager.get_total_shares("600519.SH")
    print(f"  买入后总持仓: {total}")
    assert total == 1500, f"总持仓应为1500，实际为{total}"
    print("  ✅ 买入功能正确")


def test_t1_availability():
    """测试T+1可用性"""
    print("\n" + "=" * 60)
    print("测试T+1可用性")
    print("=" * 60)

    manager = T1Manager()

    # 周一买入
    manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

    # 周一不可卖
    available = manager.get_available_shares("600519.SH", date(2024, 1, 8))
    print(f"  周一可卖: {available}")
    assert available == 0, "周一买入，周一应不可卖"

    # 周二可卖
    available = manager.get_available_shares("600519.SH", date(2024, 1, 9))
    print(f"  周二可卖: {available}")
    assert available == 1000, "周一买入，周二应可卖"
    print("  ✅ T+1可用性正确")


def test_can_sell():
    """测试卖出检查"""
    print("\n" + "=" * 60)
    print("测试卖出检查")
    print("=" * 60)

    manager = T1Manager()

    manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

    # 周一不可卖
    can_sell = manager.can_sell("600519.SH", 500, date(2024, 1, 8))
    assert not can_sell, "周一不应能卖出"
    print("  ✅ 周一不可卖")

    # 周二可卖
    can_sell = manager.can_sell("600519.SH", 500, date(2024, 1, 9))
    assert can_sell, "周二应能卖出"
    print("  ✅ 周二可卖")


def test_deduct_available():
    """测试扣减可用持仓"""
    print("\n" + "=" * 60)
    print("测试扣减可用持仓")
    print("=" * 60)

    manager = T1Manager()

    manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

    # 周二卖出
    success = manager.deduct_available("600519.SH", 800, date(2024, 1, 9))
    assert success, "卖出应成功"
    print("  ✅ 卖出成功")

    # 检查剩余
    total = manager.get_total_shares("600519.SH")
    available = manager.get_available_shares("600519.SH", date(2024, 1, 9))
    print(f"  剩余总持仓: {total}, 可卖: {available}")
    assert total == 200, f"总持仓应为200，实际为{total}"
    assert available == 200, f"可卖应为200，实际为{available}"
    print("  ✅ 扣减正确")


def test_position_info():
    """测试持仓信息获取"""
    print("\n" + "=" * 60)
    print("测试持仓信息获取")
    print("=" * 60)

    manager = T1Manager()

    manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

    info = manager.get_position_info("600519.SH", date(2024, 1, 9))
    print(f"  持仓信息: {info}")

    assert info["code"] == "600519.SH"
    assert info["total_shares"] == 1000
    assert info["available_shares"] == 1000
    print("  ✅ 持仓信息获取正确")


def test_multiple_stocks():
    """测试多只股票"""
    print("\n" + "=" * 60)
    print("测试多只股票管理")
    print("=" * 60)

    manager = T1Manager()

    # 买入多只股票
    manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)
    manager.record_buy("000001.SZ", 2000, date(2024, 1, 8), 10.0)

    # 检查各股票持仓
    positions = manager.get_all_positions(date(2024, 1, 9))
    print(f"  持仓股票数: {len(positions)}")

    for code, info in positions.items():
        print(f"    {code}: {info['total_shares']}股")

    assert len(positions) == 2, "应持有2只股票"
    print("  ✅ 多股票管理正确")


def test_locked_shares():
    """测试锁定持仓"""
    print("\n" + "=" * 60)
    print("测试锁定持仓")
    print("=" * 60)

    manager = T1Manager()

    manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

    # 周一锁定数量应为1000
    locked = manager.get_locked_shares("600519.SH", date(2024, 1, 8))
    print(f"  周一锁定: {locked}")
    assert locked == 1000, "周一应锁定全部"

    # 周二锁定数量应为0
    locked = manager.get_locked_shares("600519.SH", date(2024, 1, 9))
    print(f"  周二锁定: {locked}")
    assert locked == 0, "周二应无锁定"
    print("  ✅ 锁定持仓计算正确")


def main():
    """运行所有测试"""
    print("开始测试T+1持仓管理模块")
    print("=" * 60)

    test_position_lot()
    test_t1_manager_buy()
    test_t1_availability()
    test_can_sell()
    test_deduct_available()
    test_position_info()
    test_multiple_stocks()
    test_locked_shares()

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
