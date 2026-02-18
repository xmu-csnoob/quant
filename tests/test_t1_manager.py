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


# ============ pytest 风格的单元测试 ============

import pytest
from src.trading.t1_manager import (
    T1Manager, BuyRecord, PositionLot,
    is_trading_day, get_next_trading_day, get_t1_manager
)


class TestBuyRecord:
    """测试买入记录"""

    def test_buy_record_creation(self):
        """测试买入记录创建"""
        record = BuyRecord(
            code="600519.SH",
            shares=1000,
            buy_date=date(2024, 1, 8),
            remaining_shares=1000,
            price=100.0
        )

        assert record.code == "600519.SH"
        assert record.shares == 1000
        assert record.price == 100.0
        # 可卖日期应该是下一交易日
        assert record.available_date == date(2024, 1, 9)

    def test_buy_record_weekend_skip(self):
        """测试周末跳过"""
        # 周五买入
        record = BuyRecord(
            code="600519.SH",
            shares=1000,
            buy_date=date(2024, 1, 5),  # 周五
            remaining_shares=1000,
            price=100.0
        )

        # 可卖日期应该是周一
        assert record.available_date == date(2024, 1, 8)

    def test_buy_record_custom_available_date(self):
        """测试自定义可卖日期"""
        record = BuyRecord(
            code="600519.SH",
            shares=1000,
            buy_date=date(2024, 1, 8),
            remaining_shares=1000,
            price=100.0,
            available_date=date(2024, 1, 10)  # 自定义
        )

        assert record.available_date == date(2024, 1, 10)

    def test_is_available_property(self):
        """测试 is_available 属性"""
        # 使用 mock 或直接检查逻辑
        record = BuyRecord(
            code="600519.SH",
            shares=1000,
            buy_date=date(2024, 1, 8),
            remaining_shares=1000,
            price=100.0
        )
        # 注意：is_available 依赖 date.today()，这里只检查属性存在
        assert hasattr(record, 'is_available')


class TestPositionLot:
    """测试持仓批次"""

    def test_position_lot_creation(self):
        """测试持仓批次创建"""
        lot = PositionLot(
            code="600519.SH",
            shares=1000,
            buy_date=date(2024, 1, 8),
            available_date=date(2024, 1, 9),
            cost_price=100.0
        )

        assert lot.code == "600519.SH"
        assert lot.shares == 1000
        assert lot.remaining == 1000  # 初始时等于 shares

    def test_position_lot_remaining(self):
        """测试持仓批次剩余"""
        lot = PositionLot(
            code="600519.SH",
            shares=1000,
            buy_date=date(2024, 1, 8),
            available_date=date(2024, 1, 9),
            cost_price=100.0
        )

        lot.remaining = 500
        assert lot.remaining == 500

    def test_is_available_today_property(self):
        """测试 is_available_today 属性"""
        lot = PositionLot(
            code="600519.SH",
            shares=1000,
            buy_date=date(2024, 1, 8),
            available_date=date(2024, 1, 9),
            cost_price=100.0
        )
        # 属性应该存在
        assert hasattr(lot, 'is_available_today')


class TestT1ManagerRecordBuy:
    """测试买入记录"""

    def test_record_buy_valid(self):
        """测试有效买入"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

        assert manager.get_total_shares("600519.SH") == 1000

    def test_record_buy_invalid_shares(self):
        """测试无效买入数量"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 0, date(2024, 1, 8), 100.0)

        # 应该不记录
        assert manager.get_total_shares("600519.SH") == 0

    def test_record_buy_negative_shares(self):
        """测试负数买入数量"""
        manager = T1Manager()
        manager.record_buy("600519.SH", -100, date(2024, 1, 8), 100.0)

        # 应该不记录
        assert manager.get_total_shares("600519.SH") == 0


class TestT1ManagerGetAvailable:
    """测试获取可卖数量"""

    def test_get_available_no_position(self):
        """测试无持仓时"""
        manager = T1Manager()

        assert manager.get_available_shares("999999.SH", date(2024, 1, 8)) == 0

    def test_get_available_today_date_none(self):
        """测试日期为 None 时使用今天"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

        # current_date=None 时应该使用 date.today()
        result = manager.get_available_shares("600519.SH")
        assert isinstance(result, int)


class TestT1ManagerDeductAvailable:
    """测试扣减可卖数量"""

    def test_deduct_available_success(self):
        """测试成功扣减"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

        result = manager.deduct_available("600519.SH", 500, date(2024, 1, 9))
        assert result is True
        assert manager.get_available_shares("600519.SH", date(2024, 1, 9)) == 500

    def test_deduct_available_insufficient(self):
        """测试可卖数量不足"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

        result = manager.deduct_available("600519.SH", 2000, date(2024, 1, 9))
        assert result is False

    def test_deduct_available_same_day(self):
        """测试当日扣减失败（T+1限制）"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

        result = manager.deduct_available("600519.SH", 500, date(2024, 1, 8))
        assert result is False

    def test_deduct_available_fifo(self):
        """测试 FIFO 扣减"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)
        manager.record_buy("600519.SH", 500, date(2024, 1, 9), 102.0)

        # 周二可卖1000，周三可卖1500
        result = manager.deduct_available("600519.SH", 800, date(2024, 1, 9))
        assert result is True

        # 检查第一批次剩余
        lots = manager._position_lots["600519.SH"]
        assert lots[0].remaining == 200

    def test_deduct_available_date_none(self):
        """测试日期为 None"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2020, 1, 1), 100.0)  # 很久以前

        # current_date=None 时应该使用 date.today()
        result = manager.deduct_available("600519.SH", 500)
        assert result is True


class TestT1ManagerPositionInfo:
    """测试持仓信息"""

    def test_get_position_info_date_none(self):
        """测试日期为 None"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

        info = manager.get_position_info("600519.SH")
        assert info["code"] == "600519.SH"
        assert info["total_shares"] == 1000

    def test_get_position_info_average_cost(self):
        """测试平均成本计算"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)
        manager.record_buy("600519.SH", 1000, date(2024, 1, 9), 110.0)

        info = manager.get_position_info("600519.SH", date(2024, 1, 10))
        # 平均成本 = (100*1000 + 110*1000) / 2000 = 105
        assert info["average_cost"] == 105.0

    def test_get_position_info_lots(self):
        """测试批次信息"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)

        info = manager.get_position_info("600519.SH", date(2024, 1, 9))
        assert len(info["lots"]) == 1
        assert info["lots"][0]["shares"] == 1000


class TestT1ManagerClear:
    """测试清空记录"""

    def test_clear(self):
        """测试清空"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)
        manager.record_buy("000001.SZ", 2000, date(2024, 1, 8), 10.0)

        manager.clear()

        assert manager.get_total_shares("600519.SH") == 0
        assert manager.get_total_shares("000001.SZ") == 0


class TestT1ManagerGetAllPositions:
    """测试获取所有持仓"""

    def test_get_all_positions(self):
        """测试获取所有持仓"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)
        manager.record_buy("000001.SZ", 2000, date(2024, 1, 8), 10.0)

        positions = manager.get_all_positions(date(2024, 1, 9))

        assert len(positions) == 2
        assert "600519.SH" in positions
        assert "000001.SZ" in positions

    def test_get_all_positions_after_sell(self):
        """测试卖出后获取所有持仓"""
        manager = T1Manager()
        manager.record_buy("600519.SH", 1000, date(2024, 1, 8), 100.0)
        manager.deduct_available("600519.SH", 1000, date(2024, 1, 9))

        positions = manager.get_all_positions(date(2024, 1, 10))

        # 卖出后应该没有持仓
        assert len(positions) == 0


class TestTradingDayFunctions:
    """测试交易日函数"""

    def test_is_trading_day_weekday(self):
        """测试工作日是交易日"""
        assert is_trading_day(date(2024, 1, 8)) is True   # 周一
        assert is_trading_day(date(2024, 1, 9)) is True   # 周二
        assert is_trading_day(date(2024, 1, 10)) is True  # 周三
        assert is_trading_day(date(2024, 1, 11)) is True  # 周四
        assert is_trading_day(date(2024, 1, 12)) is True  # 周五

    def test_is_trading_day_weekend(self):
        """测试周末不是交易日"""
        assert is_trading_day(date(2024, 1, 6)) is False  # 周六
        assert is_trading_day(date(2024, 1, 7)) is False  # 周日

    def test_get_next_trading_day_from_weekday(self):
        """测试工作日后的下一交易日"""
        next_day = get_next_trading_day(date(2024, 1, 8))  # 周一
        assert next_day == date(2024, 1, 9)  # 周二

    def test_get_next_trading_day_from_friday(self):
        """测试周五后的下一交易日"""
        next_day = get_next_trading_day(date(2024, 1, 5))  # 周五
        assert next_day == date(2024, 1, 8)  # 周一

    def test_get_next_trading_day_from_saturday(self):
        """测试周六后的下一交易日"""
        next_day = get_next_trading_day(date(2024, 1, 6))  # 周六
        assert next_day == date(2024, 1, 8)  # 周一

    def test_get_next_trading_day_from_sunday(self):
        """测试周日后的下一交易日"""
        next_day = get_next_trading_day(date(2024, 1, 7))  # 周日
        assert next_day == date(2024, 1, 8)  # 周一


class TestGetT1ManagerSingleton:
    """测试单例获取"""

    def test_get_t1_manager_returns_instance(self):
        """测试获取单例"""
        manager = get_t1_manager()
        assert isinstance(manager, T1Manager)

    def test_get_t1_manager_same_instance(self):
        """测试多次获取返回同一实例"""
        manager1 = get_t1_manager()
        manager2 = get_t1_manager()
        assert manager1 is manager2


if __name__ == "__main__":
    main()
