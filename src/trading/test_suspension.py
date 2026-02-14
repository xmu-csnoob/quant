"""
停牌处理模块测试

测试停牌状态管理和交易检查
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.suspension import (
    SuspensionManager,
    SuspensionReason,
    SuspensionRecord,
    get_suspension_manager,
)


def test_add_suspension():
    """测试添加停牌记录"""
    print("=" * 60)
    print("测试添加停牌记录")
    print("=" * 60)

    manager = SuspensionManager()

    # 添加停牌记录
    record = manager.add_suspension(
        code="600519.SH",
        start_date="20240115",
        reason=SuspensionReason.MAJOR_EVENT,
        prev_close=1800.0,
        description="重大资产重组",
    )

    print(f"\n添加记录:")
    print(f"  股票: {record.code}")
    print(f"  开始日期: {record.start_date}")
    print(f"  原因: {record.reason.value}")
    print(f"  停牌前收盘价: {record.prev_close}")
    print(f"  是否活跃: {record.is_active}")

    status = "✅" if record.is_active else "❌"
    print(f"  状态: {status}")

    print()


def test_is_suspended():
    """测试停牌状态检查"""
    print("=" * 60)
    print("测试停牌状态检查")
    print("=" * 60)

    manager = SuspensionManager()

    # 添加停牌记录：1月15日-1月20日
    manager.add_suspension(
        code="600519.SH",
        start_date="20240115",
        end_date="20240120",
        reason=SuspensionReason.MAJOR_EVENT,
    )

    test_cases = [
        ("20240114", False, "停牌前一天"),
        ("20240115", True, "停牌第一天"),
        ("20240117", True, "停牌期间"),
        ("20240120", True, "复牌日（仍在停牌期内）"),
        ("20240121", False, "复牌后一天"),
    ]

    print(f"\n停牌期间: 20240115 - 20240120")
    print("-" * 40)

    for date, expected, desc in test_cases:
        result = manager.is_suspended("600519.SH", date)
        status = "✅" if result == expected else "❌"
        suspended = "停牌" if result else "正常"
        print(f"  {status} {date} ({desc}): {suspended}")

    print()


def test_resume():
    """测试复牌"""
    print("=" * 60)
    print("测试复牌")
    print("=" * 60)

    manager = SuspensionManager()

    # 添加持续停牌
    manager.add_suspension(
        code="300750.SZ",
        start_date="20240110",
        reason=SuspensionReason.REORGANIZATION,
        prev_close=200.0,
    )

    print(f"\n初始状态:")
    print(f"  20240115 停牌: {manager.is_suspended('300750.SZ', '20240115')}")

    # 设置复牌
    success = manager.resume("300750.SZ", "20240125", prev_close=200.0)

    print(f"\n复牌操作: {'成功' if success else '失败'}")
    print(f"  20240124 停牌: {manager.is_suspended('300750.SZ', '20240124')}")
    print(f"  20240125 停牌: {manager.is_suspended('300750.SZ', '20240125')}")
    print(f"  20240126 停牌: {manager.is_suspended('300750.SZ', '20240126')}")

    print()


def test_can_trade():
    """测试交易检查"""
    print("=" * 60)
    print("测试交易检查")
    print("=" * 60)

    manager = SuspensionManager()

    manager.add_suspension(
        code="688981.SH",
        start_date="20240115",
        end_date="20240120",
        reason=SuspensionReason.ABNORMAL_MOVE,
    )

    test_cases = [
        ("688981.SH", "20240114"),
        ("688981.SH", "20240117"),
        ("688981.SH", "20240121"),
        ("600000.SH", "20240117"),  # 未停牌股票
    ]

    print()
    for code, date in test_cases:
        can_trade, reason = manager.can_trade(code, date)
        status = "✅" if can_trade else "❌"
        print(f"  {status} {code} {date}: {reason}")

    print()


def test_prev_close_for_resumption():
    """测试复牌后的前收盘价获取"""
    print("=" * 60)
    print("测试复牌后前收盘价获取")
    print("=" * 60)

    manager = SuspensionManager()

    # 添加停牌记录，停牌前收盘价1800
    manager.add_suspension(
        code="600519.SH",
        start_date="20240110",
        end_date="20240120",
        reason=SuspensionReason.MAJOR_EVENT,
        prev_close=1800.0,
    )

    # 复牌日（20240121）的前收盘价应该是停牌前的1800
    prev_close = manager.get_prev_close_for_date("600519.SH", "20240120")
    print(f"\n停牌前收盘价: 1800.0")
    print(f"复牌日 (20240120) 前收盘价: {prev_close}")

    status = "✅" if prev_close == 1800.0 else "❌"
    print(f"验证: {status}")

    print()


def test_multiple_suspensions():
    """测试多次停牌"""
    print("=" * 60)
    print("测试多次停牌")
    print("=" * 60)

    manager = SuspensionManager()

    # 第一次停牌
    manager.add_suspension(
        code="000001.SZ",
        start_date="20240110",
        end_date="20240115",
        reason=SuspensionReason.MAJOR_EVENT,
    )

    # 第二次停牌
    manager.add_suspension(
        code="000001.SZ",
        start_date="20240201",
        end_date="20240205",
        reason=SuspensionReason.REGULATORY,
    )

    test_dates = [
        ("20240109", False),
        ("20240112", True),
        ("20240116", False),
        ("20240203", True),
        ("20240206", False),
    ]

    print()
    for date, expected in test_dates:
        result = manager.is_suspended("000001.SZ", date)
        status = "✅" if result == expected else "❌"
        state = "停牌" if result else "正常"
        print(f"  {status} {date}: {state}")

    print()


def test_get_active_suspensions():
    """测试获取活跃停牌"""
    print("=" * 60)
    print("测试获取活跃停牌")
    print("=" * 60)

    manager = SuspensionManager()

    manager.add_suspension("600519.SH", "20240115", end_date="20240120")
    manager.add_suspension("300750.SZ", "20240115", end_date="20240118")
    manager.add_suspension("688981.SH", "20240117", end_date="20240125")

    # 检查20240117的活跃停牌
    active = manager.get_active_suspensions("20240117")

    print(f"\n20240117 活跃停牌股票:")
    for record in active:
        print(f"  - {record.code}: {record.start_date} - {record.end_date}")

    status = "✅" if len(active) == 3 else "❌"
    print(f"\n验证（应有3只）: {status}")

    print()


def test_suspension_record():
    """测试停牌记录数据类"""
    print("=" * 60)
    print("测试停牌记录数据类")
    print("=" * 60)

    # 有结束日期的记录
    record1 = SuspensionRecord(
        code="600519.SH",
        start_date="20240115",
        end_date="20240120",
        reason=SuspensionReason.MAJOR_EVENT,
        prev_close=1800.0,
    )

    # 无结束日期的记录（持续停牌）
    record2 = SuspensionRecord(
        code="300750.SZ",
        start_date="20240115",
        end_date=None,
        reason=SuspensionReason.REORGANIZATION,
        prev_close=200.0,
    )

    print(f"\n记录1 (有结束日期):")
    print(f"  is_active: {record1.is_active}")
    print(f"  covers_date('20240117'): {record1.covers_date('20240117')}")
    print(f"  covers_date('20240125'): {record1.covers_date('20240125')}")

    print(f"\n记录2 (持续停牌):")
    print(f"  is_active: {record2.is_active}")
    print(f"  covers_date('20240301'): {record2.covers_date('20240301')}")

    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始停牌处理模块测试")
    print("=" * 60)
    print()

    test_add_suspension()
    test_is_suspended()
    test_resume()
    test_can_trade()
    test_prev_close_for_resumption()
    test_multiple_suspensions()
    test_get_active_suspensions()
    test_suspension_record()

    print("=" * 60)
    print("所有测试完成 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
