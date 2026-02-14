"""
新股管理模块测试

测试新股交易特殊规则
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.new_stock import (
    NewStockManager,
    NewStockType,
    NewStockRecord,
    NEW_STOCK_NO_LIMIT_DAYS,
    get_new_stock_manager,
)


def test_no_limit_days_config():
    """测试无涨跌停限制天数配置"""
    print("=" * 60)
    print("测试无涨跌停限制天数配置")
    print("=" * 60)

    print(f"\n各板块无涨跌停限制天数:")
    for stock_type, days in NEW_STOCK_NO_LIMIT_DAYS.items():
        print(f"  {stock_type.value:10s}: {days}天")

    print()


def test_add_new_stock():
    """测试添加新股"""
    print("=" * 60)
    print("测试添加新股")
    print("=" * 60)

    manager = NewStockManager()

    # 添加主板新股
    record1 = manager.add_new_stock(
        code="601328.SH",
        list_date="20240115",
        name="中国人寿",
        issue_price=30.0,
    )

    print(f"\n主板新股:")
    print(f"  代码: {record1.code}")
    print(f"  上市日期: {record1.list_date}")
    print(f"  类型: {record1.stock_type.value}")
    print(f"  无限制天数: {record1.no_limit_days}")
    print(f"  限制开始日期: {record1.get_no_limit_end_date()}")

    # 添加科创板新股
    record2 = manager.add_new_stock(
        code="688123.SH",
        list_date="20240115",
        name="科创板新股",
    )

    print(f"\n科创板新股:")
    print(f"  代码: {record2.code}")
    print(f"  类型: {record2.stock_type.value}")
    print(f"  无限制天数: {record2.no_limit_days}")

    print()


def test_stock_type_detection():
    """测试股票类型自动检测"""
    print("=" * 60)
    print("测试股票类型自动检测")
    print("=" * 60)

    manager = NewStockManager()

    test_cases = [
        ("600519.SH", NewStockType.MAIN, "主板"),
        ("601318.SH", NewStockType.MAIN, "主板"),
        ("000001.SZ", NewStockType.MAIN, "主板（深圳）"),
        ("300750.SZ", NewStockType.CHINEXT, "创业板"),
        ("301123.SZ", NewStockType.CHINEXT, "创业板"),
        ("688981.SH", NewStockType.STAR, "科创板"),
        ("830799.BJ", NewStockType.BSE, "北交所"),
    ]

    print()
    for code, expected, desc in test_cases:
        record = manager.add_new_stock(code, "20240101")
        status = "✅" if record.stock_type == expected else "❌"
        print(f"  {status} {code} ({desc}): {record.stock_type.value}")

    print()


def test_is_in_no_limit_period():
    """测试无涨跌停限制期判断"""
    print("=" * 60)
    print("测试无涨跌停限制期判断")
    print("=" * 60)

    manager = NewStockManager()

    # 主板新股：2024年1月15日上市，前5日无限制
    manager.add_new_stock("601328.SH", "20240115", stock_type=NewStockType.MAIN)

    test_cases = [
        ("20240114", False, "上市前一天"),
        ("20240115", True, "上市首日"),
        ("20240117", True, "第3天"),
        ("20240119", True, "第5天（最后一天）"),
        ("20240122", False, "第6天（开始限制）"),
    ]

    print(f"\n主板新股 601328.SH，上市日期: 20240115")
    print(f"无涨跌停限制期: 前5个交易日")
    print("-" * 40)

    for date, expected, desc in test_cases:
        result = manager.is_in_no_limit_period("601328.SH", date)
        status = "✅" if result == expected else "❌"
        state = "无限制" if result else "有限制"
        print(f"  {status} {date} ({desc}): {state}")

    # 北交所新股：首日无限制
    manager.add_new_stock("830799.BJ", "20240115", stock_type=NewStockType.BSE)

    print(f"\n北交所新股 830799.BJ，上市日期: 20240115")
    print(f"无涨跌停限制期: 首日")
    print("-" * 40)

    test_cases_bse = [
        ("20240115", True, "上市首日"),
        ("20240116", False, "第2天（开始限制）"),
    ]

    for date, expected, desc in test_cases_bse:
        result = manager.is_in_no_limit_period("830799.BJ", date)
        status = "✅" if result == expected else "❌"
        state = "无限制" if result else "有限制"
        print(f"  {status} {date} ({desc}): {state}")

    print()


def test_days_remaining():
    """测试剩余无限制天数"""
    print("=" * 60)
    print("测试剩余无限制天数")
    print("=" * 60)

    manager = NewStockManager()

    # 科创板新股
    manager.add_new_stock("688123.SH", "20240115", stock_type=NewStockType.STAR)

    print(f"\n科创板新股 688123.SH，上市日期: 20240115")
    print(f"无涨跌停限制期: 前5个交易日")
    print("-" * 40)

    test_dates = ["20240115", "20240117", "20240119", "20240122"]

    for date in test_dates:
        remaining = manager.get_days_remaining("688123.SH", date)
        in_period = manager.is_in_no_limit_period("688123.SH", date)
        print(f"  {date}: 剩余{remaining}天, {'无限制' if in_period else '有限制'}")

    print()


def test_with_trading_days():
    """测试使用交易日历"""
    print("=" * 60)
    print("测试使用交易日历（精确计算）")
    print("=" * 60)

    manager = NewStockManager()

    # 假设交易日历（跳过周末）
    trading_days = [
        "20240115",  # 周一
        "20240116",  # 周二
        "20240117",  # 周三
        "20240118",  # 周四
        "20240119",  # 周五
        # 跳过周末
        "20240122",  # 周一
        "20240123",  # 周二
    ]

    manager.add_new_stock("600000.SH", "20240115", stock_type=NewStockType.MAIN)

    print(f"\n主板新股 600000.SH，上市日期: 20240115")
    print(f"使用交易日历精确计算")
    print("-" * 40)

    for date in trading_days:
        remaining = manager.get_days_remaining("600000.SH", date, trading_days)
        in_period = manager.is_in_no_limit_period("600000.SH", date, trading_days)
        print(f"  {date}: 剩余{remaining}天, {'无限制' if in_period else '有限制'}")

    print()


def test_get_recent_new_stocks():
    """测试获取最近新股"""
    print("=" * 60)
    print("测试获取最近新股")
    print("=" * 60)

    manager = NewStockManager()

    # 添加多只新股
    manager.add_new_stock("600001.SH", "20240110", name="股票A")
    manager.add_new_stock("600002.SH", "20240115", name="股票B")
    manager.add_new_stock("600003.SH", "20240120", name="股票C")
    manager.add_new_stock("600004.SH", "20240125", name="股票D")
    manager.add_new_stock("600005.SH", "20240201", name="股票E")

    # 获取20240120最近15天的新股
    recent = manager.get_recent_new_stocks("20240120", days=15)

    print(f"\n20240120 最近15天上市的新股:")
    for record in recent:
        print(f"  - {record.code}: {record.name}, 上市日期={record.list_date}")

    print()


def test_cleanup():
    """测试清理过期记录"""
    print("=" * 60)
    print("测试清理过期记录")
    print("=" * 60)

    manager = NewStockManager()

    manager.add_new_stock("600001.SH", "20231201")  # 2个月前
    manager.add_new_stock("600002.SH", "20240101")  # 20天前
    manager.add_new_stock("600003.SH", "20240120")  # 今天

    print(f"\n清理前: {len(manager._new_stocks)} 条记录")

    # 清理30天前的记录
    manager.cleanup_old_records("20240120", keep_days=30)

    print(f"清理后: {len(manager._new_stocks)} 条记录")

    remaining = [r.code for r in manager._new_stocks.values()]
    print(f"剩余: {remaining}")

    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始新股管理模块测试")
    print("=" * 60)
    print()

    test_no_limit_days_config()
    test_add_new_stock()
    test_stock_type_detection()
    test_is_in_no_limit_period()
    test_days_remaining()
    test_with_trading_days()
    test_get_recent_new_stocks()
    test_cleanup()

    print("=" * 60)
    print("所有测试完成 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
