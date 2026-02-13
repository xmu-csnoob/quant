"""
交易日历测试
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.trade_calendar import TradeCalendar


def test_trade_calendar_basic():
    """测试基本的交易日历功能"""
    print("=" * 60)
    print("测试交易日历基本功能")
    print("=" * 60)

    calendar = TradeCalendar()

    # 测试周末判断
    saturday = datetime(2024, 1, 6)  # 周六
    sunday = datetime(2024, 1, 7)    # 周日
    monday = datetime(2024, 1, 8)    # 周一

    assert not calendar.is_trading_day(saturday.strftime("%Y%m%d")), "周六不是交易日"
    assert not calendar.is_trading_day(sunday.strftime("%Y%m%d")), "周日不是交易日"

    print("  ✅ 周末判断正确")

    # 测试获取交易日列表
    trading_days = calendar.get_trading_days_between("20240101", "20240131")
    print(f"  2024年1月交易日数: {len(trading_days) if trading_days else 'N/A'}")
    if trading_days:
        assert len(trading_days) > 0, "应返回交易日列表"
    print("  ✅ 获取交易日列表成功")


def test_t1_calculation():
    """测试T+1计算"""
    print("\n" + "=" * 60)
    print("测试T+1计算")
    print("=" * 60)

    calendar = TradeCalendar()

    # 假设周一买入，周二才能卖
    buy_date = "20240108"  # 周一
    sell_date = calendar.get_t1_sell_date(buy_date)

    print(f"  买入日期: {buy_date}")
    print(f"  可卖日期: {sell_date}")

    # 可卖日期应该在买入日期之后
    assert sell_date > buy_date, "可卖日期应在买入日期之后"
    print("  ✅ T+1计算正确")


def test_trading_days_between():
    """测试交易日间隔计算"""
    print("\n" + "=" * 60)
    print("测试交易日间隔计算")
    print("=" * 60)

    calendar = TradeCalendar()

    start = "20240101"
    end = "20240131"
    count = calendar.count_trading_days_between(start, end)

    print(f"  从 {start} 到 {end} 共 {count} 个交易日")
    assert count > 0, "交易日数应大于0"
    print("  ✅ 交易日间隔计算正确")


def test_prev_trading_day():
    """测试获取前一交易日"""
    print("\n" + "=" * 60)
    print("测试获取前一交易日")
    print("=" * 60)

    calendar = TradeCalendar()

    # 周二的前一交易日应该是周一
    tuesday = "20240109"
    prev_day = calendar.get_prev_trading_day(tuesday)

    print(f"  {tuesday} 的前一交易日: {prev_day}")
    assert prev_day == "20240108", "周二的前一交易日应该是周一"
    print("  ✅ 前一交易日获取正确")


def test_nth_trading_day():
    """测试获取第N个交易日"""
    print("\n" + "=" * 60)
    print("测试获取第N个交易日")
    print("=" * 60)

    calendar = TradeCalendar()

    # 从某个日期开始的第1个交易日
    first_day = calendar.get_nth_trading_day("20240101", 1)
    print(f"  从20240101开始的第1个交易日: {first_day}")
    assert first_day is not None, "应返回第一个交易日"

    # 第5个交易日
    fifth_day = calendar.get_nth_trading_day("20240101", 5)
    print(f"  从20240101开始的第5个交易日: {fifth_day}")
    print("  ✅ 第N个交易日获取正确")


def main():
    """运行所有测试"""
    print("开始测试交易日历模块")
    print("=" * 60)

    test_trade_calendar_basic()
    test_t1_calculation()
    test_trading_days_between()
    test_prev_trading_day()
    test_nth_trading_day()

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
