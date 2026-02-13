"""
涨跌停限制检查测试
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.price_limit import PriceLimitChecker, BoardType


def test_board_type_identification():
    """测试板块类型识别"""
    print("=" * 60)
    print("测试板块类型识别")
    print("=" * 60)

    checker = PriceLimitChecker()

    # 主板
    assert checker.get_board_type("600519.SH") == BoardType.MAIN, "600519应为主板"
    assert checker.get_board_type("000001.SZ") == BoardType.MAIN, "000001应为主板"
    assert checker.get_board_type("601318.SH") == BoardType.MAIN, "601318应为主板"
    print("  ✅ 主板识别正确")

    # 创业板
    assert checker.get_board_type("300750.SZ") == BoardType.CHINEXT, "300750应为创业板"
    assert checker.get_board_type("301269.SZ") == BoardType.CHINEXT, "301269应为创业板"
    print("  ✅ 创业板识别正确")

    # 科创板
    assert checker.get_board_type("688981.SH") == BoardType.STAR, "688981应为科创板"
    print("  ✅ 科创板识别正确")

    # 北交所
    assert checker.get_board_type("830799.BJ") == BoardType.BSE, "830799应为北交所"
    print("  ✅ 北交所识别正确")


def test_price_limit_ratio():
    """测试涨跌停比例"""
    print("\n" + "=" * 60)
    print("测试涨跌停比例")
    print("=" * 60)

    checker = PriceLimitChecker()

    # 主板 ±10%
    up, down = checker.get_limit_ratio("600519.SH")
    assert up == 0.10, f"主板涨停比例应为10%，实际为{up}"
    assert down == -0.10, f"主板跌停比例应为-10%，实际为{down}"
    print("  ✅ 主板涨跌停比例正确")

    # 创业板 ±20%
    up, down = checker.get_limit_ratio("300750.SZ")
    assert up == 0.20, f"创业板涨停比例应为20%，实际为{up}"
    assert down == -0.20, f"创业板跌停比例应为-20%，实际为{down}"
    print("  ✅ 创业板涨跌停比例正确")

    # 科创板 ±20%
    up, down = checker.get_limit_ratio("688981.SH")
    assert up == 0.20, f"科创板涨停比例应为20%，实际为{up}"
    print("  ✅ 科创板涨跌停比例正确")

    # 北交所 ±30%
    up, down = checker.get_limit_ratio("830799.BJ")
    assert up == 0.30, f"北交所涨停比例应为30%，实际为{up}"
    print("  ✅ 北交所涨跌停比例正确")


def test_price_limit_calculation():
    """测试涨跌停价格计算"""
    print("\n" + "=" * 60)
    print("测试涨跌停价格计算")
    print("=" * 60)

    checker = PriceLimitChecker()

    # 主板股票，前收盘100元
    prev_close = 100.0
    low, high = checker.get_valid_price_range("600519.SH", prev_close)

    print(f"  主板股票，前收盘{prev_close}元")
    print(f"  跌停价: {low}, 涨停价: {high}")

    assert low == 90.0, f"跌停价应为90，实际为{low}"
    assert high == 110.0, f"涨停价应为110，实际为{high}"
    print("  ✅ 主板涨跌停价格计算正确")

    # 创业板股票，前收盘50元
    prev_close = 50.0
    low, high = checker.get_valid_price_range("300750.SZ", prev_close)

    print(f"  创业板股票，前收盘{prev_close}元")
    print(f"  跌停价: {low}, 涨停价: {high}")

    assert low == 40.0, f"跌停价应为40，实际为{low}"
    assert high == 60.0, f"涨停价应为60，实际为{high}"
    print("  ✅ 创业板涨跌停价格计算正确")


def test_limit_up_down_detection():
    """测试涨跌停检测"""
    print("\n" + "=" * 60)
    print("测试涨跌停检测")
    print("=" * 60)

    checker = PriceLimitChecker()

    # 涨停检测
    assert checker.is_limit_up("600519.SH", 110.0, 100.0), "110元应为涨停"
    assert not checker.is_limit_up("600519.SH", 109.0, 100.0), "109元不应为涨停"
    print("  ✅ 涨停检测正确")

    # 跌停检测
    assert checker.is_limit_down("600519.SH", 90.0, 100.0), "90元应为跌停"
    assert not checker.is_limit_down("600519.SH", 91.0, 100.0), "91元不应为跌停"
    print("  ✅ 跌停检测正确")


def test_st_stock():
    """测试ST股票"""
    print("\n" + "=" * 60)
    print("测试ST股票识别")
    print("=" * 60)

    checker = PriceLimitChecker()

    # ST股票识别
    assert checker.is_st_stock("000001.SZ", "ST某某"), "ST开头的应为ST股票"
    assert checker.is_st_stock("000001.SZ", "*ST某某"), "*ST开头的应为ST股票"
    assert not checker.is_st_stock("000001.SZ", "正常股票"), "正常股票不应识别为ST"
    print("  ✅ ST股票识别正确")

    # ST股票涨跌停比例
    up, down = checker.get_limit_ratio("000001.SZ", name="ST某某")
    assert up == 0.05, f"ST股票涨停比例应为5%，实际为{up}"
    assert down == -0.05, f"ST股票跌停比例应为-5%，实际为{down}"
    print("  ✅ ST股票涨跌停比例正确")


def test_can_buy_sell():
    """测试买卖检查"""
    print("\n" + "=" * 60)
    print("测试买卖检查")
    print("=" * 60)

    checker = PriceLimitChecker()

    # 涨停无法买入
    can_buy, reason = checker.can_buy("600519.SH", 110.0, 100.0)
    assert not can_buy, "涨停应无法买入"
    print(f"  涨停买入: {can_buy}, 原因: {reason}")
    print("  ✅ 涨停买入检查正确")

    # 跌停无法卖出
    can_sell, reason = checker.can_sell("600519.SH", 90.0, 100.0)
    assert not can_sell, "跌停应无法卖出"
    print(f"  跌停卖出: {can_sell}, 原因: {reason}")
    print("  ✅ 跌停卖出检查正确")

    # 正常可以买卖
    can_buy, _ = checker.can_buy("600519.SH", 105.0, 100.0)
    can_sell, _ = checker.can_sell("600519.SH", 95.0, 100.0)
    assert can_buy, "正常价格应可买入"
    assert can_sell, "正常价格应可卖出"
    print("  ✅ 正常买卖检查正确")


def test_price_adjustment():
    """测试价格调整"""
    print("\n" + "=" * 60)
    print("测试价格调整到涨跌停范围")
    print("=" * 60)

    checker = PriceLimitChecker()

    # 超过涨停价
    adjusted = checker.adjust_price_to_limit("600519.SH", 115.0, 100.0)
    assert adjusted == 110.0, f"超过涨停价应调整为110，实际为{adjusted}"
    print(f"  115 -> {adjusted} (涨停价)")
    print("  ✅ 超涨停价格调整正确")

    # 低于跌停价
    adjusted = checker.adjust_price_to_limit("600519.SH", 85.0, 100.0)
    assert adjusted == 90.0, f"低于跌停价应调整为90，实际为{adjusted}"
    print(f"  85 -> {adjusted} (跌停价)")
    print("  ✅ 超跌停价格调整正确")


def main():
    """运行所有测试"""
    print("开始测试涨跌停限制模块")
    print("=" * 60)

    test_board_type_identification()
    test_price_limit_ratio()
    test_price_limit_calculation()
    test_limit_up_down_detection()
    test_st_stock()
    test_can_buy_sell()
    test_price_adjustment()

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
