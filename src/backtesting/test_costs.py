"""
交易成本和滑点模块测试

测试内容：
1. TransactionCostCalculator 成本计算
2. SlippageModel 滑点模型
3. SimpleBacktester 集成测试
"""

import sys
from pathlib import Path
from decimal import Decimal

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting.costs import (
    CostConfig,
    CostBreakdown,
    TransactionCostCalculator,
    TradeSide,
    Market,
    calculate_cost,
    calculate_cost_with_code,
)
from src.backtesting.slippage import (
    NoSlippage,
    FixedSlippage,
    VolumeBasedSlippage,
    RandomSlippage,
    MarketImpactSlippage,
)


def test_cost_calculator():
    """测试成本计算器"""
    print("=" * 60)
    print("测试交易成本计算器")
    print("=" * 60)

    config = CostConfig.default()
    calculator = TransactionCostCalculator(config)

    # 测试买入成本
    amount = Decimal("100000")  # 10万元
    buy_cost = calculator.calculate_buy_cost(amount)
    print(f"\n买入 {amount} 元:")
    print(f"  佣金: {buy_cost.commission:.2f}")
    print(f"  印花税: {buy_cost.stamp_duty:.2f}")
    print(f"  过户费: {buy_cost.transfer_fee:.2f}")
    print(f"  总成本: {buy_cost.total:.2f}")

    # 验证买入成本
    expected_commission = max(amount * Decimal("0.00025"), Decimal("5"))
    expected_transfer = amount * Decimal("0.00001")
    expected_total = expected_commission + expected_transfer

    assert buy_cost.commission == expected_commission, "买入佣金计算错误"
    assert buy_cost.stamp_duty == Decimal("0"), "买入不应有印花税"
    assert buy_cost.total == expected_total, "买入总成本计算错误"
    print("  ✅ 买入成本计算正确")

    # 测试卖出成本
    sell_cost = calculator.calculate_sell_cost(amount)
    print(f"\n卖出 {amount} 元:")
    print(f"  佣金: {sell_cost.commission:.2f}")
    print(f"  印花税: {sell_cost.stamp_duty:.2f}")
    print(f"  过户费: {sell_cost.transfer_fee:.2f}")
    print(f"  总成本: {sell_cost.total:.2f}")

    # 验证卖出成本
    expected_stamp_duty = amount * Decimal("0.001")
    expected_sell_total = expected_commission + expected_stamp_duty + expected_transfer

    assert sell_cost.stamp_duty == expected_stamp_duty, "卖出印花税计算错误"
    assert sell_cost.total == expected_sell_total, "卖出总成本计算错误"
    print("  ✅ 卖出成本计算正确")

    # 测试往返成本
    round_trip = calculator.estimate_round_trip_cost(amount)
    print(f"\n往返交易成本: {round_trip:.2f}")
    print(f"  占比: {float(round_trip / amount) * 100:.4f}%")

    # 测试有效费率
    buy_rate = calculator.get_effective_buy_rate()
    sell_rate = calculator.get_effective_sell_rate()
    print(f"\n买入有效费率: {float(buy_rate) * 100:.4f}%")
    print(f"卖出有效费率: {float(sell_rate) * 100:.4f}%")

    print("\n✅ 成本计算器基础测试通过")


def test_market_identification():
    """测试市场识别"""
    print("\n" + "=" * 60)
    print("测试市场识别功能")
    print("=" * 60)

    # 测试上海市场股票代码
    sh_codes = ["600000", "600000.SH", "sh600000", "601318", "603259", "688981"]
    for code in sh_codes:
        market = TransactionCostCalculator.get_market(code)
        is_sh = TransactionCostCalculator.is_shanghai_market(code)
        print(f"  {code}: {market.value}, 是否上海: {is_sh}")
        assert market == Market.SSE, f"{code} 应该识别为上海市场"
        assert is_sh, f"{code} 应该是上海市场"
    print("  ✅ 上海市场识别正确")

    # 测试深圳市场股票代码
    sz_codes = ["000001", "000001.SZ", "sz000001", "002415", "300750", "301269"]
    for code in sz_codes:
        market = TransactionCostCalculator.get_market(code)
        is_sh = TransactionCostCalculator.is_shanghai_market(code)
        print(f"  {code}: {market.value}, 是否上海: {is_sh}")
        assert market == Market.SZSE, f"{code} 应该识别为深圳市场"
        assert not is_sh, f"{code} 不应该是上海市场"
    print("  ✅ 深圳市场识别正确")

    # 测试北京市场股票代码
    bj_codes = ["830799", "430047"]
    for code in bj_codes:
        market = TransactionCostCalculator.get_market(code)
        print(f"  {code}: {market.value}")
        assert market == Market.BSE, f"{code} 应该识别为北京市场"
    print("  ✅ 北京市场识别正确")

    print("\n✅ 市场识别测试通过")


def test_transfer_fee_by_market():
    """测试根据市场区分过户费"""
    print("\n" + "=" * 60)
    print("测试根据市场区分过户费")
    print("=" * 60)

    config = CostConfig.default()
    calculator = TransactionCostCalculator(config)
    amount = Decimal("100000")  # 10万元

    # 上海市场股票（有过户费）
    sh_code = "600000.SH"
    sh_buy_cost = calculator.calculate_buy_cost_with_code(sh_code, amount)
    sh_sell_cost = calculator.calculate_sell_cost_with_code(sh_code, amount)

    print(f"\n上海市场股票 {sh_code}:")
    print(f"  买入过户费: {sh_buy_cost.transfer_fee:.2f}")
    print(f"  卖出过户费: {sh_sell_cost.transfer_fee:.2f}")

    expected_transfer = amount * config.transfer_fee_rate
    assert sh_buy_cost.transfer_fee == expected_transfer, "上海市场买入过户费错误"
    assert sh_sell_cost.transfer_fee == expected_transfer, "上海市场卖出过户费错误"
    print("  ✅ 上海市场过户费计算正确")

    # 深圳市场股票（无过户费）
    sz_code = "000001.SZ"
    sz_buy_cost = calculator.calculate_buy_cost_with_code(sz_code, amount)
    sz_sell_cost = calculator.calculate_sell_cost_with_code(sz_code, amount)

    print(f"\n深圳市场股票 {sz_code}:")
    print(f"  买入过户费: {sz_buy_cost.transfer_fee:.2f}")
    print(f"  卖出过户费: {sz_sell_cost.transfer_fee:.2f}")

    assert sz_buy_cost.transfer_fee == Decimal("0"), "深圳市场买入不应有过户费"
    assert sz_sell_cost.transfer_fee == Decimal("0"), "深圳市场卖出不应有过户费"
    print("  ✅ 深圳市场过户费计算正确（为0）")

    # 对比成本差异
    print(f"\n成本差异对比（{amount}元往返）:")
    sh_round_trip = calculator.estimate_round_trip_cost_with_code(sh_code, amount)
    sz_round_trip = calculator.estimate_round_trip_cost_with_code(sz_code, amount)
    print(f"  上海市场往返成本: {sh_round_trip:.2f}")
    print(f"  深圳市场往返成本: {sz_round_trip:.2f}")
    print(f"  差异: {sh_round_trip - sz_round_trip:.2f} (仅过户费差异)")

    expected_diff = expected_transfer * 2  # 买入+卖出各一次过户费
    assert sh_round_trip - sz_round_trip == expected_diff, "成本差异应等于往返过户费"
    print("  ✅ 成本差异计算正确")

    # 测试有效费率
    sh_buy_rate = calculator.get_effective_buy_rate_with_code(sh_code)
    sz_buy_rate = calculator.get_effective_buy_rate_with_code(sz_code)
    print(f"\n买入有效费率:")
    print(f"  上海市场: {float(sh_buy_rate) * 100:.4f}%")
    print(f"  深圳市场: {float(sz_buy_rate) * 100:.4f}%")
    print(f"  差异: {float(sh_buy_rate - sz_buy_rate) * 100:.4f}%")
    assert sh_buy_rate > sz_buy_rate, "上海市场费率应高于深圳市场"

    print("\n✅ 过户费市场区分测试通过")


def test_convenience_functions():
    """测试便捷函数"""
    print("\n" + "=" * 60)
    print("测试便捷函数")
    print("=" * 60)

    amount = 100000  # 10万元

    # 测试不区分市场的函数（旧API）
    cost_old = calculate_cost(TradeSide.BUY, amount)
    print(f"\n旧API买入成本（不区分市场）: {cost_old:.2f}")

    # 测试区分市场的函数（新API）
    sh_cost = calculate_cost_with_code(TradeSide.BUY, "600000.SH", amount)
    sz_cost = calculate_cost_with_code(TradeSide.BUY, "000001.SZ", amount)
    print(f"新API上海市场买入成本: {sh_cost:.2f}")
    print(f"新API深圳市场买入成本: {sz_cost:.2f}")

    # 旧API应该等于上海市场（包含过户费）
    assert cost_old == sh_cost, "旧API应该返回与上海市场相同的成本"
    print("  ✅ 旧API兼容性正确")

    # 深圳市场应该比旧API便宜
    assert sz_cost < cost_old, "深圳市场成本应低于旧API"
    print("  ✅ 新API区分市场正确")

    print("\n✅ 便捷函数测试通过")


def test_slippage_models():
    """测试滑点模型"""
    print("\n" + "=" * 60)
    print("测试滑点模型")
    print("=" * 60)

    price = Decimal("10.00")

    # 1. 无滑点
    print("\n1. 无滑点模型:")
    model = NoSlippage()
    result = model.apply_slippage(price, TradeSide.BUY)
    print(f"   买入: {price} -> {result.adjusted_price}")
    assert result.adjusted_price == price
    result = model.apply_slippage(price, TradeSide.SELL)
    print(f"   卖出: {price} -> {result.adjusted_price}")
    assert result.adjusted_price == price
    print("   ✅ 无滑点测试通过")

    # 2. 固定滑点
    print("\n2. 固定滑点模型 (0.1%):")
    model = FixedSlippage(slippage_rate=Decimal("0.001"))
    result = model.apply_slippage(price, TradeSide.BUY)
    print(f"   买入: {price} -> {result.adjusted_price}")
    assert result.adjusted_price == price * Decimal("1.001")
    result = model.apply_slippage(price, TradeSide.SELL)
    print(f"   卖出: {price} -> {result.adjusted_price}")
    assert result.adjusted_price == price * Decimal("0.999")
    print("   ✅ 固定滑点测试通过")

    # 3. 成交量滑点
    print("\n3. 成交量滑点模型:")
    model = VolumeBasedSlippage(base_rate=Decimal("0.0005"))
    # 小单
    result = model.apply_slippage(price, TradeSide.BUY, Decimal("1000"), Decimal("100000"))
    print(f"   小单买入 (1%成交量): {price} -> {result.adjusted_price:.4f}")
    # 大单
    result = model.apply_slippage(price, TradeSide.BUY, Decimal("50000"), Decimal("100000"))
    print(f"   大单买入 (50%成交量): {price} -> {result.adjusted_price:.4f}")
    print("   ✅ 成交量滑点测试通过")

    # 4. 随机滑点
    print("\n4. 随机滑点模型:")
    model = RandomSlippage(min_rate=Decimal("0.0005"), max_rate=Decimal("0.002"), seed=42)
    results = []
    for i in range(5):
        result = model.apply_slippage(price, TradeSide.BUY)
        results.append(result.adjusted_price)
        print(f"   随机{i+1}: {price} -> {result.adjusted_price:.4f} (滑点: {float(result.slippage_rate)*100:.4f}%)")
    print("   ✅ 随机滑点测试通过")

    # 5. 市场冲击滑点
    print("\n5. 市场冲击滑点模型:")
    model = MarketImpactSlippage()
    result = model.apply_slippage(price, TradeSide.BUY, Decimal("10000"), Decimal("50000"))
    print(f"   买入: {price} -> {result.adjusted_price:.4f} (滑点: {float(result.slippage_rate)*100:.4f}%)")
    print("   ✅ 市场冲击滑点测试通过")

    print("\n✅ 滑点模型测试通过")


def test_backtest_with_costs():
    """测试带回测成本的回测"""
    print("\n" + "=" * 60)
    print("测试带成本和滑点的回测")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    from src.backtesting.simple_backtester import SimpleBacktester
    from src.backtesting.costs import CostConfig
    from src.backtesting.slippage import FixedSlippage, NoSlippage
    from src.strategies.base import BaseStrategy, Signal, SignalType

    # 创建简单的测试策略
    class SimpleTestStrategy(BaseStrategy):
        def __init__(self):
            super().__init__("SimpleTest")
            self.buy_triggered = False

        def generate_signals(self, df):
            signals = []
            # 第一天买入
            if len(df) > 0 and not self.buy_triggered:
                signals.append(Signal(
                    date=str(df.iloc[0]["trade_date"]),
                    signal_type=SignalType.BUY,
                    price=float(df.iloc[0]["close"]),
                    reason="测试买入"
                ))
                self.buy_triggered = True

            # 最后一天卖出
            if len(df) > 1 and self.buy_triggered:
                signals.append(Signal(
                    date=str(df.iloc[-1]["trade_date"]),
                    signal_type=SignalType.SELL,
                    price=float(df.iloc[-1]["close"]),
                    reason="测试卖出"
                ))
            return signals

    # 创建测试数据（价格上涨10%）
    dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
    close_prices = [100.0 + i * 0.5 for i in range(20)]  # 从100涨到109.5
    df = pd.DataFrame({
        "trade_date": dates,
        "open": close_prices,
        "high": [p + 0.5 for p in close_prices],
        "low": [p - 0.5 for p in close_prices],
        "close": close_prices,
        "volume": [1000000] * 20,
    })

    strategy = SimpleTestStrategy()

    # 1. 无成本回测
    print("\n1. 无成本回测:")
    backtester_no_cost = SimpleBacktester(
        initial_capital=100000,
        cost_config=CostConfig.no_cost(),
        slippage_model=NoSlippage(),
    )
    result_no_cost = backtester_no_cost.run(strategy, df)
    print(f"   初始资金: {result_no_cost.initial_capital:,.2f}")
    print(f"   最终资金: {result_no_cost.final_capital:,.2f}")
    print(f"   总收益率: {result_no_cost.total_return*100:.2f}%")
    print(f"   总成本: {result_no_cost.total_costs:.2f}")

    # 重置策略
    strategy = SimpleTestStrategy()

    # 2. 带成本回测
    print("\n2. 带成本回测:")
    backtester_with_cost = SimpleBacktester(
        initial_capital=100000,
        cost_config=CostConfig.default(),
        slippage_model=FixedSlippage(Decimal("0.001")),
    )
    result_with_cost = backtester_with_cost.run(strategy, df)
    print(f"   初始资金: {result_with_cost.initial_capital:,.2f}")
    print(f"   最终资金: {result_with_cost.final_capital:,.2f}")
    print(f"   总收益率: {result_with_cost.total_return*100:.2f}%")
    print(f"   总成本: {result_with_cost.total_costs:.2f}")
    print(f"   总滑点: {result_with_cost.total_slippage:.2f}")

    # 3. 对比分析
    print("\n3. 成本影响分析:")
    return_diff = result_no_cost.total_return - result_with_cost.total_return
    print(f"   收益率差异: {return_diff*100:.2f}%")
    print(f"   成本占初始资金比: {result_with_cost.total_costs / result_with_cost.initial_capital * 100:.2f}%")

    # 验证带成本收益率低于无成本
    assert result_with_cost.total_return < result_no_cost.total_return, "带成本收益率应低于无成本"
    assert result_with_cost.total_costs > 0, "应有交易成本"

    print("\n✅ 回测集成测试通过")


def main():
    """运行所有测试"""
    print("开始测试交易成本和滑点模块")
    print("=" * 60)

    test_cost_calculator()
    test_market_identification()
    test_transfer_fee_by_market()
    test_convenience_functions()
    test_slippage_models()
    test_backtest_with_costs()

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
