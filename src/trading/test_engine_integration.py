"""
交易引擎集成测试

测试涨跌停和滑点集成到实时交易引擎
"""

import sys
from pathlib import Path
from datetime import time
from decimal import Decimal

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.engine import LiveTradingEngine, create_paper_trading_engine
from src.trading.api import MockTradingAPI
from src.trading.price_limit import get_price_limit_checker
from src.strategies.base import Signal, SignalType
from src.backtesting.slippage import TimeAwareSlippage, FixedSlippage
from src.risk import RiskManager, PositionSizer


def test_price_limit_in_engine():
    """测试交易引擎中的涨跌停检查"""
    print("=" * 60)
    print("测试交易引擎涨跌停集成")
    print("=" * 60)

    # 创建模拟API
    api = MockTradingAPI(initial_cash=1000000)
    api.connect()

    # 创建简单的仓位管理器和风控管理器
    position_sizer = PositionSizer(
        initial_capital=1000000,
        method="fixed_ratio",
        max_position_ratio=0.3,  # 单只股票最大30%仓位
    )
    risk_manager = RiskManager(
        initial_capital=1000000,
        position_sizer=position_sizer,
        stop_loss=0.05,
        take_profit=0.15,
    )

    # 创建一个简单的策略
    from src.strategies.base import BaseStrategy

    class DummyStrategy(BaseStrategy):
        def __init__(self):
            super().__init__("TestStrategy")

        def generate_signals(self, df):
            return []

    strategy = DummyStrategy()

    # 创建交易引擎
    engine = LiveTradingEngine(
        strategy=strategy,
        trading_api=api,
        risk_manager=risk_manager,
        symbols=["600519.SH", "300750.SZ"],
    )

    # 测试1：正常买入
    print("\n1. 测试正常买入（未涨停）")
    signal = Signal(
        date="600519.SH",
        signal_type=SignalType.BUY,
        price=1800.0,
        reason="测试买入",
        confidence=0.8
    )
    order = engine.process_signal(signal, current_price=1800.0, prev_close=1780.0)
    status = "✅ 成功" if order else "❌ 失败"
    print(f"  结果: {status}")

    # 测试2：涨停买入被拒绝
    print("\n2. 测试涨停买入（应该被拒绝）")
    # 假设600519涨停（+10%），前收盘1800，涨停价1980
    signal2 = Signal(
        date="600519.SH",
        signal_type=SignalType.BUY,
        price=1980.0,  # 涨停价
        reason="涨停买入测试",
        confidence=0.8
    )
    order2 = engine.process_signal(signal2, current_price=1980.0, prev_close=1800.0)
    status = "❌ 被拒绝" if order2 is None else "⚠️ 未被拒绝"
    print(f"  结果: {status}")

    # 测试3：跌停卖出被拒绝
    print("\n3. 测试跌停卖出（应该被拒绝）")
    # 先买入一些股票
    api.positions["300750.SZ"] = {
        "quantity": 100,
        "avg_price": 200.0,
        "current_price": 180.0
    }

    signal3 = Signal(
        date="300750.SZ",
        signal_type=SignalType.SELL,
        price=160.0,  # 跌停价（-20%创业板）
        reason="跌停卖出测试",
        confidence=0.8
    )
    order3 = engine.process_signal(signal3, current_price=160.0, prev_close=200.0)
    status = "❌ 被拒绝" if order3 is None else "⚠️ 未被拒绝"
    print(f"  结果: {status}")

    # 测试4：正常卖出
    print("\n4. 测试正常卖出（未跌停）")
    signal4 = Signal(
        date="300750.SZ",
        signal_type=SignalType.SELL,
        price=190.0,
        reason="正常卖出测试",
        confidence=0.8
    )
    order4 = engine.process_signal(signal4, current_price=190.0, prev_close=200.0)
    status = "✅ 成功" if order4 else "❌ 失败"
    print(f"  结果: {status}")

    print()


def test_slippage_in_mock_api():
    """测试MockTradingAPI中的滑点集成"""
    print("=" * 60)
    print("测试MockTradingAPI滑点集成")
    print("=" * 60)

    # 创建带滑点模型的API
    slippage = TimeAwareSlippage(base_rate=Decimal("0.001"))  # 0.1%基础滑点
    api = MockTradingAPI(initial_cash=100000, slippage_model=slippage)
    api.connect()

    from src.trading.orders import Order, OrderSide, OrderType

    # 创建订单
    order = Order(
        order_id="TEST001",
        symbol="600519.SH",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=1800.0,
    )
    api.place_order(order)

    # 测试1：上午正常时段成交
    print("\n1. 上午正常时段 (10:30)")
    api.simulate_fill(order, fill_price=1800.0, fill_quantity=100, current_time=time(10, 30))
    print(f"  成交价: 1800.0 (基础滑点)")

    # 买入另一只股票测试开盘时段
    order2 = Order(
        order_id="TEST002",
        symbol="300750.SZ",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=200.0,
    )
    api.place_order(order2)

    # 测试2：开盘时段成交（滑点应该更大）
    print("\n2. 开盘时段 (9:45)")
    api.simulate_fill(order2, fill_price=200.0, fill_quantity=100, current_time=time(9, 45))
    print(f"  成交价: 应高于200.0 (开盘滑点+50%)")

    # 测试3：尾盘时段
    order3 = Order(
        order_id="TEST003",
        symbol="000001.SZ",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=10.0,
    )
    api.place_order(order3)

    print("\n3. 尾盘时段 (14:45)")
    api.simulate_fill(order3, fill_price=10.0, fill_quantity=100, current_time=time(14, 45))
    print(f"  成交价: 应高于10.0 (尾盘滑点+30%)")

    # 显示账户状态
    account = api.get_account()
    print(f"\n账户状态:")
    print(f"  现金: {account.cash:.2f}")
    print(f"  持仓数: {len(api.positions)}")

    print()


def test_price_limit_checker():
    """直接测试涨跌停检查器"""
    print("=" * 60)
    print("测试涨跌停检查器")
    print("=" * 60)

    checker = get_price_limit_checker()

    # 测试主板
    print("\n1. 主板股票 (600519.SH)")
    can_buy, reason = checker.can_buy("600519.SH", 1980.0, 1800.0)
    print(f"  涨停价1980买入: {can_buy} - {reason}")

    can_buy, reason = checker.can_buy("600519.SH", 1850.0, 1800.0)
    print(f"  正常价1850买入: {can_buy} - {reason}")

    can_sell, reason = checker.can_sell("600519.SH", 1620.0, 1800.0)
    print(f"  跌停价1620卖出: {can_sell} - {reason}")

    # 测试创业板
    print("\n2. 创业板股票 (300750.SZ)")
    can_buy, reason = checker.can_buy("300750.SZ", 240.0, 200.0)
    print(f"  涨停价240买入: {can_buy} - {reason}")

    # 测试科创板
    print("\n3. 科创板股票 (688981.SH)")
    can_buy, reason = checker.can_buy("688981.SH", 120.0, 100.0)
    print(f"  涨停价120买入: {can_buy} - {reason}")

    # 测试ST股票
    print("\n4. ST股票 (模拟)")
    can_buy, reason = checker.can_buy("600688.SH", 10.5, 10.0, name="ST海航")
    print(f"  ST涨停价10.5买入: {can_buy} - {reason}")

    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始交易引擎集成测试")
    print("=" * 60)
    print()

    test_price_limit_checker()
    test_price_limit_in_engine()
    test_slippage_in_mock_api()

    print("=" * 60)
    print("所有测试完成 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
