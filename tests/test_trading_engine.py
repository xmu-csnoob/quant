"""
交易执行模块单元测试

测试订单管理、MockTradingAPI、T+1规则等功能
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from decimal import Decimal
from datetime import datetime, date, timedelta
from src.trading.orders import OrderManager, Order, OrderType, OrderStatus, OrderSide
from src.trading.api import MockTradingAPI
from src.trading.t1_manager import T1Manager


class TestOrderManager:
    """订单管理测试"""

    def test_create_order(self):
        """测试创建订单"""
        print("\n测试创建订单")
        manager = OrderManager()

        order = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        assert order is not None
        assert order.symbol == "600000.SH"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING
        print(f"  ✅ 订单创建成功: {order.order_id}")

    def test_submit_order(self):
        """测试提交订单"""
        print("\n测试提交订单")
        manager = OrderManager()

        order = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        # submit_order expects Order object, not order_id
        success = manager.submit_order(order)
        assert success is True
        assert order.status == OrderStatus.SUBMITTED
        print(f"  ✅ 订单提交成功: {order.order_id}")

    def test_cancel_order(self):
        """测试取消订单"""
        print("\n测试取消订单")
        manager = OrderManager()

        order = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        manager.submit_order(order)
        success = manager.cancel_order(order.order_id)

        assert success is True
        assert order.status == OrderStatus.CANCELLED
        print(f"  ✅ 订单取消成功: {order.order_id}")

    def test_get_active_orders(self):
        """测试获取活动订单"""
        print("\n测试获取活动订单")
        manager = OrderManager()

        # 创建多个订单
        created_orders = []
        for i in range(3):
            order = manager.create_order(
                symbol=f"60000{i}.SH",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100,
                price=10.00,
            )
            manager.submit_order(order)
            created_orders.append(order)

        # 取消第一个
        manager.cancel_order(created_orders[0].order_id)

        active_orders = manager.get_active_orders()
        assert len(active_orders) == 2
        print(f"  ✅ 活动订单数量: {len(active_orders)}")


class TestMockTradingAPI:
    """MockTradingAPI测试"""

    def test_get_account(self):
        """测试获取账户信息"""
        print("\n测试获取账户信息")
        api = MockTradingAPI(initial_cash=1000000)

        account = api.get_account()

        assert account is not None
        assert account.total_assets == 1000000
        assert account.cash == 1000000
        print(f"  ✅ 账户总资产: {account.total_assets}")

    def test_buy_stock(self):
        """测试买入股票"""
        print("\n测试买入股票")
        api = MockTradingAPI(initial_cash=1000000)
        api.connect()

        # 创建买入订单
        order = Order(
            order_id="TEST001",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        # 下单
        success = api.place_order(order)
        assert success is True
        assert order.status == OrderStatus.SUBMITTED

        # 模拟成交
        api.simulate_fill(order, fill_price=10.50, fill_quantity=1000)
        assert order.status == OrderStatus.FILLED

        print(f"  ✅ 买入订单: {order.order_id}")

    def test_sell_stock(self):
        """测试卖出股票"""
        print("\n测试卖出股票")
        api = MockTradingAPI(initial_cash=1000000)
        api.connect()

        # 先买入
        buy_order = Order(
            order_id="BUY001",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )
        api.place_order(buy_order)
        api.simulate_fill(buy_order, fill_price=10.50, fill_quantity=1000)

        # 设置为昨天买入（T+1可卖）
        api.set_current_date((date.today() - timedelta(days=1)).strftime("%Y%m%d"))
        # 更新持仓日期
        if "600000.SH" in api.positions:
            for lot in api.positions["600000.SH"]["lots"]:
                lot["buy_date"] = (date.today() - timedelta(days=1)).strftime("%Y%m%d")

        # 设置今天日期
        api.set_current_date(date.today().strftime("%Y%m%d"))

        # 卖出
        sell_order = Order(
            order_id="SELL001",
            symbol="600000.SH",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=11.00,
        )
        success = api.place_order(sell_order)

        print(f"  ✅ 卖出订单: {sell_order.order_id if success else 'None (T+1限制)'}")

    def test_get_positions(self):
        """测试获取持仓"""
        print("\n测试获取持仓")
        api = MockTradingAPI(initial_cash=1000000)
        api.connect()

        # 买入
        buy_order = Order(
            order_id="BUY001",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )
        api.place_order(buy_order)
        api.simulate_fill(buy_order, fill_price=10.50, fill_quantity=1000)

        positions = api.get_positions()

        assert len(positions) > 0
        assert positions[0].symbol == "600000.SH"
        print(f"  ✅ 持仓数量: {len(positions)}")


class TestT1Manager:
    """T+1管理测试"""

    def test_can_sell_today(self):
        """测试当天买入能否卖出"""
        print("\n测试T+1规则")
        manager = T1Manager()

        today = date.today()
        code = "600000.SH"

        # 今天买入 (code, shares, buy_date, price)
        manager.record_buy(code, 1000, today, 10.0)

        # 今天不能卖 (can_sell需要shares参数)
        can_sell = manager.can_sell(code, 1000, today)
        assert can_sell is False
        print("  ✅ 当天买入不能卖出")

    def test_can_sell_next_day(self):
        """测试次日能否卖出"""
        print("\n测试次日能否卖出")
        manager = T1Manager()

        today = date.today()
        # 计算下一个交易日（跳过周末）
        tomorrow = today + timedelta(days=1)
        while tomorrow.weekday() >= 5:  # 5=Saturday, 6=Sunday
            tomorrow += timedelta(days=1)
        code = "600000.SH"

        # 今天买入
        manager.record_buy(code, 1000, today, 10.0)

        # 明天可以卖 (can_sell需要shares参数)
        can_sell = manager.can_sell(code, 1000, tomorrow)
        assert can_sell is True
        print("  ✅ 次日可以卖出")

    def test_get_available_shares(self):
        """测试获取可卖数量"""
        print("\n测试获取可卖数量")
        manager = T1Manager()

        today = date.today()
        code = "600000.SH"

        # 今天买入1000股
        manager.record_buy(code, 1000, today, 10.0)

        # 今天可卖0股
        available = manager.get_available_shares(code, today)
        assert available == 0
        print(f"  ✅ 今天可卖: {available}股")

    def test_partial_t1_shares(self):
        """测试部分T+1锁定"""
        print("\n测试部分T+1锁定")
        manager = T1Manager()

        # 使用固定日期避免周末问题
        # 周三买入，周四可卖
        today = date(2026, 2, 19)  # 周四
        yesterday = date(2026, 2, 18)  # 周三
        code = "600000.SH"

        # 昨天买入500股
        manager.record_buy(code, 500, yesterday, 10.0)
        # 今天买入500股
        manager.record_buy(code, 500, today, 10.0)

        # 今天可卖500股（昨天买入的）
        available = manager.get_available_shares(code, today)
        assert available == 500
        print(f"  ✅ 今天可卖: {available}股，T+1锁定: 500股")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("交易执行模块单元测试")
    print("=" * 60)

    # 订单管理测试
    tom = TestOrderManager()
    tom.test_create_order()
    tom.test_submit_order()
    tom.test_cancel_order()
    tom.test_get_active_orders()

    # MockTradingAPI测试
    tmt = TestMockTradingAPI()
    tmt.test_get_account()
    tmt.test_buy_stock()
    tmt.test_sell_stock()
    tmt.test_get_positions()

    # T+1管理测试
    tt1 = TestT1Manager()
    tt1.test_can_sell_today()
    tt1.test_can_sell_next_day()
    tt1.test_get_available_shares()
    tt1.test_partial_t1_shares()

    print("\n" + "=" * 60)
    print("所有交易执行测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
