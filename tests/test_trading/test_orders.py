"""
订单管理模块单元测试

测试OrderManager和Order的核心功能
"""

import pytest
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.orders import OrderManager, Order, OrderType, OrderStatus, OrderSide


class TestOrder:
    """测试Order数据类"""

    def test_order_creation(self):
        """测试创建订单"""
        order = Order(
            order_id="TEST001",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        assert order.order_id == "TEST001"
        assert order.symbol == "600000.SH"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 1000
        assert order.price == 10.50
        assert order.status == OrderStatus.PENDING

    def test_order_market_type(self):
        """测试市价单"""
        order = Order(
            order_id="TEST002",
            symbol="600000.SH",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=500,
        )

        assert order.order_type == OrderType.MARKET
        assert order.price is None

    def test_order_remaining_quantity(self):
        """测试剩余数量"""
        order = Order(
            order_id="TEST003",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.0,
        )

        assert order.remaining_quantity == 1000

        order.filled_quantity = 300
        assert order.remaining_quantity == 700

    def test_order_is_fully_filled(self):
        """测试完全成交"""
        order = Order(
            order_id="TEST004",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.0,
        )

        assert order.is_fully_filled is False

        order.filled_quantity = 1000
        assert order.is_fully_filled is True

    def test_order_is_active(self):
        """测试活跃状态"""
        order = Order(
            order_id="TEST005",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.0,
        )

        assert order.is_active is True

        order.status = OrderStatus.FILLED
        assert order.is_active is False

        order.status = OrderStatus.CANCELLED
        assert order.is_active is False

    def test_order_str(self):
        """测试字符串表示"""
        order = Order(
            order_id="TEST006",
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        s = str(order)
        assert "TEST006" in s
        assert "600000.SH" in s


class TestOrderManager:
    """测试OrderManager"""

    def test_create_order(self):
        """测试创建订单"""
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
        assert order.order_id in manager.orders

    def test_create_multiple_orders(self):
        """测试创建多个订单"""
        manager = OrderManager()

        order1 = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        order2 = manager.create_order(
            symbol="600001.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=500,
            price=20.00,
        )

        assert len(manager.orders) == 2
        assert order1.order_id != order2.order_id

    def test_submit_order(self):
        """测试提交订单"""
        manager = OrderManager()

        order = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        success = manager.submit_order(order)
        assert success is True
        assert order.status == OrderStatus.SUBMITTED

    def test_submit_non_pending_order(self):
        """测试提交非PENDING订单"""
        manager = OrderManager()

        order = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        manager.submit_order(order)

        # 再次提交应该失败
        success = manager.submit_order(order)
        assert success is False

    def test_cancel_order(self):
        """测试取消订单"""
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

    def test_cancel_nonexistent_order(self):
        """测试取消不存在的订单"""
        manager = OrderManager()

        success = manager.cancel_order("NONEXISTENT")
        assert success is False

    def test_cancel_filled_order(self):
        """测试取消已成交订单"""
        manager = OrderManager()

        order = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        manager.submit_order(order)
        order.status = OrderStatus.FILLED

        success = manager.cancel_order(order.order_id)
        assert success is False

    def test_update_order(self):
        """测试更新订单"""
        manager = OrderManager()

        order = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        success = manager.update_order(
            order.order_id,
            filled_quantity=500,
            avg_price=10.48,
            status=OrderStatus.PARTIAL_FILLED,
        )

        assert success is True
        assert order.filled_quantity == 500
        assert order.avg_price == 10.48
        assert order.status == OrderStatus.PARTIAL_FILLED

    def test_update_nonexistent_order(self):
        """测试更新不存在的订单"""
        manager = OrderManager()

        success = manager.update_order("NONEXISTENT", filled_quantity=100)
        assert success is False

    def test_get_order(self):
        """测试获取订单"""
        manager = OrderManager()

        order = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.50,
        )

        retrieved = manager.get_order(order.order_id)
        assert retrieved == order

    def test_get_nonexistent_order(self):
        """测试获取不存在的订单"""
        manager = OrderManager()

        retrieved = manager.get_order("NONEXISTENT")
        assert retrieved is None

    def test_get_active_orders(self):
        """测试获取活跃订单"""
        manager = OrderManager()

        # 创建并提交多个订单
        orders = []
        for i in range(3):
            order = manager.create_order(
                symbol=f"60000{i}.SH",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100,
                price=10.00,
            )
            manager.submit_order(order)
            orders.append(order)

        # 取消一个
        manager.cancel_order(orders[0].order_id)

        active = manager.get_active_orders()
        assert len(active) == 2

    def test_get_active_orders_by_symbol(self):
        """测试按股票获取活跃订单"""
        manager = OrderManager()

        order1 = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=10.00,
        )
        manager.submit_order(order1)

        order2 = manager.create_order(
            symbol="600001.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=20.00,
        )
        manager.submit_order(order2)

        active_600000 = manager.get_active_orders(symbol="600000.SH")
        assert len(active_600000) == 1
        assert active_600000[0].symbol == "600000.SH"

    def test_get_order_summary(self):
        """测试获取订单摘要"""
        manager = OrderManager()

        order1 = manager.create_order(
            symbol="600000.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=10.00,
        )
        manager.submit_order(order1)

        order2 = manager.create_order(
            symbol="600001.SH",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=20.00,
        )
        manager.submit_order(order2)
        order2.status = OrderStatus.FILLED

        summary = manager.get_order_summary()

        assert summary["total_orders"] == 2
        assert summary["active_orders"] == 1
        assert summary["filled_orders"] == 1


class TestOrderTypes:
    """测试订单类型"""

    def test_order_types(self):
        """测试所有订单类型"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_order_sides(self):
        """测试订单方向"""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_statuses(self):
        """测试订单状态"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.PARTIAL_FILLED.value == "partial_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"
