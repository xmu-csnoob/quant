"""
Trading service - 交易服务层

包含T+1规则：
- 当日买入的股票，下一交易日才能卖出
- 卖出时检查可卖数量
"""

import uuid
from datetime import datetime, date
from typing import List, Optional
from loguru import logger

from src.api.schemas.trading import (
    Order, CreateOrderRequest, OrderStatus, OrderDirection, OrderType
)
from src.trading.t1_manager import get_t1_manager


class TradingService:
    """交易服务（含T+1规则检查）"""

    def __init__(self):
        self._t1_manager = get_t1_manager()  # T+1管理器 - 先初始化
        self._orders: dict[str, Order] = {}
        # 模拟一些历史订单
        self._init_mock_orders()

    def _init_mock_orders(self):
        """初始化模拟订单数据（历史已成交订单，满足T+1规则）"""
        # 模拟历史买入（都在T+1之前，已经可卖）
        self._t1_manager.record_buy("600519.SH", 100, date(2024, 1, 10), 1680.00)
        self._t1_manager.record_buy("000858.SZ", 300, date(2024, 1, 9), 145.80)
        self._t1_manager.record_buy("601318.SH", 800, date(2024, 1, 8), 42.30)

        mock_orders = [
            Order(
                order_id=str(uuid.uuid4()),
                code="600519.SH",
                name="贵州茅台",
                direction=OrderDirection.BUY,
                order_type=OrderType.LIMIT,
                price=1680.00,
                shares=100,
                filled_shares=100,
                status=OrderStatus.FILLED,
                created_at=datetime(2024, 1, 10, 9, 32, 15),
                updated_at=datetime(2024, 1, 10, 9, 32, 45)
            ),
            Order(
                order_id=str(uuid.uuid4()),
                code="000858.SZ",
                name="五粮液",
                direction=OrderDirection.BUY,
                order_type=OrderType.MARKET,
                price=145.80,
                shares=300,
                filled_shares=300,
                status=OrderStatus.FILLED,
                created_at=datetime(2024, 1, 9, 10, 15, 30),
                updated_at=datetime(2024, 1, 9, 10, 15, 35)
            ),
            Order(
                order_id=str(uuid.uuid4()),
                code="601318.SH",
                name="中国平安",
                direction=OrderDirection.BUY,
                order_type=OrderType.LIMIT,
                price=42.30,
                shares=800,
                filled_shares=800,
                status=OrderStatus.FILLED,
                created_at=datetime(2024, 1, 8, 13, 45, 20),
                updated_at=datetime(2024, 1, 8, 14, 20, 10)
            ),
        ]
        for order in mock_orders:
            self._orders[order.order_id] = order

    def get_orders(
        self,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[Order], int]:
        """获取订单列表"""
        orders = list(self._orders.values())

        if status:
            orders = [o for o in orders if o.status == status]

        # 按时间倒序
        orders.sort(key=lambda x: x.created_at, reverse=True)

        total = len(orders)
        start = (page - 1) * page_size
        end = start + page_size

        return orders[start:end], total

    def create_order(self, request: CreateOrderRequest) -> Order:
        """
        创建订单

        Args:
            request: 订单请求

        Returns:
            创建的订单

        Raises:
            ValueError: T+1检查失败时抛出
        """
        order_id = str(uuid.uuid4())
        now = datetime.now()
        today = now.date()

        # 获取股票名称 (实际应从数据库查询)
        name_map = {
            "600000.SH": "浦发银行",
            "600519.SH": "贵州茅台",
            "000858.SZ": "五粮液",
            "601318.SH": "中国平安",
        }
        name = name_map.get(request.code, "未知股票")

        # T+1检查：卖出时检查可卖数量
        if request.direction == OrderDirection.SELL:
            available = self._t1_manager.get_available_shares(request.code, today)
            if request.shares > available:
                total = self._t1_manager.get_total_shares(request.code)
                locked = self._t1_manager.get_locked_shares(request.code, today)
                raise ValueError(
                    f"T+1规则限制：{request.code} 可卖数量不足。"
                    f"总持仓={total}股，可卖={available}股，锁定={locked}股"
                )

        order = Order(
            order_id=order_id,
            code=request.code,
            name=name,
            direction=request.direction,
            order_type=request.order_type,
            price=request.price,
            shares=request.shares,
            filled_shares=0,
            status=OrderStatus.SUBMITTED,
            created_at=now,
            updated_at=now
        )

        self._orders[order_id] = order
        logger.info(f"创建订单: {order_id[:8]}... {request.direction.value} {request.code} {request.shares}股")
        return order

    def fill_order(self, order_id: str, filled_price: Optional[float] = None) -> Optional[Order]:
        """
        订单成交（模拟）

        Args:
            order_id: 订单ID
            filled_price: 成交价格（可选，默认使用委托价）

        Returns:
            成交后的订单
        """
        if order_id not in self._orders:
            return None

        order = self._orders[order_id]
        if order.status != OrderStatus.SUBMITTED:
            return None

        order.status = OrderStatus.FILLED
        order.filled_shares = order.shares
        order.updated_at = datetime.now()

        # 记录T+1状态
        if order.direction == OrderDirection.BUY:
            self._t1_manager.record_buy(
                code=order.code,
                shares=order.shares,
                buy_date=order.created_at.date(),
                price=filled_price or order.price or 0.0
            )
            logger.info(f"买入成交: {order.code} {order.shares}股, 已记录T+1状态")
        elif order.direction == OrderDirection.SELL:
            self._t1_manager.deduct_available(
                code=order.code,
                shares=order.shares,
                current_date=order.created_at.date()
            )
            logger.info(f"卖出成交: {order.code} {order.shares}股, 已扣减可卖数量")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        return True


# 单例
trading_service = TradingService()
