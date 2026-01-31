"""
订单管理系统

管理交易订单的生命周期
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from loguru import logger


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"       # 市价单
    LIMIT = "limit"         # 限价单
    STOP = "stop"           # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"         # 待发送
    SUBMITTED = "submitted"     # 已提交
    PARTIAL_FILLED = "partial_filled"  # 部分成交
    FILLED = "filled"           # 完全成交
    CANCELLED = "cancelled"     # 已撤销
    REJECTED = "rejected"       # 被拒绝
    EXPIRED = "expired"         # 已过期


@dataclass
class Order:
    """
    订单

    Attributes:
        order_id: 订单ID（唯一标识）
        symbol: 交易标的（股票代码）
        side: 买卖方向
        order_type: 订单类型
        quantity: 数量（股数）
        price: 价格（限价单必填）
        status: 订单状态
        filled_quantity: 已成交数量
        avg_price: 平均成交价格
        create_time: 创建时间
        update_time: 更新时间
        reason: 订单原因（策略信号）
    """

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_price: float = 0
    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    reason: Optional[str] = None
    stop_price: Optional[float] = None  # 止损单的触发价格

    @property
    def remaining_quantity(self) -> int:
        """剩余数量"""
        return self.quantity - self.filled_quantity

    @property
    def is_fully_filled(self) -> bool:
        """是否完全成交"""
        return self.filled_quantity >= self.quantity

    @property
    def is_active(self) -> bool:
        """是否活跃（可成交）"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILLED,
        ]

    def __str__(self) -> str:
        return (
            f"Order({self.order_id}, {self.symbol}, "
            f"{self.side.value} {self.quantity} @ {self.price or 'MARKET'}, "
            f"status={self.status.value}, filled={self.filled_quantity})"
        )


class OrderManager:
    """
    订单管理器

    功能：
    1. 订单创建
    2. 订单状态跟踪
    3. 订单撤销
    4. 订单查询
    """

    def __init__(self):
        self.orders: dict[str, Order] = {}
        self._order_counter = 0

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> Order:
        """
        创建新订单

        Args:
            symbol: 股票代码
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单必填）
            stop_price: 止损触发价
            reason: 订单原因

        Returns:
            创建的订单
        """
        self._order_counter += 1
        order_id = f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}{self._order_counter:04d}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            reason=reason,
        )

        self.orders[order_id] = order
        logger.info(f"创建订单: {order}")
        return order

    def submit_order(self, order: Order) -> bool:
        """
        提交订单（模拟提交）

        Returns:
            是否成功提交
        """
        if order.status != OrderStatus.PENDING:
            logger.warning(f"订单 {order.order_id} 状态不是PENDING，无法提交")
            return False

        order.status = OrderStatus.SUBMITTED
        order.update_time = datetime.now()
        logger.info(f"提交订单: {order}")
        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        撤销订单

        Args:
            order_id: 订单ID

        Returns:
            是否成功撤销
        """
        if order_id not in self.orders:
            logger.warning(f"订单 {order_id} 不存在")
            return False

        order = self.orders[order_id]

        if not order.is_active:
            logger.warning(f"订单 {order_id} 状态为 {order.status.value}，无法撤销")
            return False

        order.status = OrderStatus.CANCELLED
        order.update_time = datetime.now()
        logger.info(f"撤销订单: {order}")
        return True

    def update_order(
        self,
        order_id: str,
        filled_quantity: Optional[int] = None,
        status: Optional[OrderStatus] = None,
        avg_price: Optional[float] = None,
    ) -> bool:
        """
        更新订单状态

        Returns:
            是否成功更新
        """
        if order_id not in self.orders:
            logger.warning(f"订单 {order_id} 不存在")
            return False

        order = self.orders[order_id]

        if filled_quantity is not None:
            order.filled_quantity = filled_quantity

        if status is not None:
            order.status = status

        if avg_price is not None:
            order.avg_price = avg_price

        order.update_time = datetime.now()
        logger.debug(f"更新订单: {order}")

        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """
        获取活跃订单

        Args:
            symbol: 过滤股票代码（None=全部）

        Returns:
            活跃订单列表
        """
        orders = [
            o for o in self.orders.values()
            if o.is_active and (symbol is None or o.symbol == symbol)
        ]
        return sorted(orders, key=lambda o: o.create_time)

    def get_order_summary(self) -> dict:
        """
        获取订单摘要

        Returns:
            订单统计信息
        """
        total = len(self.orders)
        active = len([o for o in self.orders.values() if o.is_active])
        filled = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        cancelled = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])

        return {
            "total_orders": total,
            "active_orders": active,
            "filled_orders": filled,
            "cancelled_orders": cancelled,
        }
