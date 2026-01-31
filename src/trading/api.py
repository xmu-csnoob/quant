"""
交易API接口

定义统一的交易接口，支持实盘和模拟盘
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from src.trading.orders import Order, OrderType, OrderSide, OrderStatus


@dataclass
class Account:
    """账户信息"""
    account_id: str
    total_assets: float      # 总资产
    cash: float              # 可用资金
    market_value: float      # 市值
    frozen_cash: float = 0   # 冻结资金


@dataclass
class Position:
    """持仓信息"""
    symbol: str              # 股票代码
    quantity: int            # 持仓数量
    available_quantity: int  # 可用数量
    avg_price: float         # 成本价
    current_price: float     # 当前价
    market_value: float      # 市值
    pnl: float               # 盈亏
    pnl_ratio: float         # 盈亏比例


@dataclass
class Trade:
    """成交记录"""
    trade_id: str            # 成交ID
    order_id: str            # 订单ID
    symbol: str              # 股票代码
    side: OrderSide          # 买卖方向
    quantity: int            # 成交数量
    price: float             # 成交价格
    trade_time: str          # 成交时间
    commission: float = 0    # 手续费


class TradingAPI(ABC):
    """
    交易API抽象类

    定义统一的交易接口，支持：
    1. 订单下单、撤单、查询
    2. 持仓查询
    3. 账户查询
    4. 成交查询
    """

    @abstractmethod
    def connect(self) -> bool:
        """连接交易接口"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查是否已连接"""
        pass

    @abstractmethod
    def place_order(self, order: Order) -> bool:
        """
        下单

        Returns:
            是否成功下单
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        撤单

        Returns:
            是否成功撤单
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        查询订单

        Returns:
            订单信息，不存在返回None
        """
        pass

    @abstractmethod
    def get_orders(self) -> List[Order]:
        """
        查询所有订单

        Returns:
            订单列表
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        查询持仓

        Returns:
            持仓列表
        """
        pass

    @abstractmethod
    def get_account(self) -> Account:
        """
        查询账户

        Returns:
            账户信息
        """
        pass

    @abstractmethod
    def get_trades(self, order_id: Optional[str] = None) -> List[Trade]:
        """
        查询成交

        Args:
            order_id: 订单ID（None=查询全部）

        Returns:
            成交列表
        """
        pass


class MockTradingAPI(TradingAPI):
    """
    模拟交易API

    用于回测和模拟盘测试
    """

    def __init__(self, initial_cash: float = 100000):
        """
        初始化

        Args:
            initial_cash: 初始资金
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, dict] = {}
        self.orders: dict[str, Order] = {}
        self.trades: list[Trade] = []
        self._connected = False

    def connect(self) -> bool:
        """连接"""
        self._connected = True
        logger.info("模拟交易API已连接")
        return True

    def disconnect(self) -> bool:
        """断开连接"""
        self._connected = False
        logger.info("模拟交易API已断开")
        return True

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected

    def place_order(self, order: Order) -> bool:
        """下单（模拟）"""
        if not self._connected:
            logger.error("未连接到交易API")
            return False

        # 检查资金（买入）
        if order.side == OrderSide.BUY:
            required = order.quantity * (order.price or 0)  # 市价单假设价格合理
            if self.cash < required:
                logger.warning(f"资金不足: 需要{required:.2f}, 可用{self.cash:.2f}")
                order.status = OrderStatus.REJECTED
                return False

        # 检查持仓（卖出）
        if order.side == OrderSide.SELL:
            if order.symbol not in self.positions:
                logger.warning(f"无持仓: {order.symbol}")
                order.status = OrderStatus.REJECTED
                return False

            available = self.positions[order.symbol]["quantity"]
            if available < order.quantity:
                logger.warning(f"持仓不足: 需要{order.quantity}, 可用{available}")
                order.status = OrderStatus.REJECTED
                return False

        # 模拟订单成功
        order.status = OrderStatus.SUBMITTED
        self.orders[order.order_id] = order
        logger.info(f"模拟下单成功: {order}")
        return True

    def cancel_order(self, order_id: str) -> bool:
        """撤单（模拟）"""
        if order_id not in self.orders:
            logger.warning(f"订单不存在: {order_id}")
            return False

        order = self.orders[order_id]

        if not order.is_active:
            logger.warning(f"订单状态不允许撤销: {order.status}")
            return False

        order.status = OrderStatus.CANCELLED
        logger.info(f"模拟撤单成功: {order_id}")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单"""
        return self.orders.get(order_id)

    def get_orders(self) -> List[Order]:
        """查询所有订单"""
        return list(self.orders.values())

    def get_positions(self) -> List[Position]:
        """查询持仓"""
        result = []
        for symbol, pos in self.positions.items():
            quantity = pos["quantity"]
            avg_price = pos["avg_price"]
            current_price = pos.get("current_price", avg_price)

            result.append(Position(
                symbol=symbol,
                quantity=quantity,
                available_quantity=quantity,  # 简化处理
                avg_price=avg_price,
                current_price=current_price,
                market_value=quantity * current_price,
                pnl=(current_price - avg_price) * quantity,
                pnl_ratio=(current_price / avg_price - 1) if avg_price > 0 else 0,
            ))
        return result

    def get_account(self) -> Account:
        """查询账户"""
        # 计算持仓市值
        market_value = sum(
            pos["quantity"] * pos.get("current_price", pos["avg_price"])
            for pos in self.positions.values()
        )

        return Account(
            account_id="mock_account",
            total_assets=self.cash + market_value,
            cash=self.cash,
            market_value=market_value,
        )

    def get_trades(self, order_id: Optional[str] = None) -> List[Trade]:
        """查询成交"""
        if order_id:
            return [t for t in self.trades if t.order_id == order_id]
        return self.trades

    def simulate_fill(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: int,
    ):
        """
        模拟成交

        Args:
            order: 订单
            fill_price: 成交价格
            fill_quantity: 成交数量
        """
        if fill_quantity <= 0:
            return

        # 更新订单
        order.filled_quantity += fill_quantity
        order.avg_price = (
            (order.avg_price * (order.filled_quantity - fill_quantity) + fill_price * fill_quantity)
            / order.filled_quantity
        )

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL_FILLED

        # 更新资金和持仓
        if order.side == OrderSide.BUY:
            self.cash -= fill_price * fill_quantity
            commission = fill_price * fill_quantity * 0.0003  # 万分之三手续费

            if order.symbol not in self.positions:
                self.positions[order.symbol] = {
                    "quantity": 0,
                    "avg_price": 0,
                }

            # 更新持仓成本
            old_qty = self.positions[order.symbol]["quantity"]
            old_cost = self.positions[order.symbol]["avg_price"] * old_qty
            new_qty = old_qty + fill_quantity
            self.positions[order.symbol]["quantity"] = new_qty
            self.positions[order.symbol]["avg_price"] = (old_cost + fill_price * fill_quantity) / new_qty
            self.positions[order.symbol]["current_price"] = fill_price

        else:  # SELL
            self.cash += fill_price * fill_quantity

            if order.symbol in self.positions:
                self.positions[order.symbol]["quantity"] -= fill_quantity
                if self.positions[order.symbol]["quantity"] <= 0:
                    del self.positions[order.symbol]

        # 记录成交
        trade = Trade(
            trade_id=f"TRD{datetime.now().strftime('%Y%m%d%H%M%S')}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            trade_time=datetime.now().strftime("%Y%m%d %H:%M:%S"),
            commission=fill_price * fill_quantity * 0.0003,
        )
        self.trades.append(trade)

        logger.info(
            f"模拟成交: {order.symbol}, {order.side.value}, "
            f"数量={fill_quantity}, 价格={fill_price:.2f}"
        )
