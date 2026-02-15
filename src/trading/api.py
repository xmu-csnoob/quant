"""
交易API接口

定义统一的交易接口，支持实盘和模拟盘
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from loguru import logger

from src.trading.orders import Order, OrderType, OrderSide, OrderStatus

if TYPE_CHECKING:
    from src.backtesting.slippage import BaseSlippage


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

    用于回测和模拟盘测试，支持滑点模拟和T+1规则

    T+1规则说明：
    - A股T+1规则：当日买入的股票，当日不可卖出，需次日才能卖出
    - 本实现按"批次"追踪持仓，每笔买入记录买入日期
    - 卖出时优先卖出可卖批次（非当日买入的）
    """

    def __init__(
        self,
        initial_cash: float = 100000,
        slippage_model: Optional["BaseSlippage"] = None,
        enable_t1_rule: bool = True,
    ):
        """
        初始化

        Args:
            initial_cash: 初始资金
            slippage_model: 滑点模型（可选）
            enable_t1_rule: 是否启用T+1规则（A股当日买入次日才能卖出）
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        # 持仓结构：{symbol: {"lots": [{"quantity": 100, "buy_date": "20240101", "price": 10.0}], "current_price": 10.5}}
        self.positions: dict[str, dict] = {}
        self.orders: dict[str, Order] = {}
        self.trades: list[Trade] = []
        self._connected = False
        self.slippage_model = slippage_model
        self.enable_t1_rule = enable_t1_rule
        self._current_date: Optional[str] = None  # 当前交易日期

    def set_current_date(self, date_str: str):
        """
        设置当前交易日期（用于T+1检查）

        Args:
            date_str: 日期字符串 (YYYYMMDD 或 YYYY-MM-DD)
        """
        self._current_date = date_str.replace("-", "")[:8]

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

            # 计算可卖数量（T+1规则）
            available = self._get_available_quantity(order.symbol)
            if available < order.quantity:
                locked = self.positions[order.symbol]["quantity"] - available
                logger.warning(
                    f"可卖数量不足: 需要{order.quantity}, 可用{available}（锁定{locked}因T+1规则）"
                )
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

    def _get_available_quantity(self, symbol: str) -> int:
        """
        获取可卖数量（考虑T+1规则，按批次计算）

        Args:
            symbol: 股票代码

        Returns:
            可卖数量（非当日买入的批次总和）
        """
        if symbol not in self.positions:
            return 0

        pos = self.positions[symbol]
        lots = pos.get("lots", [])

        if not self.enable_t1_rule:
            # T+1禁用时，所有持仓都可卖
            return sum(lot["quantity"] for lot in lots)

        # 获取当前日期
        current_date = self._current_date or datetime.now().strftime("%Y%m%d")

        # 计算非当日买入的批次数量（可卖）
        available = sum(
            lot["quantity"]
            for lot in lots
            if lot.get("buy_date", "") != current_date
        )

        return available

    def _get_total_quantity(self, symbol: str) -> int:
        """获取总持仓数量"""
        if symbol not in self.positions:
            return 0
        return sum(lot["quantity"] for lot in self.positions[symbol].get("lots", []))

    def _get_avg_price(self, symbol: str) -> float:
        """计算加权平均成本价"""
        if symbol not in self.positions:
            return 0
        lots = self.positions[symbol].get("lots", [])
        if not lots:
            return 0
        total_qty = sum(lot["quantity"] for lot in lots)
        if total_qty == 0:
            return 0
        total_cost = sum(lot["quantity"] * lot["price"] for lot in lots)
        return total_cost / total_qty

    def get_positions(self) -> List[Position]:
        """查询持仓"""
        result = []
        for symbol, pos in self.positions.items():
            lots = pos.get("lots", [])
            quantity = sum(lot["quantity"] for lot in lots)
            if quantity <= 0:
                continue

            avg_price = self._get_avg_price(symbol)
            current_price = pos.get("current_price", avg_price)
            available_quantity = self._get_available_quantity(symbol)

            result.append(Position(
                symbol=symbol,
                quantity=quantity,
                available_quantity=available_quantity,
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
        market_value = 0
        for pos in self.positions.values():
            lots = pos.get("lots", [])
            current_price = pos.get("current_price", 0)
            total_qty = sum(lot["quantity"] for lot in lots)
            market_value += total_qty * current_price

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
        current_time: Optional[time] = None,
    ):
        """
        模拟成交

        Args:
            order: 订单
            fill_price: 成交价格
            fill_quantity: 成交数量
            current_time: 当前时间（用于滑点计算）
        """
        if fill_quantity <= 0:
            return

        # 应用滑点模型
        actual_fill_price = fill_price
        if self.slippage_model:
            from src.backtesting.costs import TradeSide

            trade_side = TradeSide.BUY if order.side == OrderSide.BUY else TradeSide.SELL
            result = self.slippage_model.apply_slippage(
                price=Decimal(str(fill_price)),
                side=trade_side,
                current_time=current_time,
            )
            actual_fill_price = float(result.adjusted_price)
            logger.debug(
                f"滑点调整: {fill_price:.3f} -> {actual_fill_price:.3f} "
                f"(滑点: {float(result.slippage_rate)*100:.4f}%)"
            )

        # 更新订单
        order.filled_quantity += fill_quantity
        order.avg_price = (
            (order.avg_price * (order.filled_quantity - fill_quantity) + actual_fill_price * fill_quantity)
            / order.filled_quantity
        )

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL_FILLED

        # 更新资金和持仓
        if order.side == OrderSide.BUY:
            self.cash -= actual_fill_price * fill_quantity
            commission = actual_fill_price * fill_quantity * 0.0003  # 万分之三手续费

            # 获取当前交易日期
            trade_date = self._current_date or datetime.now().strftime("%Y%m%d")

            # 初始化持仓结构
            if order.symbol not in self.positions:
                self.positions[order.symbol] = {"lots": [], "current_price": actual_fill_price}

            # 添加新的批次（用于T+1追踪）
            self.positions[order.symbol]["lots"].append({
                "quantity": fill_quantity,
                "buy_date": trade_date,
                "price": actual_fill_price,
            })
            self.positions[order.symbol]["current_price"] = actual_fill_price

        else:  # SELL
            self.cash += actual_fill_price * fill_quantity

            if order.symbol in self.positions:
                lots = self.positions[order.symbol].get("lots", [])
                current_date = self._current_date or datetime.now().strftime("%Y%m%d")

                # 按FIFO原则卖出（优先卖出非当日买入的批次）
                remaining_to_sell = fill_quantity

                # 先尝试卖出可卖的批次（非当日买入）
                new_lots = []
                for lot in lots:
                    if remaining_to_sell <= 0:
                        new_lots.append(lot)
                        continue

                    # 当日买入的批次跳过（不可卖）
                    if self.enable_t1_rule and lot.get("buy_date", "") == current_date:
                        new_lots.append(lot)
                        continue

                    # 卖出此批次
                    if lot["quantity"] <= remaining_to_sell:
                        remaining_to_sell -= lot["quantity"]
                        # 批次全部卖出，不保留
                    else:
                        # 部分卖出
                        lot["quantity"] -= remaining_to_sell
                        remaining_to_sell = 0
                        new_lots.append(lot)

                self.positions[order.symbol]["lots"] = new_lots

                # 如果没有持仓了，删除
                if not new_lots:
                    del self.positions[order.symbol]

        # 记录成交
        trade = Trade(
            trade_id=f"TRD{datetime.now().strftime('%Y%m%d%H%M%S')}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=actual_fill_price,
            trade_time=datetime.now().strftime("%Y%m%d %H:%M:%S"),
            commission=actual_fill_price * fill_quantity * 0.0003,
        )
        self.trades.append(trade)

        logger.info(
            f"模拟成交: {order.symbol}, {order.side.value}, "
            f"数量={fill_quantity}, 价格={actual_fill_price:.2f}"
        )
