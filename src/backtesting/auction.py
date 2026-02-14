"""
集合竞价模拟模块

A股集合竞价规则：
1. 9:15-9:20: 可以申报和撤销订单
2. 9:20-9:25: 可以申报但不能撤销订单
3. 9:25: 撮合成交，产生开盘价

集合竞价成交原则：
1. 成交量最大化原则
2. 价格优先、时间优先原则

使用示例：
    from src.backtesting.auction import CallAuction, AuctionOrder

    auction = CallAuction()

    # 添加订单
    auction.add_order(AuctionOrder(symbol="600519.SH", side="buy", price=1800.0, quantity=100))
    auction.add_order(AuctionOrder(symbol="600519.SH", side="sell", price=1810.0, quantity=100))

    # 执行集合竞价
    result = auction.execute()
    print(f"开盘价: {result.open_price}")
    print(f"成交量: {result.volume}")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
from decimal import Decimal
from datetime import time
from loguru import logger


class AuctionPhase(Enum):
    """集合竞价阶段"""
    PRE_OPEN = "pre_open"           # 9:15之前
    FREE_CANCEL = "free_cancel"     # 9:15-9:20 可撤单
    NO_CANCEL = "no_cancel"         # 9:20-9:25 不可撤单
    MATCHING = "matching"           # 9:25 撮合
    CONTINUOUS = "continuous"       # 9:30后连续竞价


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class AuctionOrder:
    """集合竞价订单"""
    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: int
    time: time = field(default_factory=lambda: time(9, 15))

    def __lt__(self, other):
        """用于排序 - 价格优先、时间优先"""
        if self.side == OrderSide.BUY:
            # 买单：价格高的优先
            if self.price != other.price:
                return self.price > other.price
        else:
            # 卖单：价格低的优先
            if self.price != other.price:
                return self.price < other.price
        # 时间优先
        return self.time < other.time


@dataclass
class AuctionMatch:
    """竞价撮合结果"""
    buy_order_id: str
    sell_order_id: str
    price: Decimal
    quantity: int


@dataclass
class AuctionResult:
    """集合竞价结果"""
    symbol: str
    open_price: Optional[Decimal]  # 开盘价
    volume: int                    # 成交量
    turnover: Decimal              # 成交额
    buy_surplus: int               # 买方剩余（未成交）
    sell_surplus: int              # 卖方剩余（未成交）
    matches: List[AuctionMatch] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否成功产生开盘价"""
        return self.open_price is not None and self.volume > 0


class CallAuction:
    """
    集合竞价模拟器

    实现A股开盘集合竞价机制：
    1. 收集买卖订单
    2. 计算最大成交量价格
    3. 撮合成交
    """

    def __init__(self, tick_size: Decimal = Decimal("0.01")):
        """
        初始化

        Args:
            tick_size: 价格最小变动单位
        """
        self.tick_size = tick_size
        self.orders: Dict[str, List[AuctionOrder]] = {}  # symbol -> orders
        self.phase = AuctionPhase.PRE_OPEN

    def get_phase(self, t: time) -> AuctionPhase:
        """
        根据时间判断集合竞价阶段

        Args:
            t: 当前时间

        Returns:
            竞价阶段
        """
        if t < time(9, 15):
            return AuctionPhase.PRE_OPEN
        elif time(9, 15) <= t < time(9, 20):
            return AuctionPhase.FREE_CANCEL
        elif time(9, 20) <= t < time(9, 25):
            return AuctionPhase.NO_CANCEL
        elif t == time(9, 25):
            return AuctionPhase.MATCHING
        else:
            return AuctionPhase.CONTINUOUS

    def add_order(self, order: AuctionOrder) -> bool:
        """
        添加订单

        Args:
            order: 竞价订单

        Returns:
            是否成功添加
        """
        if order.symbol not in self.orders:
            self.orders[order.symbol] = []

        self.orders[order.symbol].append(order)
        logger.debug(f"集合竞价订单添加: {order.symbol} {order.side.value} {order.quantity}@{order.price}")
        return True

    def cancel_order(self, order_id: str, symbol: str, current_time: time) -> bool:
        """
        撤销订单

        Args:
            order_id: 订单ID
            symbol: 股票代码
            current_time: 当前时间

        Returns:
            是否成功撤销
        """
        phase = self.get_phase(current_time)

        # 9:20-9:25不可撤单
        if phase == AuctionPhase.NO_CANCEL:
            logger.warning(f"集合竞价不可撤单阶段: {current_time}")
            return False

        # 9:25后已撮合，不可撤单
        if phase == AuctionPhase.MATCHING or phase == AuctionPhase.CONTINUOUS:
            logger.warning(f"集合竞价已结束，不可撤单: {current_time}")
            return False

        if symbol not in self.orders:
            return False

        for i, order in enumerate(self.orders[symbol]):
            if order.order_id == order_id:
                del self.orders[symbol][i]
                logger.debug(f"集合竞价订单撤销: {order_id}")
                return True

        return False

    def calculate_open_price(
        self,
        symbol: str,
        prev_close: Decimal,
        limit_up: Optional[Decimal] = None,
        limit_down: Optional[Decimal] = None,
    ) -> Tuple[Optional[Decimal], int]:
        """
        计算开盘价

        原则：选取使成交量最大的价格

        Args:
            symbol: 股票代码
            prev_close: 前收盘价
            limit_up: 涨停价
            limit_down: 跌停价

        Returns:
            (开盘价, 成交量)
        """
        if symbol not in self.orders or not self.orders[symbol]:
            return (None, 0)

        orders = self.orders[symbol]
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]

        if not buy_orders or not sell_orders:
            return (None, 0)

        # 收集所有价格点
        prices = set()
        for o in orders:
            prices.add(o.price)

        # 添加涨跌停价格（作为边界）
        if limit_up:
            prices.add(limit_up)
        if limit_down:
            prices.add(limit_down)

        # 对每个价格计算可能的成交量
        best_price = None
        best_volume = 0

        for price in sorted(prices):
            # 该价格下买方愿意买入的总量
            buy_volume = sum(
                o.quantity for o in buy_orders
                if o.price >= price
            )

            # 该价格下卖方愿意卖出的总量
            sell_volume = sum(
                o.quantity for o in sell_orders
                if o.price <= price
            )

            # 成交量 = min(买量, 卖量)
            volume = min(buy_volume, sell_volume)

            if volume > best_volume:
                best_volume = volume
                best_price = price
            elif volume == best_volume and best_price is not None:
                # 成交量相同，选择更接近前收盘价的价格
                if abs(price - prev_close) < abs(best_price - prev_close):
                    best_price = price

        return (best_price, best_volume)

    def execute(
        self,
        symbol: str,
        prev_close: Decimal,
        limit_up: Optional[Decimal] = None,
        limit_down: Optional[Decimal] = None,
    ) -> AuctionResult:
        """
        执行集合竞价撮合

        Args:
            symbol: 股票代码
            prev_close: 前收盘价
            limit_up: 涨停价
            limit_down: 跌停价

        Returns:
            竞价结果
        """
        orders = self.orders.get(symbol, [])
        buy_orders = sorted([o for o in orders if o.side == OrderSide.BUY])
        sell_orders = sorted([o for o in orders if o.side == OrderSide.SELL])

        # 计算开盘价
        open_price, total_volume = self.calculate_open_price(
            symbol, prev_close, limit_up, limit_down
        )

        if open_price is None or total_volume == 0:
            logger.info(f"集合竞价未成交: {symbol}")
            return AuctionResult(
                symbol=symbol,
                open_price=None,
                volume=0,
                turnover=Decimal("0"),
                buy_surplus=len(buy_orders),
                sell_surplus=len(sell_orders),
            )

        # 执行撮合
        matches = []
        remaining_volume = total_volume

        for buy_order in buy_orders:
            if remaining_volume <= 0:
                break
            if buy_order.price < open_price:
                continue

            for sell_order in sell_orders:
                if remaining_volume <= 0:
                    break
                if sell_order.price > open_price:
                    continue

                # 计算可成交量
                match_qty = min(
                    buy_order.quantity,
                    sell_order.quantity,
                    remaining_volume
                )

                if match_qty > 0:
                    matches.append(AuctionMatch(
                        buy_order_id=buy_order.order_id,
                        sell_order_id=sell_order.order_id,
                        price=open_price,
                        quantity=match_qty
                    ))
                    remaining_volume -= match_qty

        turnover = open_price * total_volume

        logger.info(
            f"集合竞价成交: {symbol}, "
            f"开盘价={open_price}, 成交量={total_volume}, 成交额={turnover}"
        )

        # 计算未成交订单数
        buy_matched = set(m.buy_order_id for m in matches)
        sell_matched = set(m.sell_order_id for m in matches)

        return AuctionResult(
            symbol=symbol,
            open_price=open_price,
            volume=total_volume,
            turnover=turnover,
            buy_surplus=len([o for o in buy_orders if o.order_id not in buy_matched]),
            sell_surplus=len([o for o in sell_orders if o.order_id not in sell_matched]),
            matches=matches,
        )

    def clear(self, symbol: Optional[str] = None):
        """
        清空订单

        Args:
            symbol: 股票代码（None=清空全部）
        """
        if symbol:
            self.orders.pop(symbol, None)
        else:
            self.orders.clear()


class CallAuctionSimulator:
    """
    集合竞价模拟器（用于回测）

    集成到回测引擎中，模拟真实的开盘集合竞价过程
    """

    def __init__(self):
        """初始化"""
        self.auction = CallAuction()
        self.results: Dict[str, AuctionResult] = {}

    def simulate_open(
        self,
        symbol: str,
        prev_close: Decimal,
        limit_up: Optional[Decimal] = None,
        limit_down: Optional[Decimal] = None,
        bid_ask_spread: Decimal = Decimal("0.002"),  # 买卖价差
        avg_volume: int = 100000,                     # 平均成交量
        volatility: Decimal = Decimal("0.02"),        # 波动率
    ) -> AuctionResult:
        """
        模拟开盘集合竞价

        根据历史数据特征生成模拟的买卖订单，然后撮合

        Args:
            symbol: 股票代码
            prev_close: 前收盘价
            limit_up: 涨停价
            limit_down: 跌停价
            bid_ask_spread: 买卖价差比例
            avg_volume: 平均成交量
            volatility: 波动率

        Returns:
            竞价结果
        """
        import random

        # 清空旧订单
        self.auction.clear(symbol)

        # 生成模拟订单
        # 假设开盘价在前收盘价附近波动
        price_range = float(prev_close * volatility)

        # 生成买单（价格高于预期开盘价的买单）
        num_buy_orders = random.randint(50, 200)
        for i in range(num_buy_orders):
            # 买单价格：前收盘价 - 波动范围 到 涨停价
            buy_price_float = float(prev_close) + random.uniform(-price_range, price_range * 2)
            buy_price = Decimal(str(round(buy_price_float, 2)))

            if limit_up and buy_price > limit_up:
                buy_price = limit_up

            buy_qty = random.randint(100, max(100, avg_volume // 100))

            self.auction.add_order(AuctionOrder(
                order_id=f"BUY_{symbol}_{i}",
                symbol=symbol,
                side=OrderSide.BUY,
                price=buy_price,
                quantity=buy_qty,
            ))

        # 生成卖单
        num_sell_orders = random.randint(50, 200)
        for i in range(num_sell_orders):
            # 卖单价格：跌停价 到 前收盘价 + 波动范围
            sell_price_float = float(prev_close) + random.uniform(-price_range * 2, price_range)
            sell_price = Decimal(str(round(sell_price_float, 2)))

            if limit_down and sell_price < limit_down:
                sell_price = limit_down

            sell_qty = random.randint(100, max(100, avg_volume // 100))

            self.auction.add_order(AuctionOrder(
                order_id=f"SELL_{symbol}_{i}",
                symbol=symbol,
                side=OrderSide.SELL,
                price=sell_price,
                quantity=sell_qty,
            ))

        # 执行集合竞价
        result = self.auction.execute(symbol, prev_close, limit_up, limit_down)
        self.results[symbol] = result

        return result

    def get_open_price(self, symbol: str) -> Optional[Decimal]:
        """获取模拟的开盘价"""
        if symbol in self.results:
            return self.results[symbol].open_price
        return None
