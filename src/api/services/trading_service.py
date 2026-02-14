"""
Trading service - 交易服务层
支持虚拟账户(paper)和实盘账户(live)
"""

import uuid
from datetime import datetime, date
from typing import List, Optional
from loguru import logger

from src.api.schemas.trading import (
    Order, CreateOrderRequest, OrderStatus, OrderDirection, OrderType
)
from src.data.storage.sqlite_storage import SQLiteStorage


class TradingService:
    """交易服务 - 支持虚拟/实盘账户"""

    def __init__(self, account_type: str = "paper"):
        """
        初始化交易服务

        Args:
            account_type: 账户类型 'paper'(虚拟) 或 'live'(实盘)
        """
        self._storage = SQLiteStorage()
        self._account_type = account_type
        self._pending_orders: dict[str, Order] = {}

    def get_stock_name(self, code: str) -> str:
        """从数据库获取股票名称"""
        try:
            df = self._storage.get_daily_prices(code)
            if df is not None and len(df) > 0 and 'name' in df.columns:
                return df.iloc[0]['name']
        except Exception as e:
            logger.warning(f"获取股票名称失败 {code}: {e}")
        return code

    def get_orders(
        self,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[Order], int]:
        """获取订单列表"""
        orders = []

        try:
            df = self._storage.get_trades(account_type=self._account_type)
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    side = row.get('side', 'buy')
                    direction = OrderDirection.BUY if side == 'buy' else OrderDirection.SELL

                    trade_date = row.get('trade_date', '')
                    if isinstance(trade_date, str):
                        try:
                            created_at = datetime.fromisoformat(trade_date)
                        except:
                            created_at = datetime.now()
                    else:
                        created_at = trade_date

                    orders.append(Order(
                        order_id=str(row.get('id', uuid.uuid4())),
                        code=row.get('symbol', ''),
                        name=self.get_stock_name(row.get('symbol', '')),
                        direction=direction,
                        order_type=OrderType.MARKET,
                        price=float(row.get('price', 0)),
                        shares=int(row.get('quantity', 0)),
                        filled_shares=int(row.get('quantity', 0)),
                        status=OrderStatus.FILLED,
                        created_at=created_at,
                        updated_at=created_at
                    ))

            for order in self._pending_orders.values():
                if status is None or order.status == status:
                    orders.append(order)

            orders.sort(key=lambda x: x.created_at, reverse=True)

            if status:
                orders = [o for o in orders if o.status == status]

        except Exception as e:
            logger.error(f"获取订单列表失败: {e}")

        total = len(orders)
        start = (page - 1) * page_size
        end = start + page_size

        return orders[start:end], total

    def create_order(self, request: CreateOrderRequest) -> Order:
        """创建订单"""
        order_id = str(uuid.uuid4())
        now = datetime.now()

        order = Order(
            order_id=order_id,
            code=request.code,
            name=self.get_stock_name(request.code),
            direction=request.direction,
            order_type=request.order_type,
            price=request.price,
            shares=request.shares,
            filled_shares=0,
            status=OrderStatus.SUBMITTED,
            created_at=now,
            updated_at=now
        )

        self._pending_orders[order_id] = order
        logger.info(f"创建订单: {order_id[:8]}... {request.direction.value} {request.code} {request.shares}股 [{self._account_type}]")
        return order

    def fill_order(self, order_id: str, filled_price: Optional[float] = None) -> Optional[Order]:
        """订单成交"""
        if order_id not in self._pending_orders:
            logger.warning(f"订单不存在: {order_id}")
            return None

        order = self._pending_orders[order_id]
        if order.status != OrderStatus.SUBMITTED:
            return None

        order.status = OrderStatus.FILLED
        order.filled_shares = order.shares
        order.updated_at = datetime.now()

        side = 'buy' if order.direction == OrderDirection.BUY else 'sell'
        price = filled_price or order.price or 0.0
        amount = price * order.shares

        try:
            # 保存交易记录
            self._storage.save_trade(
                symbol=order.code,
                trade_date=order.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                side=side,
                price=price,
                quantity=order.shares,
                reason=f"订单成交: {order_id[:8]}",
                account_type=self._account_type
            )

            # 更新持仓
            self._update_position(order.code, order.direction, order.shares, price, amount)

            del self._pending_orders[order_id]

            logger.info(f"订单成交: {order.code} {side} {order.shares}股 @ {price} [{self._account_type}]")

        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
            order.status = OrderStatus.REJECTED

        return order

    def _update_position(self, code: str, direction: OrderDirection, shares: int, price: float, amount: float):
        """更新持仓和现金"""
        try:
            df = self._storage.get_positions(self._account_type)
            current_qty = 0
            current_cost = 0.0

            if df is not None and len(df) > 0:
                pos = df[df['symbol'] == code]
                if len(pos) > 0:
                    current_qty = int(pos.iloc[0]['quantity'])
                    current_cost = float(pos.iloc[0]['avg_cost'])

            if direction == OrderDirection.BUY:
                new_qty = current_qty + shares
                new_cost = (current_cost * current_qty + price * shares) / new_qty if new_qty > 0 else price
                cash_delta = -amount  # 买入减少现金
            else:
                new_qty = current_qty - shares
                new_cost = current_cost
                cash_delta = amount  # 卖出增加现金

            if new_qty <= 0:
                self._storage.delete_position(code, self._account_type)
            else:
                latest_price = self._get_latest_price(code) or price
                market_value = latest_price * new_qty
                unrealized_pnl = (latest_price - new_cost) * new_qty

                self._storage.save_position(
                    symbol=code,
                    quantity=new_qty,
                    avg_cost=new_cost,
                    current_price=latest_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    account_type=self._account_type
                )

            # 更新现金
            balance = self._storage.get_account_balance(self._account_type)
            new_cash = balance.get("cash", 1000000.0) + cash_delta
            self._storage.update_account_balance(new_cash, self._account_type)

        except Exception as e:
            logger.error(f"更新持仓失败: {e}")

    def _get_latest_price(self, code: str) -> Optional[float]:
        """获取最新价格"""
        try:
            df = self._storage.get_daily_prices(code)
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date', ascending=False)
                return float(df.iloc[0]['close'])
        except Exception as e:
            logger.warning(f"获取最新价格失败 {code}: {e}")
        return None

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self._pending_orders:
            return False

        order = self._pending_orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        del self._pending_orders[order_id]

        logger.info(f"订单已取消: {order_id[:8]}")
        return True


# 单例 - 默认虚拟账户
paper_trading_service = TradingService("paper")
live_trading_service = TradingService("live")

# 默认使用虚拟账户
trading_service = paper_trading_service
