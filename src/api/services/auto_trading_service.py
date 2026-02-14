"""
Auto Trading Service - 自动交易服务

T+1盯盘交易：
1. 收盘后分析当天数据
2. 生成次日交易信号
3. 次日开盘执行交易

由于只有日线数据，交易决策基于前一日收盘数据
"""

from datetime import datetime, date
from typing import List, Optional, Dict
from loguru import logger

from src.data.storage.sqlite_storage import SQLiteStorage
from src.api.services.trading_service import TradingService
from src.strategies.base import Signal, SignalType
from src.strategies import MaMacdRsiStrategy
from src.strategies.mean_reversion import MeanReversionStrategy


class AutoTradingService:
    """自动交易服务"""

    def __init__(self, account_type: str = "paper"):
        self._storage = SQLiteStorage()
        self._trading_service = TradingService(account_type)
        self._account_type = account_type

        # 可用的策略
        self._strategies = {
            "ma_macd_rsi": MaMacdRsiStrategy(),
            "mean_reversion": MeanReversionStrategy(),
        }

        # 当前激活的策略
        self._active_strategy = "ma_macd_rsi"

    def set_strategy(self, strategy_id: str) -> bool:
        """设置使用的策略"""
        if strategy_id in self._strategies:
            self._active_strategy = strategy_id
            logger.info(f"切换策略: {strategy_id}")
            return True
        return False

    def get_available_stocks(self) -> List[str]:
        """获取有数据的股票列表"""
        try:
            return self._storage.get_all_stocks()
        except:
            return []

    def analyze_and_trade(self, stock_codes: Optional[List[str]] = None) -> Dict:
        """
        分析数据并执行交易

        流程：
        1. 获取持仓，检查卖出信号
        2. 获取观察池，检查买入信号
        3. 执行交易

        Args:
            stock_codes: 要分析的股票列表，None则使用持仓+默认池

        Returns:
            交易结果
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "strategy": self._active_strategy,
            "account_type": self._account_type,
            "signals": [],
            "trades": [],
            "errors": [],
        }

        strategy = self._strategies.get(self._active_strategy)
        if not strategy:
            result["errors"].append(f"策略不存在: {self._active_strategy}")
            return result

        # 1. 获取当前持仓
        positions_df = self._storage.get_positions(self._account_type)
        held_stocks = []
        if positions_df is not None and len(positions_df) > 0:
            held_stocks = positions_df['symbol'].tolist()

        # 2. 确定分析范围：持仓 + 默认观察池
        if stock_codes is None:
            # 默认观察几只大盘股
            watch_list = ["600000.SH", "600519.SH", "000858.SZ", "601318.SH"]
            stock_codes = list(set(held_stocks + watch_list))

        logger.info(f"开始分析 {len(stock_codes)} 只股票...")

        # 3. 分析每只股票
        for code in stock_codes:
            try:
                signals = self._analyze_stock(code, strategy)
                if signals:
                    result["signals"].extend(signals)
            except Exception as e:
                result["errors"].append(f"{code}: 分析失败 - {str(e)}")
                logger.error(f"分析 {code} 失败: {e}")

        # 4. 执行交易信号（先卖后买）
        sell_signals = [s for s in result["signals"] if s["signal_type"] == "sell"]
        buy_signals = [s for s in result["signals"] if s["signal_type"] == "buy"]

        # 执行卖出
        for sig in sell_signals:
            try:
                trade = self._execute_sell(sig)
                if trade:
                    result["trades"].append(trade)
            except Exception as e:
                result["errors"].append(f"{sig['code']}: 卖出失败 - {str(e)}")

        # 执行买入
        for sig in buy_signals:
            try:
                trade = self._execute_buy(sig)
                if trade:
                    result["trades"].append(trade)
            except Exception as e:
                result["errors"].append(f"{sig['code']}: 买入失败 - {str(e)}")

        logger.info(f"交易完成: 信号{len(result['signals'])}个, 交易{len(result['trades'])}笔")
        return result

    def _analyze_stock(self, code: str, strategy) -> List[Dict]:
        """分析单只股票"""
        signals = []

        # 获取最近100天数据
        df = self._storage.get_daily_prices(code)
        if df is None or len(df) < 60:
            logger.warning(f"{code}: 数据不足")
            return signals

        # 排序
        df = df.sort_values('trade_date')

        # 生成信号
        strategy_signals = strategy.generate_signals(df)

        # 只取最新的信号（今天的）
        if strategy_signals:
            latest = strategy_signals[-1]
            latest_date = latest.date

            # 检查是否是最新日期的信号
            latest_data_date = str(df.iloc[-1]['trade_date'])[:8]
            if latest_date == latest_data_date or latest_date.replace('-', '') == latest_data_date:
                signals.append({
                    "code": code,
                    "signal_type": "buy" if latest.signal_type == SignalType.BUY else "sell",
                    "price": latest.price,
                    "reason": latest.reason,
                    "date": latest_date,
                })

        return signals

    def _execute_sell(self, signal: Dict) -> Optional[Dict]:
        """执行卖出"""
        code = signal["code"]
        price = signal["price"]

        # 检查持仓
        positions_df = self._storage.get_positions(self._account_type)
        if positions_df is None or len(positions_df) == 0:
            return None

        pos = positions_df[positions_df['symbol'] == code]
        if len(pos) == 0:
            return None

        quantity = int(pos.iloc[0]['quantity'])
        if quantity <= 0:
            return None

        # 创建卖出订单
        from src.api.schemas.trading import CreateOrderRequest, OrderDirection, OrderType

        request = CreateOrderRequest(
            code=code,
            direction=OrderDirection.SELL,
            order_type=OrderType.LIMIT,
            price=price,
            shares=quantity,
        )

        order = self._trading_service.create_order(request)
        filled = self._trading_service.fill_order(order.order_id, price)

        if filled:
            return {
                "action": "sell",
                "code": code,
                "shares": quantity,
                "price": price,
                "amount": price * quantity,
            }
        return None

    def _execute_buy(self, signal: Dict) -> Optional[Dict]:
        """执行买入"""
        code = signal["code"]
        price = signal["price"]

        # 检查现金
        balance = self._storage.get_account_balance(self._account_type)
        cash = balance.get("cash", 0)

        if cash < 10000:  # 最小交易金额
            logger.warning(f"现金不足: {cash}")
            return None

        # 检查持仓数量限制
        positions_df = self._storage.get_positions(self._account_type)
        current_positions = len(positions_df) if positions_df is not None else 0
        if current_positions >= 3:  # 最多3只持仓
            logger.warning(f"持仓数量已达上限: {current_positions}")
            return None

        # 计算买入数量（单只股票最多30%资金）
        max_amount = cash * 0.3
        shares = int(max_amount / price / 100) * 100  # 向下取整到100股
        if shares < 100:
            logger.warning(f"资金不足买入100股: {code}")
            return None

        amount = shares * price
        if amount > cash:
            shares = int(cash / price / 100) * 100
            amount = shares * price

        # 创建买入订单
        from src.api.schemas.trading import CreateOrderRequest, OrderDirection, OrderType

        request = CreateOrderRequest(
            code=code,
            direction=OrderDirection.BUY,
            order_type=OrderType.LIMIT,
            price=price,
            shares=shares,
        )

        order = self._trading_service.create_order(request)
        filled = self._trading_service.fill_order(order.order_id, price)

        if filled:
            return {
                "action": "buy",
                "code": code,
                "shares": shares,
                "price": price,
                "amount": amount,
            }
        return None

    def get_status(self) -> Dict:
        """获取自动交易状态"""
        balance = self._storage.get_account_balance(self._account_type)
        positions_df = self._storage.get_positions(self._account_type)
        trades_df = self._storage.get_trades(account_type=self._account_type)

        positions = []
        if positions_df is not None and len(positions_df) > 0:
            for _, row in positions_df.iterrows():
                positions.append({
                    "code": row['symbol'],
                    "quantity": int(row['quantity']),
                    "avg_cost": float(row['avg_cost']),
                })

        trade_count = len(trades_df) if trades_df is not None else 0

        return {
            "account_type": self._account_type,
            "active_strategy": self._active_strategy,
            "cash": balance.get("cash", 0),
            "initial_capital": balance.get("initial_capital", 1000000),
            "positions": positions,
            "position_count": len(positions),
            "trade_count": trade_count,
        }


# 单例
auto_trading_service = AutoTradingService("paper")
