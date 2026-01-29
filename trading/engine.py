"""
实时交易引擎

整合策略、风控和交易API，实现自动交易
"""

import time
import threading
from typing import Optional, List
from datetime import datetime
from loguru import logger

from trading.orders import OrderManager, Order, OrderSide, OrderType
from trading.api import TradingAPI, MockTradingAPI
from strategies.base import BaseStrategy, Signal
from risk import RiskManager, PositionSizer


class LiveTradingEngine:
    """
    实时交易引擎

    功能：
    1. 订阅行情数据
    2. 执行策略信号
    3. 风险检查
    4. 订单管理
    5. 交易日志
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        trading_api: TradingAPI,
        risk_manager: RiskManager,
        symbols: List[str],
    ):
        """
        初始化

        Args:
            strategy: 交易策略
            trading_api: 交易API
            risk_manager: 风险管理器
            symbols: 交易标的列表
        """
        self.strategy = strategy
        self.api = trading_api
        self.risk_manager = risk_manager
        self.symbols = symbols

        self.order_manager = OrderManager()
        self.is_running = False
        self.last_signal_time = {}  # 上次信号时间 {symbol: datetime}

        logger.info(
            f"LiveTradingEngine initialized: "
            f"strategy={strategy.name}, symbols={symbols}"
        )

    def start(self):
        """启动交易引擎"""
        if not self.api.connect():
            logger.error("连接交易API失败")
            return False

        self.is_running = True
        logger.info("实时交易引擎已启动")

        # 在实际应用中，这里会启动行情订阅和处理线程
        # 这里简化为单次运行
        self._run_once()

        return True

    def stop(self):
        """停止交易引擎"""
        self.is_running = False
        self.api.disconnect()
        logger.info("实时交易引擎已停止")

    def _run_once(self):
        """
        运行一次交易循环（简化版）

        实际应用中应该是：
        1. 订阅实时行情
        2. 接收行情推送
        3. 触发策略生成信号
        4. 执行风控检查
        5. 下单
        """
        logger.info("=" * 60)
        logger.info("开始交易循环")
        logger.info("=" * 60)

        for symbol in self.symbols:
            logger.info(f"\n处理标的: {symbol}")

            # 1. 获取最新行情（这里简化处理，实际应该订阅）
            # 在实际应用中，这里会从行情接口获取实时数据
            # 这里我们跳过，直接进入信号处理

            # 2. 检查活跃订单
            active_orders = self.order_manager.get_active_orders(symbol)
            if active_orders:
                logger.info(f"  当前有 {len(active_orders)} 个活跃订单")
                for order in active_orders:
                    logger.info(f"    - {order}")

            # 3. 处理策略信号（这里需要实际数据）
            # signal = self._get_latest_signal(symbol)
            # if signal:
            #     self._process_signal(signal)

    def process_signal(self, signal: Signal, current_price: float) -> Optional[Order]:
        """
        处理策略信号

        Args:
            signal: 交易信号
            current_price: 当前价格

        Returns:
            创建的订单（如果成功），None otherwise
        """
        logger.info(f"\n处理信号: {signal}")

        symbol = signal.date.split("_")[0] if "_" in signal.date else self.symbols[0]
        date_str = signal.date

        # 1. 风控检查
        if signal.signal_type.value == "buy":
            from strategies.base import SignalType

            # 检查是否允许开仓
            allowed, pos_size, reason = self.risk_manager.check_entry(
                price=current_price,
                signal_confidence=signal.confidence if signal.confidence else 0.5,
                stock_code=symbol,
            )

            if not allowed:
                logger.warning(f"  风控拒绝开仓: {reason}")
                return None

            logger.info(f"  风控通过: {pos_size.reason}")

            # 2. 创建订单
            order = self.order_manager.create_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=pos_size.shares,
                price=current_price * 1.01,  # 稍高于市价确保成交
                reason=signal.reason,
            )

        elif signal.signal_type.value == "sell":
            # 检查是否有持仓
            positions = self.api.get_positions()
            has_position = any(p.symbol == symbol for p in positions)

            if not has_position:
                logger.warning(f"  无持仓，无法卖出")
                return None

            # 创建卖单
            order = self.order_manager.create_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=positions[0].available_quantity,  # 卖出全部
                price=current_price * 0.99,  # 稍低于市价确保成交
                reason=signal.reason,
            )

        else:
            logger.warning(f"  未知的信号类型: {signal.signal_type}")
            return None

        # 3. 提交订单
        if self.api.place_order(order):
            logger.info(f"  订单已提交: {order.order_id}")
            return order
        else:
            logger.error(f"  订单提交失败")
            return None

    def check_risk_controls(self, symbol: str, current_price: float):
        """
        检查风控（止损止盈等）

        Args:
            symbol: 股票代码
            current_price: 当前价格
        """
        risk_check = self.risk_manager.check_exit(symbol, current_price, datetime.now().strftime("%Y%m%d"))

        if not risk_check.passed:
            logger.warning(f"  触发风控: {risk_check.reason}")

            # 获取持仓
            positions = self.api.get_positions()
            position = next((p for p in positions if p.symbol == symbol), None)

            if position:
                # 创建平仓订单
                order = self.order_manager.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    reason=risk_check.reason,
                )

                if self.api.place_order(order):
                    logger.info(f"  风控平仓订单已提交: {order.order_id}")

    def get_status(self) -> dict:
        """
        获取引擎状态

        Returns:
            状态信息字典
        """
        orders_summary = self.order_manager.get_order_summary()
        account = self.api.get_account()
        positions = self.api.get_positions()

        return {
            "is_running": self.is_running,
            "api_connected": self.api.is_connected(),
            "symbols": self.symbols,
            "orders_summary": orders_summary,
            "account": {
                "total_assets": account.total_assets,
                "cash": account.cash,
                "market_value": account.market_value,
            },
            "positions_count": len(positions),
        }

    def print_status(self):
        """打印状态"""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("交易引擎状态")
        print("=" * 60)
        print(f"运行中: {status['is_running']}")
        print(f"API连接: {status['api_connected']}")
        print(f"交易标的: {status['symbols']}")
        print(f"订单: 总计{status['orders_summary']['total_orders']}, "
              f"活跃{status['orders_summary']['active_orders']}, "
              f"成交{status['orders_summary']['filled_orders']}")
        print(f"账户: 总资产={status['account']['total_assets']:.2f}, "
              f"现金={status['account']['cash']:.2f}, "
              f"市值={status['account']['market_value']:.2f}")
        print(f"持仓: {status['positions_count']} 个")
        print("=" * 60)


def create_paper_trading_engine(
    strategy: BaseStrategy,
    symbols: List[str],
    initial_cash: float = 100000,
) -> LiveTradingEngine:
    """
    创建模拟盘交易引擎

    Args:
        strategy: 交易策略
        symbols: 交易标的
        initial_cash: 初始资金

    Returns:
        交易引擎
    """
    # 创建模拟API
    api = MockTradingAPI(initial_cash=initial_cash)

    # 创建仓位管理器
    position_sizer = PositionSizer(
        initial_capital=initial_cash,
        method="fixed_ratio",
    )

    # 创建风险管理器
    risk_manager = RiskManager(
        initial_capital=initial_cash,
        position_sizer=position_sizer,
        stop_loss=0.05,
        take_profit=0.15,
    )

    # 创建交易引擎
    engine = LiveTradingEngine(
        strategy=strategy,
        trading_api=api,
        risk_manager=risk_manager,
        symbols=symbols,
    )

    return engine
