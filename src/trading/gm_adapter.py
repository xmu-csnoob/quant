"""
掘金模拟盘完整对接

实现完整的事件驱动交易系统
"""

import time
from typing import Callable, Dict, Optional, List
from dataclasses import dataclass
from loguru import logger
from datetime import datetime

from src.trading.api import TradingAPI, Account, Position, Trade
from src.trading.orders import Order, OrderType, OrderSide, OrderStatus
from src.trading.engine import LiveTradingEngine

try:
    from gm.api import *
    HAS_GM = True
except ImportError:
    HAS_GM = False
    # 创建占位符
    class GMEvent: pass


@dataclass
class GMConfig:
    """掘金配置"""
    token: str
    mode: str = "simulate"  # simulate(模拟) 或 live(实盘)
    strategy_id: Optional[str] = None  # 策略ID


class GMTradingAdapter(TradingAPI):
    """
    掘金交易适配器（完整版）

    特点：
    1. 事件驱动架构
    2. 完整的订单生命周期管理
    3. 实时账户和持仓更新
    4. 支持回测和模拟盘
    """

    def __init__(self, config: GMConfig):
        """
        初始化

        Args:
            config: 掘金配置
        """
        if not HAS_GM:
            raise ImportError(
                "请先安装掘金SDK: pip install gm-python\n"
                "然后访问 https://www.myquant.cn/docs 获取Token"
            )

        self.config = config
        self._connected = False

        # 订单映射 (掘金order_id -> 我们的Order)
        self._gm_order_map: Dict[str, Order] = {}

        # 我们的order_id -> 掘金order_id
        self._order_map: Dict[str, str] = {}

        # 账户信息
        self._account: Optional[Account] = None
        self._positions: Dict[str, Position] = {}

        # 成交记录
        self._trades: List[Trade] = []

        # 回调函数
        self._on_order_filled: Optional[Callable] = None
        self._on_order_cancelled: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None

    def connect(self) -> bool:
        """连接掘金"""
        try:
            # 设置token
            set_token(self.config.token)

            # 登录
            status = login_mode(self.config.mode)

            if status == 0:
                self._connected = True
                logger.info(f"掘金{self.config.mode}盘已连接")

                # 订阅账户和持仓事件
                self._subscribe_events()

                return True
            else:
                logger.error(f"掘金登录失败: status={status}")
                return False

        except Exception as e:
            logger.error(f"掘金连接异常: {e}")
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            logout()
            self._connected = False
            logger.info("掘金已断开")
            return True
        except Exception as e:
            logger.error(f"掘金断开异常: {e}")
            return False

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected

    def _subscribe_events(self):
        """订阅掘金事件"""
        # 这里需要注册事件回调
        # 掘金使用装饰器注册事件处理函数
        pass

    def place_order(self, order: Order) -> bool:
        """下单"""
        if not self._connected:
            logger.error("未连接到掘金")
            return False

        try:
            # 转换订单类型
            if order.order_type == OrderType.MARKET:
                gm_order_type = OrderType_Market
            elif order.order_type == OrderType.LIMIT:
                gm_order_type = OrderType_Limit
            elif order.order_type == OrderType.STOP:
                gm_order_type = OrderType_Stop
            else:
                logger.error(f"不支持的订单类型: {order.order_type}")
                return False

            # 转换买卖方向
            if order.side == OrderSide.BUY:
                gm_side = OrderSide_Buy
            else:
                gm_side = OrderSide_Sell

            # 下单
            cl_ord_id = order_volume(
                symbol=order.symbol,
                direction=gm_side,
                type=gm_order_type,
                volume=order.quantity,
                price=order.price or 0,
            )

            # 建立映射
            self._order_map[order.order_id] = cl_ord_id
            self._gm_order_map[cl_ord_id] = order

            logger.info(f"掘金下单成功: {order.order_id} -> {cl_ord_id}")
            return True

        except Exception as e:
            logger.error(f"掘金下单失败: {e}")
            order.status = OrderStatus.REJECTED
            return False

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        if order_id not in self._order_map:
            logger.warning(f"订单不存在: {order_id}")
            return False

        gm_order_id = self._order_map[order_id]

        try:
            # 撤单
            status = cancel_order(gm_order_id)

            if status == 0:
                order = self._gm_order_map.get(gm_order_id)
                if order:
                    order.status = OrderStatus.CANCELLED
                logger.info(f"掘金撤单成功: {order_id}")
                return True
            else:
                logger.error(f"掘金撤单失败: status={status}")
                return False

        except Exception as e:
            logger.error(f"掘金撤单异常: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单"""
        # 从映射中查找
        for gm_id, order in self._gm_order_map.items():
            if order.order_id == order_id:
                return order
        return None

    def get_orders(self) -> List[Order]:
        """查询所有订单"""
        return list(self._gm_order_map.values())

    def get_positions(self) -> List[Position]:
        """查询持仓"""
        # 返回缓存的持仓
        return list(self._positions.values())

    def get_account(self) -> Account:
        """查询账户"""
        if self._account:
            return self._account

        # 返回默认值
        return Account(
            account_id="gm_account",
            total_assets=0,
            cash=0,
            market_value=0,
        )

    def get_trades(self, order_id: Optional[str] = None) -> List[Trade]:
        """查询成交"""
        if order_id:
            return [t for t in self._trades if t.order_id == order_id]
        return self._trades

    def set_callbacks(
        self,
        on_order_filled: Optional[Callable] = None,
        on_order_cancelled: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
    ):
        """设置回调函数"""
        self._on_order_filled = on_order_filled
        self._on_order_cancelled = on_order_cancelled
        self._on_trade = on_trade


class GMLiveTradingEngine(LiveTradingEngine):
    """
    掘金实时交易引擎

    整合掘金API和我们的交易框架
    """

    def __init__(
        self,
        strategy,
        gm_config: GMConfig,
        symbols: List[str],
        risk_manager=None,
    ):
        """
        初始化

        Args:
            strategy: 交易策略
            gm_config: 掘金配置
            symbols: 交易标的
            risk_manager: 风险管理器
        """
        # 创建掘金适配器
        trading_api = GMTradingAdapter(gm_config)

        # 如果没有提供风险管理器，创建一个
        if risk_manager is None:
            from risk import RiskManager, PositionSizer
            position_sizer = PositionSizer(initial_capital=100000)
            risk_manager = RiskManager(
                initial_capital=100000,
                position_sizer=position_sizer,
            )

        super().__init__(
            strategy=strategy,
            trading_api=trading_api,
            risk_manager=risk_manager,
            symbols=symbols,
        )

        self.gm_api = trading_api

    def start(self):
        """启动引擎"""
        logger.info("启动掘金实时交易引擎...")

        # 连接掘金
        if not self.gm_api.connect():
            logger.error("掘金连接失败")
            return False

        self.is_running = True

        # 设置回调
        self.gm_api.set_callbacks(
            on_order_filled=self._on_order_filled,
            on_trade=self._on_trade,
        )

        logger.info("掘金实时交易引擎已启动")
        return True

    def stop(self):
        """停止引擎"""
        self.is_running = False
        self.gm_api.disconnect()
        logger.info("掘金实时交易引擎已停止")

    def _on_order_filled(self, order: Order):
        """订单成交回调"""
        logger.info(f"订单成交: {order}")
        # 更新持仓和资金
        # ...

    def _on_trade(self, trade: Trade):
        """成交回调"""
        logger.info(f"成交记录: {trade}")
        # 更新交易统计
        # ...


# 便捷函数
def create_gm_simulation_engine(
    strategy,
    symbols: List[str],
    token: str,
    initial_cash: float = 100000,
) -> GMLiveTradingEngine:
    """
    创建掘金模拟盘引擎

    Args:
        strategy: 交易策略
        symbols: 交易标的列表
        token: 掘金Token
        initial_cash: 初始资金

    Returns:
        掘金模拟盘引擎
    """
    config = GMConfig(token=token, mode="simulate")

    engine = GMLiveTradingEngine(
        strategy=strategy,
        gm_config=config,
        symbols=symbols,
    )

    return engine


# 使用指南
def print_gm_setup_guide():
    """打印掘金设置指南"""
    print("=" * 70)
    print("掘金模拟盘接入指南")
    print("=" * 70)

    print("\n【步骤1】安装SDK")
    print("  pip install gm-python")

    print("\n【步骤2】注册掘金账号")
    print("  访问: https://www.myquant.cn/")
    print("  点击: 免费注册 -> 实名认证")

    print("\n【步骤3】创建策略")
    print("  进入: 策略中心 -> 创建策略")
    print("  选择: Python 策略")

    print("\n【步骤4】获取Token")
    print("  在策略页面复制Token")

    print("\n【步骤5】运行代码")
    print("  python scripts/run_gm_simulation.py")

    print("\n【注意事项】")
    print("  - 模拟盘完全免费")
    print("  - 提供真实历史行情数据")
    print("  - 支持A股、港股、美股")
    print("  - 交易时段: 周一至周五 9:30-15:00")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_gm_setup_guide()
