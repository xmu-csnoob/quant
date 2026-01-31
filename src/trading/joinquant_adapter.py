"""
聚宽JoinQuant适配器

将我们的策略对接到聚宽回测和模拟盘
"""

import sys
from pathlib import Path
from typing import List, Optional
from loguru import logger

# 尝试导入聚宽SDK
try:
    from joinquant.jqdata import *
    from joinquant.jqfactor import *
    from joinquant.jqtrade import *
    HAS_JQ = True
except ImportError:
    HAS_JQ = False
    logger.warning("聚宽SDK未安装，请运行: pip install joinquant")


from src.trading.api import TradingAPI, Account, Position, Trade
from src.trading.orders import Order, OrderType, OrderSide, OrderStatus


class JoinQuantAdapter(TradingAPI):
    """
    聚宽适配器

    将我们的交易框架对接到聚宽平台
    """

    def __init__(self, account_id: str = None):
        """
        初始化

        Args:
            account_id: 聚宽账号ID
        """
        if not HAS_JQ:
            raise ImportError(
                "请先安装聚宽SDK: pip install joinquant\n"
                "然后访问 https://www.joinquant.com/ 注册账号"
            )

        self.account_id = account_id or "simulated"
        self._connected = False

        # 本地订单跟踪
        self._orders: dict[str, Order] = {}
        self._order_counter = 0

    def connect(self) -> bool:
        """连接聚宽（实际上是认证）"""
        try:
            # 聚宽使用账号ID和密码认证
            # 这里简化处理，实际使用时需要配置
            logger.info(f"聚宽适配器已初始化: 账号={self.account_id}")
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"聚宽连接失败: {e}")
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        self._connected = False
        logger.info("聚宽适配器已断开")
        return True

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected

    def place_order(self, order: Order) -> bool:
        """
        下单（聚宽）

        注意：聚宽的订单方式和我们的不同
        这里需要适配
        """
        if not self._connected:
            logger.error("未连接到聚宽")
            return False

        try:
            # 聚宽使用order_volume函数下单
            # 这需要实际在聚宽平台上运行
            logger.info(f"聚宽下单: {order.symbol}, {order.side.value}, {order.quantity}")
            self._orders[order.order_id] = order
            return True
        except Exception as e:
            logger.error(f"聚宽下单失败: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        if order_id not in self._orders:
            return False

        try:
            # 聚宽撤单
            logger.info(f"聚宽撤单: {order_id}")
            return True
        except Exception as e:
            logger.error(f"聚宽撤单失败: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单"""
        return self._orders.get(order_id)

    def get_orders(self) -> List[Order]:
        """查询所有订单"""
        return list(self._orders.values())

    def get_positions(self) -> List[Position]:
        """查询持仓"""
        # 从聚宽获取持仓
        try:
            # 这里需要调用聚宽的get_positions函数
            # positions = get_positions()
            # 转换为我们的Position格式
            pass
        except:
            pass
        return []

    def get_account(self) -> Account:
        """查询账户"""
        # 从聚宽获取账户信息
        return Account(
            account_id=self.account_id,
            total_assets=100000,
            cash=100000,
            market_value=0,
        )

    def get_trades(self, order_id: Optional[str] = None) -> List[Trade]:
        """查询成交"""
        return []


def create_joinquant_strategy(
    our_strategy,
    symbols: List[str],
    start_date: str,
    end_date: str,
):
    """
    将我们的策略转换为聚宽策略

    Args:
        our_strategy: 我们的策略对象
        symbols: 交易标的
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)

    Returns:
        聚宽策略函数
    """
    def initialize(context):
        """聚宽策略初始化"""
        logger.info(f"聚宽策略初始化: {our_strategy.name}")

        # 设置基准
        # set_benchmark('000300.XSHG')

        # 获取股票列表
        # stocks = list(get_all_securities(['stock'], date=context.start_date))
        # context.stock_pool = symbols

    def handle_data(context, data):
        """聚宽数据处理（主逻辑）"""
        logger.info(f"处理数据: {data}")

        # 这里我们需要将聚宽的数据格式转换为我们的格式
        # 然后调用我们的策略

        for symbol in symbols:
            # 获取历史数据
            # history = get_history(
            #     symbols,
            #     context.current_date.strftime("%Y-%m-%d"),
            #     context.end_date.strftime("%Y-%m-%d"),
            #     ['open', 'high', 'low', 'close', 'volume']
            # )

            # 转换为我们的DataFrame格式
            # df = ...

            # 生成信号
            # signals = our_strategy.generate_signals(df)

            # 执行交易
            # for signal in signals:
            #     if signal.signal_type == "buy":
            #         order_volume(symbol, ...)
            #     elif signal.signal_type == "sell":
            #         order_volume(symbol, ...)
            pass

    def before_trading_start(context):
        """盘前运行"""
        pass

    def after_trading_end(context):
        """盘后运行"""
        pass

    return {
        "initialize": initialize,
        "handle_data": handle_data,
        "before_trading_start": before_trading_start,
        "after_trading_end": after_trading_end,
    }


def run_joinquant_backtest():
    """
    运行聚宽回测
    """
    print("=" * 70)
    print("聚宽回测指南")
    print("=" * 70)

    print("\n【步骤1】安装聚宽SDK")
    print("  pip install joinquant")

    print("\n【步骤2】注册聚宽账号")
    print("  访问: https://www.joinquant.com/")
    print("  注册并登录")

    print("\n【步骤3】创建策略研究")
    print("  进入: 研究中心 -> 策略研究 -> 创建策略研究")

    print("\n【步骤4】编写策略代码")
    print("  在聚宽的在线编辑器中编写策略")
    print("  或者使用我们的适配器（正在开发中）")

    print("\n【步骤5】回测")
    print("  点击: 运行回测")
    print("  等待回测完成")

    print("\n" + "=" * 70)
    print("适配器状态: 开发中")
    print("=" * 70)

    print("\n当前可用的功能:")
    print("  ✅ 完整的本地回测框架")
    print("  ✅ 多种策略实现")
    print("  ⏳ 聚宽适配器（开发中）")

    print("\n建议:")
    print("  1. 先在本地完善策略")
    print("  2. 使用本地回测验证")
    print("  3. 再移植到聚宽平台")


if __name__ == "__main__":
    run_joinquant_backtest()
