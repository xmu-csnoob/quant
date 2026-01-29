"""
掘金模拟盘API适配器

对接掘金平台的模拟盘交易
"""

from typing import List, Optional
from loguru import logger

try:
    from gm.api import *
    HAS_GM = True
except ImportError:
    HAS_GM = False
    GMInstrument = None
    GMOrder = None

from trading.api import TradingAPI, Account, Position, Trade, Order
from trading.orders import OrderType, OrderSide, OrderStatus


class GMMockAdapter(TradingAPI):
    """
    掘金模拟盘适配器

    使用说明：
    1. 安装掘金终端：https://www.myquant.cn/docs
    2. 安装Python SDK：pip install gm-python
    3. 注册账号并申请模拟盘
    4. 运行此代码
    """

    def __init__(self, token: str, mode: str = "simulate"):
        """
        初始化

        Args:
            token: 掘金token
            mode: "simulate"(模拟) 或 "live"(实盘)
        """
        if not HAS_GM:
            raise ImportError("请先安装gm-python: pip install gm-python")

        self.token = token
        self.mode = mode
        self._connected = False

        # 订单映射
        self._order_map: dict[str, Order] = {}

    def connect(self) -> bool:
        """连接掘金"""
        try:
            # 设置token
            set_token(self.token)

            # 登录
            if self.mode == "simulate":
                status = login_mode("模拟")
            else:
                status = login_mode("实盘")

            if status == 0:
                self._connected = True
                logger.info(f"掘金{self.mode}盘已连接")
                return True
            else:
                logger.error(f"掘金登录失败: {status}")
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

    def place_order(self, order: Order) -> bool:
        """
        下单

        注意：掘金使用不同的事件驱动模式
        这里简化处理
        """
        if not self._connected:
            logger.error("未连接到掘金")
            return False

        try:
            # 转换订单类型
            if order.order_type == OrderType.MARKET:
                order_type = OrderType_Market
            elif order.order_type == OrderType.LIMIT:
                order_type = OrderType_Limit
            else:
                logger.error(f"不支持的订单类型: {order.order_type}")
                return False

            # 转换买卖方向
            if order.side == OrderSide.BUY:
                side = OrderSide_Buy
            else:
                side = OrderSide_Sell

            # 下单
            if order.stop_price:
                # 止损单
                cl_ord_id = order_volume(
                    symbol=order.symbol,
                    direction=side,
                    type=order_type,
                )
            else:
                # 普通单
                cl_ord_id = order_volume(
                    symbol=order.symbol,
                    direction=side,
                    type=order_type,
                    volume=order.quantity,
                    price=order.price,
                )

            self._order_map[cl_ord_id] = order
            logger.info(f"掘金下单成功: {cl_ord_id}")
            return True

        except Exception as e:
            logger.error(f"掘金下单失败: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        # 需要在掘金中实现撤单逻辑
        logger.warning("掘金撤单功能待实现")
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单"""
        # 从映射中查找
        for gm_id, order in self._order_map.items():
            if order.order_id == order_id:
                return order
        return None

    def get_orders(self) -> List[Order]:
        """查询所有订单"""
        return list(self._order_map.values())

    def get_positions(self) -> List[Position]:
        """查询持仓"""
        # 掘金需要通过事件获取持仓
        # 这里返回空列表
        return []

    def get_account(self) -> Account:
        """查询账户"""
        # 掘金需要通过事件获取账户
        # 返回默认值
        return Account(
            account_id="gm_account",
            total_assets=0,
            cash=0,
            market_value=0,
        )

    def get_trades(self, order_id: Optional[str] = None) -> List[Trade]:
        """查询成交"""
        # 掘金需要通过事件获取成交
        return []


# 使用示例
def example_use_gm():
    """
    掘金模拟盘使用示例
    """
    print("=" * 60)
    print("掘金模拟盘接入示例")
    print("=" * 60)

    # 1. 获取token
    print("\n[1] 获取掘金Token")
    print("  访问: https://www.myquant.cn/docs")
    print("  注册账号 -> 申请模拟盘 -> 复制Token")

    token = input("  请输入Token: ").strip()

    if not token:
        print("  Token为空，使用模拟模式")
        return

    # 2. 创建适配器
    print("\n[2] 创建适配器")
    adapter = GMMockAdapter(token=token, mode="simulate")

    # 3. 连接
    print("\n[3] 连接掘金...")
    if adapter.connect():
        print("  ✓ 连接成功")
    else:
        print("  ✗ 连接失败")
        return

    # 4. 查询账户
    print("\n[4] 查询账户...")
    account = adapter.get_account()
    print(f"  总资产: {account.total_assets:.2f}")
    print(f"  现金: {account.cash:.2f}")
    print(f"  市值: {account.market_value:.2f}")

    # 5. 断开连接
    print("\n[5] 断开连接...")
    adapter.disconnect()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if HAS_GM:
        example_use_gm()
    else:
        print("请先安装掘金SDK:")
        print("  pip install gm-python")
        print("\n然后访问 https://www.myquant.cn/docs 获取Token")
