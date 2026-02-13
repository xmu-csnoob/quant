"""
T+1管理器 - A股T+1交易规则管理

A股T+1规则：
- 当日买入的股票，下一交易日才能卖出
- 需要追踪每笔买入的日期和数量
- 计算可卖数量 = T-1日及之前的持仓

使用示例:
    t1_manager = T1Manager()

    # 记录买入
    t1_manager.record_buy("600519.SH", 100, date(2024, 1, 15))

    # 查询可卖数量
    available = t1_manager.get_available_shares("600519.SH", date(2024, 1, 15))  # 0
    available = t1_manager.get_available_shares("600519.SH", date(2024, 1, 16))  # 100

    # 扣减可卖数量（卖出时调用）
    t1_manager.deduct_available("600519.SH", 50, date(2024, 1, 16))
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class BuyRecord:
    """买入记录"""
    code: str                    # 股票代码
    shares: int                  # 买入数量
    buy_date: date              # 买入日期
    remaining_shares: int       # 剩余数量
    price: float = 0.0          # 买入价格（可选，用于计算成本）
    available_date: Optional[date] = None  # 可卖日期 (T+1)，自动计算

    def __post_init__(self):
        # T+1规则：买入后下一交易日可卖
        if self.available_date is None:
            self.available_date = self._calculate_available_date(self.buy_date)

    def _calculate_available_date(self, buy_date: date) -> date:
        """
        计算可卖日期
        A股T+1：下一交易日
        注意：这里简化处理，实际需要考虑节假日
        """
        # 简单处理：下一交易日 = 买入日期 + 1天（跳过周末）
        next_day = buy_date + timedelta(days=1)
        # 跳过周末
        while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
            next_day += timedelta(days=1)
        return next_day

    @property
    def is_available(self) -> bool:
        """是否已经可以卖出"""
        return date.today() >= self.available_date


@dataclass
class PositionLot:
    """持仓批次（FIFO管理）"""
    code: str
    shares: int                  # 该批次数量
    buy_date: date              # 买入日期
    available_date: date        # 可卖日期
    cost_price: float           # 成本价
    remaining: int = field(init=False)

    def __post_init__(self):
        self.remaining = self.shares

    @property
    def is_available_today(self) -> bool:
        """今天是否可卖"""
        return date.today() >= self.available_date


class T1Manager:
    """
    T+1管理器

    功能：
    1. 记录买入交易
    2. 计算可卖数量
    3. 管理持仓批次（FIFO）
    4. 检查T+1约束
    """

    def __init__(self):
        # 按股票代码存储买入记录
        # {code: [BuyRecord, ...]}
        self._buy_records: Dict[str, List[BuyRecord]] = defaultdict(list)

        # 持仓批次（FIFO队列）
        # {code: [PositionLot, ...]}
        self._position_lots: Dict[str, List[PositionLot]] = defaultdict(list)

        logger.info("T1Manager 初始化完成")

    def record_buy(
        self,
        code: str,
        shares: int,
        buy_date: date,
        price: float = 0.0
    ) -> None:
        """
        记录买入交易

        Args:
            code: 股票代码
            shares: 买入数量
            buy_date: 买入日期
            price: 买入价格（可选）
        """
        if shares <= 0:
            logger.warning(f"买入数量无效: {shares}")
            return

        # 创建买入记录
        record = BuyRecord(
            code=code,
            shares=shares,
            buy_date=buy_date,
            remaining_shares=shares,
            price=price
        )
        self._buy_records[code].append(record)

        # 创建持仓批次
        lot = PositionLot(
            code=code,
            shares=shares,
            buy_date=buy_date,
            available_date=record.available_date,
            cost_price=price
        )
        self._position_lots[code].append(lot)

        logger.debug(
            f"记录买入: {code} {shares}股 @ {price}, "
            f"买入日:{buy_date}, 可卖日:{record.available_date}"
        )

    def get_available_shares(self, code: str, current_date: Optional[date] = None) -> int:
        """
        获取可卖数量

        Args:
            code: 股票代码
            current_date: 当前日期（默认今天）

        Returns:
            可卖数量（T+1日及之前的持仓）
        """
        if current_date is None:
            current_date = date.today()

        available = 0
        for lot in self._position_lots.get(code, []):
            if lot.remaining > 0 and current_date >= lot.available_date:
                available += lot.remaining

        return available

    def get_total_shares(self, code: str) -> int:
        """
        获取总持仓数量（包括T+1锁定的）

        Args:
            code: 股票代码

        Returns:
            总持仓数量
        """
        return sum(lot.remaining for lot in self._position_lots.get(code, []))

    def get_locked_shares(self, code: str, current_date: Optional[date] = None) -> int:
        """
        获取锁定数量（当日买入，不可卖）

        Args:
            code: 股票代码
            current_date: 当前日期

        Returns:
            锁定数量
        """
        total = self.get_total_shares(code)
        available = self.get_available_shares(code, current_date)
        return total - available

    def deduct_available(
        self,
        code: str,
        shares: int,
        current_date: Optional[date] = None
    ) -> bool:
        """
        扣减可卖数量（卖出时调用）

        使用FIFO（先进先出）原则

        Args:
            code: 股票代码
            shares: 卖出数量
            current_date: 当前日期

        Returns:
            是否成功扣减
        """
        if current_date is None:
            current_date = date.today()

        available = self.get_available_shares(code, current_date)
        if shares > available:
            logger.warning(
                f"可卖数量不足: {code} 需要卖出{shares}股，"
                f"可卖{available}股"
            )
            return False

        # FIFO扣减
        remaining_to_sell = shares
        for lot in self._position_lots.get(code, []):
            if remaining_to_sell <= 0:
                break

            if lot.remaining > 0 and current_date >= lot.available_date:
                deduct = min(lot.remaining, remaining_to_sell)
                lot.remaining -= deduct
                remaining_to_sell -= deduct

                logger.debug(
                    f"FIFO卖出: {code} {deduct}股, "
                    f"批次买入日:{lot.buy_date}, 剩余:{lot.remaining}"
                )

        # 同步更新买入记录
        self._sync_buy_records(code)

        return True

    def _sync_buy_records(self, code: str) -> None:
        """同步买入记录的剩余数量"""
        # 简化处理：重新计算
        total_remaining = sum(
            lot.remaining for lot in self._position_lots.get(code, [])
        )

        # 按比例更新（简化）
        for record in self._buy_records.get(code, []):
            if total_remaining > 0:
                record.remaining_shares = min(record.shares, total_remaining)
                total_remaining -= record.remaining_shares
            else:
                record.remaining_shares = 0

    def can_sell(self, code: str, shares: int, current_date: Optional[date] = None) -> bool:
        """
        检查是否可以卖出

        Args:
            code: 股票代码
            shares: 卖出数量
            current_date: 当前日期

        Returns:
            是否可以卖出
        """
        available = self.get_available_shares(code, current_date)
        return shares <= available

    def get_position_info(self, code: str, current_date: Optional[date] = None) -> dict:
        """
        获取持仓详情

        Args:
            code: 股票代码
            current_date: 当前日期

        Returns:
            持仓信息字典
        """
        if current_date is None:
            current_date = date.today()

        total = self.get_total_shares(code)
        available = self.get_available_shares(code, current_date)
        locked = total - available

        # 计算平均成本
        total_cost = 0
        total_shares = 0
        for lot in self._position_lots.get(code, []):
            if lot.remaining > 0:
                total_cost += lot.cost_price * lot.remaining
                total_shares += lot.remaining

        avg_cost = total_cost / total_shares if total_shares > 0 else 0

        return {
            "code": code,
            "total_shares": total,
            "available_shares": available,
            "locked_shares": locked,
            "average_cost": avg_cost,
            "lots": [
                {
                    "shares": lot.shares,
                    "remaining": lot.remaining,
                    "buy_date": lot.buy_date.isoformat(),
                    "available_date": lot.available_date.isoformat(),
                    "is_available": current_date >= lot.available_date,
                }
                for lot in self._position_lots.get(code, [])
            ]
        }

    def clear(self) -> None:
        """清空所有记录"""
        self._buy_records.clear()
        self._position_lots.clear()
        logger.info("T1Manager 记录已清空")

    def get_all_positions(self, current_date: Optional[date] = None) -> Dict[str, dict]:
        """
        获取所有持仓信息

        Args:
            current_date: 当前日期

        Returns:
            所有持仓信息
        """
        positions = {}
        for code in self._position_lots.keys():
            if self.get_total_shares(code) > 0:
                positions[code] = self.get_position_info(code, current_date)
        return positions


# 工具函数
def is_trading_day(d: date) -> bool:
    """
    判断是否为交易日（简化版）

    注意：实际应使用交易日历
    这里仅跳过周末
    """
    return d.weekday() < 5  # 0-4 = Monday to Friday


def get_next_trading_day(d: date) -> date:
    """
    获取下一交易日（简化版）

    注意：实际应使用交易日历
    """
    next_day = d + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day


# 单例
_t1_manager: Optional[T1Manager] = None


def get_t1_manager() -> T1Manager:
    """获取T1Manager单例"""
    global _t1_manager
    if _t1_manager is None:
        _t1_manager = T1Manager()
    return _t1_manager
