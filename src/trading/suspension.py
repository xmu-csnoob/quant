"""
停牌处理模块

A股停牌规则：
1. 停牌期间无法交易
2. 复牌后以停牌前最后收盘价为基准计算涨跌停
3. 长期停牌后复牌可能需要特殊处理

停牌原因：
- 重大事项公告
- 资产重组
- 异常波动
- 其他监管要求

使用示例:
    from src.trading.suspension import SuspensionManager

    manager = SuspensionManager()

    # 添加停牌信息
    manager.add_suspension("600519.SH", "20240115", "重大事项")

    # 检查是否停牌
    if manager.is_suspended("600519.SH", "20240115"):
        print("停牌中，无法交易")

    # 获取复牌后的前收盘价
    prev_close = manager.get_resume_prev_close("600519.SH", "20240120")
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger


class SuspensionReason(Enum):
    """停牌原因"""
    MAJOR_EVENT = "major_event"         # 重大事项
    REORGANIZATION = "reorganization"   # 资产重组
    ABNORMAL_MOVE = "abnormal_move"     # 异常波动
    DELISTING_RISK = "delisting_risk"   # 退市风险
    REGULATORY = "regulatory"           # 监管要求
    OTHER = "other"                     # 其他


@dataclass
class SuspensionRecord:
    """停牌记录"""
    code: str                       # 股票代码
    start_date: str                 # 停牌开始日期 (YYYYMMDD)
    end_date: Optional[str]         # 复牌日期 (YYYYMMDD)，None表示持续停牌
    reason: SuspensionReason        # 停牌原因
    prev_close: Optional[float]     # 停牌前最后收盘价
    description: str = ""           # 描述

    @property
    def is_active(self) -> bool:
        """是否仍在停牌中"""
        return self.end_date is None

    def covers_date(self, date: str) -> bool:
        """
        检查某日期是否在停牌期间

        Args:
            date: 日期 (YYYYMMDD)

        Returns:
            是否在停牌期间
        """
        if date < self.start_date:
            return False
        if self.end_date is None:
            return date >= self.start_date
        return self.start_date <= date <= self.end_date


class SuspensionManager:
    """
    停牌管理器

    功能：
    1. 管理股票停牌/复牌状态
    2. 检查某日期是否可交易
    3. 获取复牌后的前收盘价
    4. 支持从API同步停牌信息
    """

    def __init__(self):
        """初始化"""
        # 停牌记录 {股票代码: [停牌记录]}
        self._suspensions: Dict[str, List[SuspensionRecord]] = {}
        # 停牌日期缓存 {股票代码: {日期: 是否停牌}}
        self._cache: Dict[str, Dict[str, bool]] = {}

    def add_suspension(
        self,
        code: str,
        start_date: str,
        reason: SuspensionReason = SuspensionReason.OTHER,
        prev_close: Optional[float] = None,
        end_date: Optional[str] = None,
        description: str = "",
    ) -> SuspensionRecord:
        """
        添加停牌记录

        Args:
            code: 股票代码
            start_date: 停牌开始日期
            reason: 停牌原因
            prev_close: 停牌前最后收盘价
            end_date: 复牌日期（None表示持续停牌）
            description: 描述

        Returns:
            停牌记录
        """
        record = SuspensionRecord(
            code=code,
            start_date=start_date,
            end_date=end_date,
            reason=reason,
            prev_close=prev_close,
            description=description,
        )

        if code not in self._suspensions:
            self._suspensions[code] = []
        self._suspensions[code].append(record)

        # 清除缓存
        if code in self._cache:
            del self._cache[code]

        logger.info(f"添加停牌记录: {code}, {start_date} - {end_date or '持续'}, {reason.value}")
        return record

    def resume(
        self,
        code: str,
        resume_date: str,
        prev_close: Optional[float] = None,
    ) -> bool:
        """
        设置复牌

        Args:
            code: 股票代码
            resume_date: 复牌日期
            prev_close: 复牌前收盘价（可选，用于更新）

        Returns:
            是否成功复牌
        """
        if code not in self._suspensions:
            logger.warning(f"未找到 {code} 的停牌记录")
            return False

        # 找到最近的活跃停牌记录
        active_records = [r for r in self._suspensions[code] if r.is_active]
        if not active_records:
            logger.warning(f"{code} 没有活跃的停牌记录")
            return False

        # 更新最近的停牌记录
        record = max(active_records, key=lambda r: r.start_date)
        record.end_date = resume_date
        if prev_close is not None:
            record.prev_close = prev_close

        # 清除缓存
        if code in self._cache:
            del self._cache[code]

        logger.info(f"设置复牌: {code}, 复牌日期={resume_date}")
        return True

    def is_suspended(self, code: str, date: str) -> bool:
        """
        检查某日期是否停牌

        Args:
            code: 股票代码
            date: 日期 (YYYYMMDD)

        Returns:
            是否停牌
        """
        # 检查缓存
        if code in self._cache and date in self._cache[code]:
            return self._cache[code][date]

        # 查找停牌记录
        result = False
        if code in self._suspensions:
            for record in self._suspensions[code]:
                if record.covers_date(date):
                    result = True
                    break

        # 更新缓存
        if code not in self._cache:
            self._cache[code] = {}
        self._cache[code][date] = result

        return result

    def can_trade(self, code: str, date: str) -> tuple[bool, str]:
        """
        检查是否可以交易

        Args:
            code: 股票代码
            date: 日期 (YYYYMMDD)

        Returns:
            (是否可以交易, 原因)
        """
        if self.is_suspended(code, date):
            return (False, f"股票 {code} 在 {date} 停牌中")
        return (True, "可以交易")

    def get_prev_close_for_date(
        self,
        code: str,
        date: str,
        default_prev_close: Optional[float] = None,
    ) -> Optional[float]:
        """
        获取某日期的前收盘价

        对于复牌后的第一个交易日，前收盘价是停牌前的最后收盘价

        Args:
            code: 股票代码
            date: 日期 (YYYYMMDD)
            default_prev_close: 默认前收盘价

        Returns:
            前收盘价
        """
        if code not in self._suspensions:
            return default_prev_close

        # 查找该日期之前的停牌记录
        for record in self._suspensions[code]:
            if record.end_date == date and record.prev_close is not None:
                # 复牌日，返回停牌前收盘价
                return record.prev_close

        return default_prev_close

    def get_suspension_record(self, code: str, date: str) -> Optional[SuspensionRecord]:
        """
        获取某日期的停牌记录

        Args:
            code: 股票代码
            date: 日期 (YYYYMMDD)

        Returns:
            停牌记录（如果停牌），否则None
        """
        if code not in self._suspensions:
            return None

        for record in self._suspensions[code]:
            if record.covers_date(date):
                return record

        return None

    def get_active_suspensions(self, date: str) -> List[SuspensionRecord]:
        """
        获取某日期所有停牌中的股票

        Args:
            date: 日期 (YYYYMMDD)

        Returns:
            停牌记录列表
        """
        result = []
        for records in self._suspensions.values():
            for record in records:
                if record.covers_date(date):
                    result.append(record)
        return result

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

    def remove_suspension(self, code: str, start_date: str) -> bool:
        """
        移除停牌记录

        Args:
            code: 股票代码
            start_date: 停牌开始日期

        Returns:
            是否成功移除
        """
        if code not in self._suspensions:
            return False

        for i, record in enumerate(self._suspensions[code]):
            if record.start_date == start_date:
                del self._suspensions[code][i]
                if code in self._cache:
                    del self._cache[code]
                logger.info(f"移除停牌记录: {code}, {start_date}")
                return True

        return False


# 单例
_suspension_manager: Optional[SuspensionManager] = None


def get_suspension_manager() -> SuspensionManager:
    """获取停牌管理器单例"""
    global _suspension_manager
    if _suspension_manager is None:
        _suspension_manager = SuspensionManager()
    return _suspension_manager
