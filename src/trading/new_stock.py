"""
新股管理模块

A股新股交易规则：
1. 主板新股上市前5个交易日无涨跌幅限制
2. 科创板/创业板新股上市前5个交易日无涨跌幅限制
3. 北交所新股上市首日无涨跌幅限制

第6个交易日开始执行正常涨跌停规则

使用示例:
    from src.trading.new_stock import NewStockManager

    manager = NewStockManager()

    # 添加新股
    manager.add_new_stock("601328.SH", "20240115", "主板新股")

    # 检查是否为新股（无涨跌停限制）
    if manager.is_in_no_limit_period("601328.SH", "20240117"):
        print("新股上市初期，无涨跌停限制")
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Callable
from datetime import datetime
from enum import Enum
from loguru import logger

from src.trading.price_limit import BoardType


class NewStockType(Enum):
    """新股类型"""
    MAIN = "main"           # 主板
    CHINEXT = "chinext"     # 创业板
    STAR = "star"           # 科创板
    BSE = "bse"             # 北交所


# 新股无涨跌停限制天数配置
NEW_STOCK_NO_LIMIT_DAYS = {
    NewStockType.MAIN: 5,       # 主板前5日
    NewStockType.CHINEXT: 5,    # 创业板前5日
    NewStockType.STAR: 5,       # 科创板前5日
    NewStockType.BSE: 1,        # 北交所首日
}


@dataclass
class NewStockRecord:
    """新股记录"""
    code: str                   # 股票代码
    list_date: str              # 上市日期 (YYYYMMDD)
    stock_type: NewStockType    # 新股类型
    name: str = ""              # 股票名称
    issue_price: Optional[float] = None  # 发行价

    @property
    def no_limit_days(self) -> int:
        """无涨跌停限制天数"""
        return NEW_STOCK_NO_LIMIT_DAYS.get(self.stock_type, 5)

    def get_no_limit_end_date(self) -> str:
        """获取涨跌停限制开始日期（无限制期结束后的第一天）"""
        list_dt = datetime.strptime(self.list_date, "%Y%m%d")
        # 计算N个交易日后的日期（简化处理，按自然日+缓冲）
        # 实际应该用交易日历
        from datetime import timedelta
        end_dt = list_dt + timedelta(days=self.no_limit_days + 2)  # 加2天缓冲
        return end_dt.strftime("%Y%m%d")


class NewStockManager:
    """
    新股管理器

    功能：
    1. 管理新股上市信息
    2. 判断是否在无涨跌停限制期
    3. 计算剩余无限制天数
    4. 支持从API同步新股信息
    """

    def __init__(self):
        """初始化"""
        # 新股记录 {股票代码: 新股记录}
        self._new_stocks: Dict[str, NewStockRecord] = {}
        # 股票类型提供者
        self._type_provider: Optional[Callable[[str], BoardType]] = None

    def set_type_provider(self, provider: Callable[[str], BoardType]):
        """
        设置股票类型提供者

        Args:
            provider: 函数，接受股票代码，返回板块类型
        """
        self._type_provider = provider

    def _get_stock_type(self, code: str) -> NewStockType:
        """
        获取股票类型

        Args:
            code: 股票代码

        Returns:
            新股类型
        """
        # 如果有提供者，使用提供者
        if self._type_provider:
            board_type = self._type_provider(code)
            type_map = {
                BoardType.MAIN: NewStockType.MAIN,
                BoardType.CHINEXT: NewStockType.CHINEXT,
                BoardType.STAR: NewStockType.STAR,
                BoardType.BSE: NewStockType.BSE,
            }
            return type_map.get(board_type, NewStockType.MAIN)

        # 根据代码判断
        pure_code = code.split('.')[0]

        if pure_code.startswith('688'):
            return NewStockType.STAR
        if pure_code.startswith(('300', '301')):
            return NewStockType.CHINEXT
        if pure_code.startswith(('8', '4')) and len(pure_code) == 6:
            return NewStockType.BSE
        return NewStockType.MAIN

    def add_new_stock(
        self,
        code: str,
        list_date: str,
        name: str = "",
        issue_price: Optional[float] = None,
        stock_type: Optional[NewStockType] = None,
    ) -> NewStockRecord:
        """
        添加新股记录

        Args:
            code: 股票代码
            list_date: 上市日期 (YYYYMMDD)
            name: 股票名称
            issue_price: 发行价
            stock_type: 新股类型（None自动判断）

        Returns:
            新股记录
        """
        if stock_type is None:
            stock_type = self._get_stock_type(code)

        record = NewStockRecord(
            code=code,
            list_date=list_date,
            stock_type=stock_type,
            name=name,
            issue_price=issue_price,
        )

        self._new_stocks[code] = record
        logger.info(
            f"添加新股记录: {code}, 上市日期={list_date}, "
            f"类型={stock_type.value}, 无限制期={record.no_limit_days}天"
        )
        return record

    def get_new_stock(self, code: str) -> Optional[NewStockRecord]:
        """
        获取新股记录

        Args:
            code: 股票代码

        Returns:
            新股记录（如果不是新股返回None）
        """
        return self._new_stocks.get(code)

    def is_new_stock(self, code: str) -> bool:
        """
        检查是否为新股（有记录）

        Args:
            code: 股票代码

        Returns:
            是否为新股
        """
        return code in self._new_stocks

    def is_in_no_limit_period(
        self,
        code: str,
        check_date: str,
        trading_days: Optional[List[str]] = None,
    ) -> bool:
        """
        检查是否在无涨跌停限制期

        Args:
            code: 股票代码
            check_date: 检查日期 (YYYYMMDD)
            trading_days: 交易日列表（可选，用于精确计算）

        Returns:
            是否在无限制期
        """
        record = self._new_stocks.get(code)
        if record is None:
            return False

        list_dt = datetime.strptime(record.list_date, "%Y%m%d")
        check_dt = datetime.strptime(check_date, "%Y%m%d")

        # 上市前不是新股
        if check_dt < list_dt:
            return False

        if trading_days:
            # 使用交易日历精确计算
            trading_days_between = [
                d for d in trading_days
                if record.list_date <= d <= check_date
            ]
            days_since_list = len(trading_days_between) - 1  # 减去上市日本身
        else:
            # 简化计算：使用自然日
            days_since_list = (check_dt - list_dt).days

        return days_since_list < record.no_limit_days

    def get_days_remaining(
        self,
        code: str,
        check_date: str,
        trading_days: Optional[List[str]] = None,
    ) -> int:
        """
        获取剩余无涨跌停限制天数

        Args:
            code: 股票代码
            check_date: 检查日期 (YYYYMMDD)
            trading_days: 交易日列表

        Returns:
            剩余天数（负数表示已过限制期）
        """
        record = self._new_stocks.get(code)
        if record is None:
            return -1  # 不是新股

        list_dt = datetime.strptime(record.list_date, "%Y%m%d")
        check_dt = datetime.strptime(check_date, "%Y%m%d")

        if check_dt < list_dt:
            return record.no_limit_days  # 还未上市

        if trading_days:
            trading_days_between = [
                d for d in trading_days
                if record.list_date <= d <= check_date
            ]
            days_since_list = len(trading_days_between) - 1
        else:
            days_since_list = (check_dt - list_dt).days

        return max(0, record.no_limit_days - days_since_list - 1)

    def get_recent_new_stocks(
        self,
        check_date: str,
        days: int = 30,
    ) -> List[NewStockRecord]:
        """
        获取最近N天内上市的新股

        Args:
            check_date: 检查日期 (YYYYMMDD)
            days: 天数

        Returns:
            新股记录列表
        """
        check_dt = datetime.strptime(check_date, "%Y%m%d")
        result = []

        for record in self._new_stocks.values():
            list_dt = datetime.strptime(record.list_date, "%Y%m%d")
            delta = (check_dt - list_dt).days
            if 0 <= delta <= days:
                result.append(record)

        return sorted(result, key=lambda r: r.list_date, reverse=True)

    def remove_new_stock(self, code: str) -> bool:
        """
        移除新股记录（当股票不再是新股时）

        Args:
            code: 股票代码

        Returns:
            是否成功移除
        """
        if code in self._new_stocks:
            del self._new_stocks[code]
            logger.info(f"移除新股记录: {code}")
            return True
        return False

    def cleanup_old_records(self, check_date: str, keep_days: int = 30):
        """
        清理过期的新股记录

        Args:
            check_date: 检查日期 (YYYYMMDD)
            keep_days: 保留天数（超过此天数的新股记录将被删除）
        """
        check_dt = datetime.strptime(check_date, "%Y%m%d")
        to_remove = []

        for code, record in self._new_stocks.items():
            list_dt = datetime.strptime(record.list_date, "%Y%m%d")
            if (check_dt - list_dt).days > keep_days:
                to_remove.append(code)

        for code in to_remove:
            self.remove_new_stock(code)

        if to_remove:
            logger.info(f"清理了 {len(to_remove)} 条过期新股记录")


# 单例
_new_stock_manager: Optional[NewStockManager] = None


def get_new_stock_manager() -> NewStockManager:
    """获取新股管理器单例"""
    global _new_stock_manager
    if _new_stock_manager is None:
        _new_stock_manager = NewStockManager()
    return _new_stock_manager
