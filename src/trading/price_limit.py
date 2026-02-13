"""
涨跌停限制检查模块

A股涨跌停规则：
- 主板（沪深主板）：±10%
- 创业板（300xxx, 301xxx）：±20%
- 科创板（688xxx）：±20%
- 北交所（8xxxxx, 4xxxxx）：±30%
- ST股票：±5%（优先级最高）

新股特殊规则：
- 主板新股上市前5个交易日：无涨跌幅限制
- 科创板/创业板新股上市前5个交易日：无涨跌幅限制
- 北交所新股上市首日：无涨跌幅限制

使用示例:
    from src.trading.price_limit import PriceLimitChecker

    checker = PriceLimitChecker()

    # 检查是否涨停
    if checker.is_limit_up("600519.SH", 1852.50, 1685.00):
        print("涨停，无法买入")

    # 检查是否跌停
    if checker.is_limit_down("600519.SH", 1517.50, 1685.00):
        print("跌停，无法卖出")

    # 获取有效价格范围
    low, high = checker.get_valid_price_range("600519.SH", 1685.00)

    # 使用数据库获取股票名称进行ST判断
    checker.set_stock_name_provider(lambda code: db.get_stock_name(code))
"""

import re
from typing import Tuple, Optional, Callable, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from loguru import logger


class BoardType(Enum):
    """板块类型"""
    MAIN = "main"           # 主板
    CHINEXT = "chinext"     # 创业板
    STAR = "star"           # 科创板
    BSE = "bse"             # 北交所
    UNKNOWN = "unknown"     # 未知


@dataclass
class PriceLimitInfo:
    """涨跌停信息"""
    code: str
    board_type: BoardType
    is_st: bool             # 是否ST股票
    is_new_stock: bool      # 是否新股（无涨跌停限制）
    limit_up_ratio: float   # 涨停比例
    limit_down_ratio: float # 跌停比例
    limit_up_price: Optional[float]   # 涨停价（新股可能为None）
    limit_down_price: Optional[float] # 跌停价（新股可能为None）
    prev_close: float       # 前收盘价
    stock_name: Optional[str] = None  # 股票名称


class PriceLimitChecker:
    """
    涨跌停限制检查器

    功能：
    1. 识别股票板块类型
    2. 检测ST股票（支持动态更新）
    3. 计算涨跌停价格
    4. 判断是否触及涨跌停
    5. 新股特殊处理
    """

    # 涨跌停比例配置
    LIMIT_RATIOS = {
        BoardType.MAIN: (0.10, -0.10),      # 主板 ±10%
        BoardType.CHINEXT: (0.20, -0.20),   # 创业板 ±20%
        BoardType.STAR: (0.20, -0.20),      # 科创板 ±20%
        BoardType.BSE: (0.30, -0.30),       # 北交所 ±30%
    }

    # ST股票涨跌停比例
    ST_LIMIT_RATIO = 0.05  # ST股票 ±5%

    # ST股票名称前缀模式
    ST_PATTERNS = ['ST', '*ST', 'SST', 'S*ST', 'NST', 'S', 'C']

    # 新股无涨跌停限制天数
    NEW_STOCK_NO_LIMIT_DAYS = {
        BoardType.MAIN: 5,      # 主板前5日
        BoardType.CHINEXT: 5,   # 创业板前5日
        BoardType.STAR: 5,      # 科创板前5日
        BoardType.BSE: 1,       # 北交所首日
    }

    def __init__(self):
        """初始化检查器"""
        # ST股票缓存：{股票代码: 是否ST}
        self._st_cache: Dict[str, bool] = {}

        # 股票名称提供者（可从数据库获取）
        self._name_provider: Optional[Callable[[str], Optional[str]]] = None

        # 上市日期提供者（可从数据库获取）
        self._list_date_provider: Optional[Callable[[str], Optional[str]]] = None

    def set_name_provider(self, provider: Callable[[str], Optional[str]]):
        """
        设置股票名称提供者

        Args:
            provider: 函数，接受股票代码，返回股票名称
        """
        self._name_provider = provider

    def set_list_date_provider(self, provider: Callable[[str], Optional[str]]):
        """
        设置上市日期提供者

        Args:
            provider: 函数，接受股票代码，返回上市日期（YYYYMMDD格式）
        """
        self._list_date_provider = provider

    def get_stock_name(self, code: str) -> Optional[str]:
        """
        获取股票名称

        优先使用名称提供者，否则返回None

        Args:
            code: 股票代码

        Returns:
            股票名称
        """
        if self._name_provider:
            try:
                return self._name_provider(code)
            except Exception as e:
                logger.warning(f"获取股票名称失败 {code}: {e}")
        return None

    def get_list_date(self, code: str) -> Optional[str]:
        """
        获取上市日期

        Args:
            code: 股票代码

        Returns:
            上市日期（YYYYMMDD格式）
        """
        if self._list_date_provider:
            try:
                return self._list_date_provider(code)
            except Exception as e:
                logger.warning(f"获取上市日期失败 {code}: {e}")
        return None

    def update_st_status(self, code: str, is_st: bool):
        """
        更新ST状态缓存

        Args:
            code: 股票代码
            is_st: 是否ST
        """
        self._st_cache[code] = is_st

    def clear_st_cache(self):
        """清空ST缓存"""
        self._st_cache.clear()

    def get_board_type(self, code: str) -> BoardType:
        """
        根据股票代码判断板块类型

        Args:
            code: 股票代码（带后缀，如 600519.SH）

        Returns:
            板块类型
        """
        # 去掉后缀
        pure_code = code.split('.')[0]

        # 科创板：688xxx
        if pure_code.startswith('688'):
            return BoardType.STAR

        # 创业板：300xxx, 301xxx
        if pure_code.startswith('300') or pure_code.startswith('301'):
            return BoardType.CHINEXT

        # 北交所：8xxxxx, 4xxxxx
        if pure_code.startswith('8') or pure_code.startswith('4'):
            if len(pure_code) == 6:
                return BoardType.BSE

        # 主板：600xxx, 601xxx, 603xxx, 000xxx, 001xxx, 002xxx, 003xxx
        if (pure_code.startswith('60') or
            pure_code.startswith('000') or
            pure_code.startswith('001') or
            pure_code.startswith('002') or
            pure_code.startswith('003')):
            return BoardType.MAIN

        return BoardType.UNKNOWN

    def is_st_stock(self, code: str, name: Optional[str] = None) -> bool:
        """
        判断是否为ST股票

        判断逻辑（优先级从高到低）：
        1. 检查缓存
        2. 使用传入的名称判断
        3. 使用名称提供者获取名称判断

        Args:
            code: 股票代码
            name: 股票名称（可选，用于更准确判断）

        Returns:
            是否为ST股票
        """
        # 1. 检查缓存
        if code in self._st_cache:
            return self._st_cache[code]

        # 2. 如果没有传入名称，尝试从提供者获取
        if name is None:
            name = self.get_stock_name(code)

        # 3. 根据名称判断
        if name:
            for pattern in self.ST_PATTERNS:
                # 精确匹配前缀（避免误判如 "STAKE"）
                if name.startswith(pattern):
                    # 特殊处理：单独的 "S" 或 "C" 后面需要跟其他字符
                    if pattern in ('S', 'C'):
                        if len(name) > 1 and (name[1].isdigit() or name[1].isalpha()):
                            continue
                    return True

        # 4. 无法判断，默认不是ST
        return False

    def is_new_stock(
        self,
        code: str,
        check_date: Optional[str] = None,
        list_date: Optional[str] = None
    ) -> bool:
        """
        判断是否为新股（在无涨跌停限制期内）

        Args:
            code: 股票代码
            check_date: 检查日期（YYYYMMDD格式），默认为今天
            list_date: 上市日期（YYYYMMDD格式），可选

        Returns:
            是否为新股
        """
        if check_date is None:
            check_date = datetime.now().strftime('%Y%m%d')

        # 获取上市日期
        if list_date is None:
            list_date = self.get_list_date(code)

        if list_date is None:
            return False

        try:
            list_dt = datetime.strptime(list_date, '%Y%m%d')
            check_dt = datetime.strptime(check_date, '%Y%m%d')

            # 计算上市天数
            days_since_list = (check_dt - list_dt).days

            # 获取板块类型
            board_type = self.get_board_type(code)
            no_limit_days = self.NEW_STOCK_NO_LIMIT_DAYS.get(board_type, 0)

            return 0 <= days_since_list < no_limit_days

        except ValueError as e:
            logger.warning(f"日期解析失败: {e}")
            return False

    def get_limit_ratio(
        self,
        code: str,
        name: Optional[str] = None,
        check_date: Optional[str] = None,
        list_date: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        获取涨跌停比例

        Args:
            code: 股票代码
            name: 股票名称（可选）
            check_date: 检查日期（YYYYMMDD格式），默认为今天
            list_date: 上市日期（YYYYMMDD格式），可选

        Returns:
            (涨停比例, 跌停比例)，新股返回 (None, None) 表示无限制
        """
        # 检查是否为新股
        if self.is_new_stock(code, check_date, list_date):
            return (None, None)  # 无涨跌停限制

        # ST股票优先
        if self.is_st_stock(code, name):
            return (self.ST_LIMIT_RATIO, -self.ST_LIMIT_RATIO)

        # 根据板块类型
        board_type = self.get_board_type(code)
        return self.LIMIT_RATIOS.get(board_type, (0.10, -0.10))  # 默认主板

    def get_valid_price_range(
        self,
        code: str,
        prev_close: float,
        name: Optional[str] = None,
        check_date: Optional[str] = None,
        list_date: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        获取有效价格范围（涨跌停价格）

        Args:
            code: 股票代码
            prev_close: 前收盘价
            name: 股票名称（可选）
            check_date: 检查日期（YYYYMMDD格式）
            list_date: 上市日期（YYYYMMDD格式）

        Returns:
            (跌停价, 涨停价)，新股返回 (None, None)
        """
        limit_up_ratio, limit_down_ratio = self.get_limit_ratio(
            code, name, check_date, list_date
        )

        # 新股无涨跌停限制
        if limit_up_ratio is None or limit_down_ratio is None:
            return (None, None)

        # 计算涨跌停价格（需要精确到分）
        limit_up = round(prev_close * (1 + limit_up_ratio), 2)
        limit_down = round(prev_close * (1 + limit_down_ratio), 2)

        return (limit_down, limit_up)

    def is_limit_up(
        self,
        code: str,
        price: float,
        prev_close: float,
        name: Optional[str] = None,
        tolerance: float = 0.001,
        check_date: Optional[str] = None,
        list_date: Optional[str] = None
    ) -> bool:
        """
        判断是否涨停

        Args:
            code: 股票代码
            price: 当前价格
            prev_close: 前收盘价
            name: 股票名称
            tolerance: 容差（用于浮点数比较）
            check_date: 检查日期
            list_date: 上市日期

        Returns:
            是否涨停
        """
        _, limit_up = self.get_valid_price_range(
            code, prev_close, name, check_date, list_date
        )

        # 新股无涨跌停
        if limit_up is None:
            return False

        return price >= limit_up - tolerance

    def is_limit_down(
        self,
        code: str,
        price: float,
        prev_close: float,
        name: Optional[str] = None,
        tolerance: float = 0.001,
        check_date: Optional[str] = None,
        list_date: Optional[str] = None
    ) -> bool:
        """
        判断是否跌停

        Args:
            code: 股票代码
            price: 当前价格
            prev_close: 前收盘价
            name: 股票名称
            tolerance: 容差
            check_date: 检查日期
            list_date: 上市日期

        Returns:
            是否跌停
        """
        limit_down, _ = self.get_valid_price_range(
            code, prev_close, name, check_date, list_date
        )

        # 新股无涨跌停
        if limit_down is None:
            return False

        return price <= limit_down + tolerance

    def can_buy(
        self,
        code: str,
        price: float,
        prev_close: float,
        name: Optional[str] = None,
        check_date: Optional[str] = None,
        list_date: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        检查是否可以买入

        涨停时无法买入（排板除外）

        Args:
            code: 股票代码
            price: 委托价格
            prev_close: 前收盘价
            name: 股票名称
            check_date: 检查日期
            list_date: 上市日期

        Returns:
            (是否可以买入, 原因)
        """
        # 检查是否为新股
        if self.is_new_stock(code, check_date, list_date):
            return (True, "新股无涨跌停限制")

        if self.is_limit_up(code, price, prev_close, name, 0.001, check_date, list_date):
            return (False, f"涨停价{price}，无法买入")
        return (True, "可以买入")

    def can_sell(
        self,
        code: str,
        price: float,
        prev_close: float,
        name: Optional[str] = None,
        check_date: Optional[str] = None,
        list_date: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        检查是否可以卖出

        跌停时无法卖出（排板除外）

        Args:
            code: 股票代码
            price: 委托价格
            prev_close: 前收盘价
            name: 股票名称
            check_date: 检查日期
            list_date: 上市日期

        Returns:
            (是否可以卖出, 原因)
        """
        # 检查是否为新股
        if self.is_new_stock(code, check_date, list_date):
            return (True, "新股无涨跌停限制")

        if self.is_limit_down(code, price, prev_close, name, 0.001, check_date, list_date):
            return (False, f"跌停价{price}，无法卖出")
        return (True, "可以卖出")

    def get_price_limit_info(
        self,
        code: str,
        prev_close: float,
        name: Optional[str] = None,
        check_date: Optional[str] = None,
        list_date: Optional[str] = None
    ) -> PriceLimitInfo:
        """
        获取完整的涨跌停信息

        Args:
            code: 股票代码
            prev_close: 前收盘价
            name: 股票名称
            check_date: 检查日期
            list_date: 上市日期

        Returns:
            涨跌停信息
        """
        # 如果没有传入名称，尝试获取
        if name is None:
            name = self.get_stock_name(code)

        board_type = self.get_board_type(code)
        is_st = self.is_st_stock(code, name)
        is_new = self.is_new_stock(code, check_date, list_date)
        limit_up_ratio, limit_down_ratio = self.get_limit_ratio(code, name, check_date, list_date)
        limit_down, limit_up = self.get_valid_price_range(code, prev_close, name, check_date, list_date)

        return PriceLimitInfo(
            code=code,
            board_type=board_type,
            is_st=is_st,
            is_new_stock=is_new,
            limit_up_ratio=limit_up_ratio if limit_up_ratio is not None else 0.0,
            limit_down_ratio=limit_down_ratio if limit_down_ratio is not None else 0.0,
            limit_up_price=limit_up,
            limit_down_price=limit_down,
            prev_close=prev_close,
            stock_name=name
        )

    def adjust_price_to_limit(
        self,
        code: str,
        price: float,
        prev_close: float,
        name: Optional[str] = None,
        check_date: Optional[str] = None,
        list_date: Optional[str] = None
    ) -> float:
        """
        将价格调整到涨跌停范围内

        Args:
            code: 股票代码
            price: 原始价格
            prev_close: 前收盘价
            name: 股票名称
            check_date: 检查日期
            list_date: 上市日期

        Returns:
            调整后的价格（新股返回原价格）
        """
        limit_down, limit_up = self.get_valid_price_range(
            code, prev_close, name, check_date, list_date
        )

        # 新股无限制
        if limit_down is None or limit_up is None:
            return price

        if price > limit_up:
            logger.warning(f"价格{price}超过涨停价{limit_up}，已调整")
            return limit_up
        elif price < limit_down:
            logger.warning(f"价格{price}低于跌停价{limit_down}，已调整")
            return limit_down

        return price


# 单例
_price_limit_checker: Optional[PriceLimitChecker] = None


def get_price_limit_checker() -> PriceLimitChecker:
    """获取涨跌停检查器单例"""
    global _price_limit_checker
    if _price_limit_checker is None:
        _price_limit_checker = PriceLimitChecker()
    return _price_limit_checker
