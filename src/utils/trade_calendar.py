"""
交易日历工具

提供 A 股交易日历功能：
- 判断是否为交易日
- 获取下一/上一交易日
- T+1 可卖日期计算
- 从 AkShare/Tushare API 动态获取交易日历
- 本地缓存以减少 API 调用
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Optional, Set
import json
import hashlib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from config import settings
from loguru import logger


class TradeCalendar:
    """
    交易日历管理

    支持多种数据源：
    1. AkShare API（免费，无需token）
    2. Tushare API（需要token，更准确）
    3. 本地缓存
    4. 数据库推断（备选）

    Example:
        >>> calendar = TradeCalendar()
        >>> calendar.is_trading_day('20240115')
        True
        >>> calendar.get_t1_sell_date('20240115')
        '20240116'
    """

    def __init__(self, prefer_source: str = "auto"):
        """
        初始化交易日历

        Args:
            prefer_source: 首选数据源，可选 'akshare', 'tushare', 'local', 'auto'
                          'auto' 会自动选择可用的数据源
        """
        self.storage = SQLiteStorage()
        self.cache_file = settings.DATA_DIR / "trade_calendar_cache.json"
        self._holidays_cache: Optional[Set[str]] = None
        self._trading_days_cache: Optional[Set[str]] = None
        self._cache_data: Optional[dict] = None
        self.prefer_source = prefer_source

    def load_holidays(self) -> Set[str]:
        """
        加载节假日缓存

        Returns:
            set: 节假日日期集合 (YYYYMMDD格式字符串)
        """
        if self._holidays_cache is not None:
            return self._holidays_cache

        # 尝试从缓存加载
        cache_data = self._load_cache()
        if cache_data and 'holidays' in cache_data:
            self._holidays_cache = set(cache_data['holidays'])
            return self._holidays_cache

        # 尝试从 API 获取
        self._holidays_cache = self._fetch_holidays_from_api()

        if not self._holidays_cache:
            # 从数据库推断非交易日
            self._holidays_cache = self._infer_holidays_from_db()

        # 保存缓存
        self._save_cache()

        return self._holidays_cache

    def load_trading_days(self) -> Set[str]:
        """
        加载交易日集合

        Returns:
            set: 交易日集合 (YYYYMMDD格式字符串)
        """
        if self._trading_days_cache is not None:
            return self._trading_days_cache

        # 尝试从缓存加载
        cache_data = self._load_cache()
        if cache_data and 'trading_days' in cache_data:
            self._trading_days_cache = set(cache_data['trading_days'])
            return self._trading_days_cache

        # 尝试从 API 获取
        self._trading_days_cache = self._fetch_trading_days_from_api()

        if not self._trading_days_cache:
            # 从数据库推断交易日
            self._trading_days_cache = self._infer_trading_days_from_db()

        # 保存缓存
        self._save_cache()

        return self._trading_days_cache

    def _load_cache(self) -> Optional[dict]:
        """加载缓存数据"""
        if self._cache_data is not None:
            return self._cache_data

        if not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self._cache_data = json.load(f)
                return self._cache_data
        except Exception as e:
            logger.warning(f"加载交易日历缓存失败: {e}")
            return None

    def _save_cache(self):
        """保存缓存"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            cache_data = {
                'holidays': list(self._holidays_cache or []),
                'trading_days': list(self._trading_days_cache or []),
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'api'
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            self._cache_data = cache_data
            logger.info(f"交易日历缓存已保存到 {self.cache_file}")
        except Exception as e:
            logger.warning(f"保存交易日历缓存失败: {e}")

    def _fetch_holidays_from_api(self) -> Set[str]:
        """从 API 获取节假日"""
        holidays = set()

        # 根据 prefer_source 选择数据源
        if self.prefer_source in ('auto', 'akshare'):
            holidays = self._fetch_holidays_from_akshare()
            if holidays:
                logger.info(f"从 AkShare 获取到 {len(holidays)} 个节假日")
                return holidays

        if self.prefer_source in ('auto', 'tushare'):
            holidays = self._fetch_holidays_from_tushare()
            if holidays:
                logger.info(f"从 Tushare 获取到 {len(holidays)} 个节假日")
                return holidays

        return holidays

    def _fetch_trading_days_from_api(self) -> Set[str]:
        """从 API 获取交易日"""
        trading_days = set()

        if self.prefer_source in ('auto', 'akshare'):
            trading_days = self._fetch_trading_days_from_akshare()
            if trading_days:
                logger.info(f"从 AkShare 获取到 {len(trading_days)} 个交易日")
                return trading_days

        if self.prefer_source in ('auto', 'tushare'):
            trading_days = self._fetch_trading_days_from_tushare()
            if trading_days:
                logger.info(f"从 Tushare 获取到 {len(trading_days)} 个交易日")
                return trading_days

        return trading_days

    def _fetch_holidays_from_akshare(self) -> Set[str]:
        """从 AkShare 获取节假日"""
        holidays = set()
        try:
            import akshare as ak

            # 获取交易日历
            df = ak.tool_trade_date_hist_sina()

            # 转换为日期集合
            all_dates = set(df['trade_date'].astype(str).tolist())

            # 推断节假日：工作日但非交易日
            current_year = datetime.now().year
            for year in range(current_year - 1, current_year + 2):
                start = datetime(year, 1, 1)
                end = datetime(year, 12, 31)
                current = start

                while current <= end:
                    if current.weekday() < 5:  # 工作日
                        date_str = current.strftime('%Y%m%d')
                        if date_str not in all_dates:
                            holidays.add(date_str)
                    current += timedelta(days=1)

        except ImportError:
            logger.warning("AkShare 未安装，跳过")
        except Exception as e:
            logger.warning(f"从 AkShare 获取节假日失败: {e}")

        return holidays

    def _fetch_trading_days_from_akshare(self) -> Set[str]:
        """从 AkShare 获取交易日"""
        trading_days = set()
        try:
            import akshare as ak

            df = ak.tool_trade_date_hist_sina()
            trading_days = set(df['trade_date'].astype(str).tolist())

        except ImportError:
            logger.warning("AkShare 未安装，跳过")
        except Exception as e:
            logger.warning(f"从 AkShare 获取交易日失败: {e}")

        return trading_days

    def _fetch_holidays_from_tushare(self) -> Set[str]:
        """从 Tushare 获取节假日"""
        holidays = set()
        try:
            import tushare as ts
            import os

            token = os.environ.get('TUSHARE_TOKEN')
            if not token:
                return holidays

            pro = ts.pro_api(token)
            df = pro.trade_cal(exchange='SSE', is_open='0')

            if df is not None and not df.empty:
                holidays = set(df['cal_date'].astype(str).tolist())

        except ImportError:
            logger.warning("Tushare 未安装，跳过")
        except Exception as e:
            logger.warning(f"从 Tushare 获取节假日失败: {e}")

        return holidays

    def _fetch_trading_days_from_tushare(self) -> Set[str]:
        """从 Tushare 获取交易日"""
        trading_days = set()
        try:
            import tushare as ts
            import os

            token = os.environ.get('TUSHARE_TOKEN')
            if not token:
                return trading_days

            pro = ts.pro_api(token)
            df = pro.trade_cal(exchange='SSE', is_open='1')

            if df is not None and not df.empty:
                trading_days = set(df['cal_date'].astype(str).tolist())

        except ImportError:
            logger.warning("Tushare 未安装，跳过")
        except Exception as e:
            logger.warning(f"从 Tushare 获取交易日失败: {e}")

        return trading_days

    def _infer_holidays_from_db(self) -> Set[str]:
        """
        从数据库推断非交易日

        通过检查某个股票的数据缺口来推断节假日
        """
        holidays = set()

        try:
            stocks = self.storage.get_all_stocks()
            if not stocks:
                return holidays

            ts_code = stocks[0]

            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')

            df = self.storage.get_daily_prices(ts_code, start_date, end_date)

            if df is None or df.empty:
                return holidays

            trading_dates = set(df['trade_date'].astype(str).tolist())

            current = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')

            while current <= end:
                if current.weekday() < 5:
                    date_str = current.strftime('%Y%m%d')
                    if date_str not in trading_dates:
                        holidays.add(date_str)
                current += timedelta(days=1)

            logger.info(f"从数据库推断出 {len(holidays)} 个节假日")

        except Exception as e:
            logger.warning(f"从数据库推断节假日失败: {e}")

        return holidays

    def _infer_trading_days_from_db(self) -> Set[str]:
        """从数据库推断交易日"""
        trading_days = set()

        try:
            stocks = self.storage.get_all_stocks()
            if not stocks:
                return trading_days

            ts_code = stocks[0]

            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')

            df = self.storage.get_daily_prices(ts_code, start_date, end_date)

            if df is not None and not df.empty:
                trading_days = set(df['trade_date'].astype(str).tolist())
                logger.info(f"从数据库推断出 {len(trading_days)} 个交易日")

        except Exception as e:
            logger.warning(f"从数据库推断交易日失败: {e}")

        return trading_days

    def is_trading_day(self, check_date: str = None) -> bool:
        """
        判断是否为交易日

        Args:
            check_date: 检查日期 (YYYYMMDD格式)，默认为今天

        Returns:
            bool: True表示是交易日
        """
        if check_date is None:
            check_date = datetime.now().strftime('%Y%m%d')

        try:
            dt = datetime.strptime(check_date, '%Y%m%d')
        except:
            return False

        # 检查是否为周末
        if dt.weekday() >= 5:  # 5=周六, 6=周日
            return False

        # 检查是否为节假日
        holidays = self.load_holidays()
        if check_date in holidays:
            return False

        return True

    def get_t1_sell_date(self, buy_date: str) -> Optional[str]:
        """
        获取 T+1 可卖日期

        A 股实行 T+1 交易制度，今日买入的股票需要等到下一个交易日才能卖出。

        Args:
            buy_date: 买入日期 (YYYYMMDD格式)

        Returns:
            str: 可卖日期 (YYYYMMDD格式)，如果买入日期无效则返回 None
        """
        try:
            dt = datetime.strptime(buy_date, '%Y%m%d')
        except ValueError:
            logger.warning(f"无效的日期格式: {buy_date}")
            return None

        # 从买入日期的下一天开始查找
        for i in range(1, 15):  # 最多查找15天（考虑长假）
            candidate = (dt + timedelta(days=i)).strftime('%Y%m%d')
            if self.is_trading_day(candidate):
                return candidate

        logger.warning(f"无法找到买入日期 {buy_date} 之后的交易日")
        return None

    def get_prev_trading_day(self, check_date: str = None) -> Optional[str]:
        """
        获取指定日期的前一个交易日

        Args:
            check_date: 检查日期 (YYYYMMDD格式)，默认为今天

        Returns:
            str: 前一个交易日 (YYYYMMDD格式)
        """
        if check_date is None:
            check_date = datetime.now().strftime('%Y%m%d')

        try:
            dt = datetime.strptime(check_date, '%Y%m%d')
        except ValueError:
            return None

        # 向前查找交易日
        for i in range(1, 15):
            candidate = (dt - timedelta(days=i)).strftime('%Y%m%d')
            if self.is_trading_day(candidate):
                return candidate

        return None

    def get_trading_days_between(
        self, start_date: str, end_date: str
    ) -> List[str]:
        """
        获取两个日期之间的所有交易日

        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)

        Returns:
            list: 交易日列表
        """
        try:
            start = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')
        except ValueError:
            return []

        if start > end:
            return []

        trading_days = []
        current = start

        while current <= end:
            date_str = current.strftime('%Y%m%d')
            if self.is_trading_day(date_str):
                trading_days.append(date_str)
            current += timedelta(days=1)

        return trading_days

    def count_trading_days_between(
        self, start_date: str, end_date: str
    ) -> int:
        """
        计算两个日期之间的交易日数量

        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)

        Returns:
            int: 交易日数量
        """
        return len(self.get_trading_days_between(start_date, end_date))

    def get_nth_trading_day(self, start_date: str, n: int) -> Optional[str]:
        """
        获取从指定日期开始的第 n 个交易日

        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            n: 第几个交易日（正数向后，负数向前）

        Returns:
            str: 交易日 (YYYYMMDD格式)
        """
        try:
            dt = datetime.strptime(start_date, '%Y%m%d')
        except ValueError:
            return None

        if n > 0:
            # 向后查找
            count = 0
            for i in range(1, 365):  # 最多查找一年
                candidate = (dt + timedelta(days=i)).strftime('%Y%m%d')
                if self.is_trading_day(candidate):
                    count += 1
                    if count == n:
                        return candidate
        elif n < 0:
            # 向前查找
            count = 0
            for i in range(1, 365):
                candidate = (dt - timedelta(days=i)).strftime('%Y%m%d')
                if self.is_trading_day(candidate):
                    count -= 1
                    if count == n:
                        return candidate

        return start_date if n == 0 else None

    def is_same_or_later(self, date1: str, date2: str) -> bool:
        """
        判断 date1 是否晚于或等于 date2（仅考虑交易日）

        Args:
            date1: 日期1 (YYYYMMDD格式)
            date2: 日期2 (YYYYMMDD格式)

        Returns:
            bool: date1 >= date2（在交易日维度上）
        """
        return date1 >= date2

    def refresh_cache(self, source: str = "auto") -> bool:
        """
        刷新交易日历缓存

        Args:
            source: 数据源，可选 'akshare', 'tushare', 'auto'

        Returns:
            bool: 是否刷新成功
        """
        self._holidays_cache = None
        self._trading_days_cache = None
        self._cache_data = None

        old_source = self.prefer_source
        self.prefer_source = source

        holidays = self.load_holidays()
        trading_days = self.load_trading_days()

        self.prefer_source = old_source

        success = len(holidays) > 0 or len(trading_days) > 0
        if success:
            logger.info(f"交易日历缓存已刷新，节假日: {len(holidays)}, 交易日: {len(trading_days)}")
        else:
            logger.warning("交易日历缓存刷新失败")

        return success

    def get_latest_trading_date(self, check_date: str = None) -> str:
        """
        获取指定日期之前的最近交易日

        Args:
            check_date: 检查日期 (YYYYMMDD格式)，默认为今天

        Returns:
            str: 最近交易日期 (YYYYMMDD格式)
        """
        if check_date is None:
            check_date = datetime.now().strftime('%Y%m%d')

        try:
            dt = datetime.strptime(check_date, '%Y%m%d')
        except:
            dt = datetime.now()

        # 向前查找交易日
        for i in range(10):  # 最多查找10天
            candidate = (dt - timedelta(days=i)).strftime('%Y%m%d')
            if self.is_trading_day(candidate):
                # 验证数据库中是否有这个日期的数据
                stocks = self.storage.get_all_stocks()
                if stocks:
                    df = self.storage.get_daily_prices(stocks[0], candidate, candidate)
                    if df is not None and not df.empty:
                        return candidate

        # 如果找不到，返回None
        return None

    def get_next_trading_day(self, check_date: str = None) -> str:
        """
        获取指定日期之后的下一个交易日

        Args:
            check_date: 检查日期 (YYYYMMDD格式)，默认为今天

        Returns:
            str: 下一个交易日 (YYYYMMDD格式)
        """
        if check_date is None:
            check_date = datetime.now().strftime('%Y%m%d')

        try:
            dt = datetime.strptime(check_date, '%Y%m%d')
        except:
            dt = datetime.now()

        # 向后查找交易日
        for i in range(1, 11):  # 最多查找10天
            candidate = (dt + timedelta(days=i)).strftime('%Y%m%d')
            if self.is_trading_day(candidate):
                return candidate

        return None

    def should_run_daily_routine(self) -> tuple:
        """
        判断今天是否应该运行每日例程

        Returns:
            tuple: (should_run: bool, reason: str)
        """
        today = datetime.now()

        # 检查是否为交易日
        if not self.is_trading_day():
            return False, f"今天({today.strftime('%Y-%m-%d')})不是交易日"

        # 检查是否已过执行时间
        execution_time = today.replace(
            hour=settings.DAILY_ROUTINE_HOUR,
            minute=settings.DAILY_ROUTINE_MINUTE,
            second=0,
            microsecond=0
        )

        if today < execution_time:
            return False, f"未到执行时间({settings.DAILY_ROUTINE_HOUR:02d}:{settings.DAILY_ROUTINE_MINUTE:02d})"

        # 检查今日数据是否已更新
        latest_date = self.get_latest_trading_date()
        if latest_date != today.strftime('%Y%m%d'):
            return False, f"今日数据尚未更新(最新: {latest_date})"

        return True, "满足执行条件"


# 便捷函数
_calendar_instance = None

def get_trade_calendar() -> TradeCalendar:
    """获取交易日历单例"""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = TradeCalendar()
    return _calendar_instance


def is_trading_day(check_date: str = None) -> bool:
    """判断是否为交易日（便捷函数）"""
    return get_trade_calendar().is_trading_day(check_date)


def get_t1_sell_date(buy_date: str) -> Optional[str]:
    """获取 T+1 可卖日期（便捷函数）"""
    return get_trade_calendar().get_t1_sell_date(buy_date)


def get_next_trading_day(check_date: str = None) -> Optional[str]:
    """获取下一个交易日（便捷函数）"""
    return get_trade_calendar().get_next_trading_day(check_date)


def get_prev_trading_day(check_date: str = None) -> Optional[str]:
    """获取前一个交易日（便捷函数）"""
    return get_trade_calendar().get_prev_trading_day(check_date)


def should_run_daily_routine() -> tuple:
    """判断是否应该运行每日例程（便捷函数）"""
    return get_trade_calendar().should_run_daily_routine()


if __name__ == "__main__":
    """测试交易日历"""
    calendar = get_trade_calendar()

    print("交易日历测试")
    print("=" * 60)

    # 测试今天
    today = datetime.now().strftime('%Y%m%d')
    print(f"\n今天: {today}")
    print(f"  是否交易日: {calendar.is_trading_day(today)}")

    # 测试最近交易日
    latest = calendar.get_latest_trading_date()
    print(f"  最近交易日: {latest}")

    # 测试下一个交易日
    next_day = calendar.get_next_trading_day()
    print(f"  下一个交易日: {next_day}")

    # 测试前一个交易日
    prev_day = calendar.get_prev_trading_day()
    print(f"  前一个交易日: {prev_day}")

    # 测试 T+1 可卖日期
    if latest:
        t1_date = calendar.get_t1_sell_date(latest)
        print(f"  T+1 可卖日期（从{latest}买入）: {t1_date}")

    # 测试是否应该运行
    should_run, reason = calendar.should_run_daily_routine()
    print(f"\n是否应该运行每日例程: {should_run}")
    print(f"  原因: {reason}")

    # 测试交易日区间
    print("\n测试交易日区间:")
    start = "20240101"
    end = "20240110"
    trading_days = calendar.get_trading_days_between(start, end)
    print(f"  {start} 到 {end} 之间的交易日: {trading_days}")
    print(f"  交易日数量: {len(trading_days)}")

    # 测试第N个交易日
    print("\n测试第N个交易日:")
    nth_day = calendar.get_nth_trading_day("20240101", 5)
    print(f"  从 20240101 起第5个交易日: {nth_day}")

    # 显示最近10天的交易日状态
    print("\n最近10天:")
    for i in range(10):
        date_str = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d (%A)')
        date_num = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        is_trading = calendar.is_trading_day(date_num)
        status = "交易日" if is_trading else "非交易日"
        print(f"  {date_str}: {status}")
