"""
交易日历工具

判断是否为交易日，避免在节假日/周末执行交易
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from config import settings


class TradeCalendar:
    """交易日历管理"""

    def __init__(self):
        """初始化交易日历"""
        self.storage = SQLiteStorage()
        self.cache_file = settings.DATA_DIR / "trade_calendar_cache.json"
        self._holidays_cache = None

    def load_holidays(self) -> set:
        """
        加载节假日缓存

        Returns:
            set: 节假日日期集合 (YYYYMMDD格式字符串)
        """
        if self._holidays_cache is not None:
            return self._holidays_cache

        # 尝试从缓存加载
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self._holidays_cache = set(data.get('holidays', []))
                    return self._holidays_cache
            except:
                pass

        # 从数据库推断非交易日
        self._holidays_cache = self._infer_holidays_from_db()

        # 保存缓存
        self._save_holidays_cache()

        return self._holidays_cache

    def _infer_holidays_from_db(self) -> set:
        """
        从数据库推断非交易日

        通过检查某个股票的数据缺口来推断节假日
        """
        holidays = set()

        try:
            # 获取第一只股票的数据
            stocks = self.storage.get_all_stocks()
            if not stocks:
                return holidays

            ts_code = stocks[0]

            # 获取最近一年的数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

            df = self.storage.get_daily_prices(ts_code, start_date, end_date)

            if df is None or df.empty:
                return holidays

            # 获取所有交易日
            trading_dates = set(df['trade_date'].astype(str).tolist())

            # 推断非交易日
            current = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')

            while current <= end:
                # 检查是否为周一到周五
                if current.weekday() < 5:  # 0-4 = 周一到周五
                    date_str = current.strftime('%Y%m%d')
                    if date_str not in trading_dates:
                        holidays.add(date_str)

                current += timedelta(days=1)

        except Exception as e:
            print(f"推断节假日失败: {e}")

        return holidays

    def _save_holidays_cache(self):
        """保存节假日缓存"""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'holidays': list(self._holidays_cache),
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
        except Exception as e:
            print(f"保存节假日缓存失败: {e}")

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

    # 测试是否应该运行
    should_run, reason = calendar.should_run_daily_routine()
    print(f"\n是否应该运行每日例程: {should_run}")
    print(f"  原因: {reason}")

    # 显示最近10天的交易日状态
    print("\n最近10天:")
    for i in range(10):
        date_str = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d (%A)')
        date_num = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        is_trading = calendar.is_trading_day(date_num)
        status = "交易日" if is_trading else "非交易日"
        print(f"  {date_str}: {status}")
