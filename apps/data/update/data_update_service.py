#!/usr/bin/env python3
"""
数据更新服务 - 使用高级token

定期从Tushare获取最新数据并更新数据库
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.fetchers.tushare import TushareDataFetcher
from loguru import logger


# 高级token配置
ADVANCED_TOKEN = "464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976"
PROXY_URL = "http://lianghua.nanyangqiankun.top"


class DataUpdateService:
    """数据更新服务"""

    def __init__(self, update_interval_minutes=30):
        """
        初始化

        Args:
            update_interval_minutes: 更新间隔（分钟）
        """
        # 使用高级token和代理
        self.fetcher = TushareDataFetcher(
            token=ADVANCED_TOKEN,
            proxy_url=PROXY_URL
        )
        self.storage = SQLiteStorage()
        self.update_interval = update_interval_minutes * 60

        logger.info(f"数据更新服务初始化完成")
        logger.info(f"  更新间隔: {update_interval_minutes} 分钟")
        logger.info(f"  使用代理: {PROXY_URL}")

    def get_latest_data_date(self, ts_code):
        """获取指定股票的最新数据日期"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')

            df = self.fetcher.get_daily_price(ts_code, start_date, end_date)
            if df is not None and not df.empty:
                return df['trade_date'].max()
            return None
        except Exception as e:
            logger.warning(f"获取 {ts_code} 最新日期失败: {e}")
            return None

    def update_all_stocks(self):
        """更新所有股票的最新数据"""
        logger.info("=" * 60)
        logger.info(f"开始更新数据: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # 获取股票列表
        stocks = self.storage.get_all_stocks()
        logger.info(f"股票池: {len(stocks)} 只")

        # 获取今天的日期
        today = datetime.now().strftime('%Y%m%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

        success = 0
        failed = 0
        no_new_data = 0

        for i, ts_code in enumerate(stocks):
            try:
                # 检查是否需要更新（看是否有今天的数据）
                existing = self.storage.get_daily_prices(ts_code, today, today)

                # 如果没有今天的数据，尝试获取
                if existing is None or existing.empty:
                    # 获取最近3天数据
                    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y%m%d')
                    df = self.fetcher.get_daily_price(ts_code, start_date, today)

                    if df is not None and not df.empty:
                        self.storage.save_daily_prices(df, ts_code)
                        success += 1

                        # 显示最新数据日期
                        latest_date = df['trade_date'].max()
                        if i < 10 or (i + 1) % 100 == 0:
                            logger.info(f"  [{i+1:4d}/{len(stocks)}] {ts_code}: 最新日期 {latest_date}")
                    else:
                        failed += 1
                else:
                    no_new_data += 1
                    if i < 5:
                        logger.info(f"  [{i+1:4d}/{len(stocks)}] {ts_code}: 已是最新")

                # 每50只股票稍微休息一下（避免代理服务器压力）
                if (i + 1) % 50 == 0:
                    logger.info(f"进度: {i+1}/{len(stocks)}, 成功: {success}, 无新数据: {no_new_data}, 失败: {failed}")
                    time.sleep(1)

            except Exception as e:
                failed += 1
                logger.warning(f"更新 {ts_code} 失败: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("更新完成")
        logger.info(f"  成功: {success}")
        logger.info(f"  无新数据: {no_new_data}")
        logger.info(f"  失败: {failed}")
        logger.info("=" * 60)

        return success, no_new_data, failed

    def check_data freshness(self):
        """检查数据新鲜度"""
        today = datetime.now().strftime('%Y%m%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

        stocks = self.storage.get_all_stocks()[:50]

        has_today = 0
        has_yesterday = 0
        older = 0

        for ts_code in stocks:
            try:
                df = self.storage.get_daily_prices(ts_code, today, today)
                if df is not None and not df.empty:
                    has_today += 1
                    continue

                df = self.storage.get_daily_prices(ts_code, yesterday, yesterday)
                if df is not None and not df.empty:
                    has_yesterday += 1
                    continue

                older += 1
            except:
                older += 1

        logger.info(f"数据新鲜度（前50只股票）:")
        logger.info(f"  有今天数据: {has_today}")
        logger.info(f"  只有昨天: {has_yesterday}")
        logger.info(f"  更旧数据: {older}")

        return has_today, has_yesterday, older

    def run_loop(self):
        """持续运行"""
        logger.info("数据更新服务启动...")
        logger.info(f"更新间隔: {self.update_interval / 60:.1f} 分钟")

        while True:
            try:
                # 检查数据新鲜度
                self.check_data_freshness()

                # 更新数据
                self.update_all_stocks()

            except Exception as e:
                logger.error(f"更新出错: {e}")

            # 等待下次更新
            logger.info(f"\n等待 {self.update_interval / 60:.1f} 分钟后下次更新...")
            time.sleep(self.update_interval)


def main():
    """主函数"""
    from loguru import logger

    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/data_update_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG"
    )

    # 创建服务（每30分钟更新一次）
    service = DataUpdateService(update_interval_minutes=30)

    # 先运行一次
    service.check_data_freshness()
    service.update_all_stocks()

    # 进入循环
    service.run_loop()


if __name__ == "__main__":
    main()
