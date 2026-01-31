#!/usr/bin/env python3
"""
获取今天（1月30日）的最新数据
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.fetchers.tushare import TushareDataFetcher
from loguru import logger


# 高级token配置
ADVANCED_TOKEN = "464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976"
PROXY_URL = "http://lianghua.nanyangqiankun.top"


def main():
    """获取今天的数据"""
    today = datetime.now().strftime('%Y%m%d')
    logger.info(f"获取 {today} 的最新数据...")

    # 初始化
    fetcher = TushareDataFetcher(
        token=ADVANCED_TOKEN,
        proxy_url=PROXY_URL
    )
    storage = SQLiteStorage()

    # 获取所有股票
    stocks = storage.get_all_stocks()
    logger.info(f"股票池: {len(stocks)} 只")

    success = 0
    failed = 0
    no_new_data = 0

    for i, ts_code in enumerate(stocks):
        try:
            # 获取今天的数据
            df = fetcher.get_daily_price(ts_code, today, today)

            if df is not None and not df.empty:
                storage.save_daily_prices(df, ts_code)
                success += 1

                if (i + 1) % 500 == 0 or i < 10:
                    logger.info(f"  [{i+1:4d}/{len(stocks)}] {ts_code}: 获取 {len(df)} 条")
            else:
                no_new_data += 1

        except Exception as e:
            failed += 1
            logger.warning(f"获取 {ts_code} 失败: {e}")

    logger.info(f"\n完成!")
    logger.info(f"  成功: {success}")
    logger.info(f"  无数据: {no_new_data}")
    logger.info(f"  失败: {failed}")


if __name__ == "__main__":
    main()
