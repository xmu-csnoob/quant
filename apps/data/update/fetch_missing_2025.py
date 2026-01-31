#!/usr/bin/env python3
"""
补充2025年缺失数据

获取1,296只在2026年有数据但2025年缺失的股票
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import sqlite3

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.fetchers.tushare import TushareDataFetcher
from loguru import logger


# 高级token配置
ADVANCED_TOKEN = "464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976"
PROXY_URL = "http://lianghua.nanyangqiankun.top"


def get_missing_stocks():
    """找出2025年缺失数据的股票"""
    conn = sqlite3.connect('data/quant.db')

    cursor = conn.execute('''
        SELECT DISTINCT a.ts_code
        FROM (
            SELECT DISTINCT ts_code FROM daily_prices WHERE trade_date >= '20260101'
        ) a
        LEFT JOIN (
            SELECT DISTINCT ts_code FROM daily_prices
            WHERE trade_date >= '20250101' AND trade_date <= '20251231'
        ) b ON a.ts_code = b.ts_code
        WHERE b.ts_code IS NULL
        ORDER BY a.ts_code
    ''')

    missing_stocks = [row[0] for row in cursor.fetchall()]
    conn.close()

    return missing_stocks


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("补充2025年缺失数据")
    logger.info("=" * 60)

    # 获取缺失股票列表
    missing_stocks = get_missing_stocks()
    logger.info(f"需要补充 {len(missing_stocks)} 只股票的2025年数据")

    if len(missing_stocks) == 0:
        logger.info("没有缺失数据，退出")
        return

    # 初始化
    fetcher = TushareDataFetcher(
        token=ADVANCED_TOKEN,
        proxy_url=PROXY_URL
    )
    storage = SQLiteStorage()

    # 2025年全年
    start_date = '20250101'
    end_date = '20251231'

    success = 0
    failed = 0
    skipped = 0

    for i, ts_code in enumerate(missing_stocks):
        try:
            # 检查是否已有数据
            existing = storage.get_daily_prices(ts_code, start_date, end_date)
            if existing is not None and not existing.empty:
                skipped += 1
                continue

            # 获取数据
            df = fetcher.get_daily_price(ts_code, start_date, end_date)

            if df is not None and not df.empty:
                storage.save_daily_prices(df, ts_code)
                success += 1

                if (i + 1) % 100 == 0:
                    logger.info(f"进度: {i+1}/{len(missing_stocks)}, 成功: {success}, 跳过: {skipped}, 失败: {failed}")
            else:
                failed += 1

        except Exception as e:
            failed += 1
            logger.warning(f"获取 {ts_code} 失败: {e}")

        # 每50只休息一下
        if (i + 1) % 50 == 0:
            time.sleep(1)

    logger.info("\n" + "=" * 60)
    logger.info("补充完成")
    logger.info(f"  成功: {success}")
    logger.info(f"  跳过: {skipped}")
    logger.info(f"  失败: {failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
