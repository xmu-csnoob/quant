"""
批量下载股票数据

扩充数据量，支持SQLite和CSV双存储
"""

import sys
import time
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.storage import DataStorage
from data.storage.sqlite_storage import SQLiteStorage
from data.fetchers.tushare import TushareDataFetcher
from data.fetchers.base import Exchange
from data.api.data_manager import DataManager
from loguru import logger


# A股主要股票列表（精选优质标的）
STOCK_LISTS = {
    # 上交所主板
    "sse_blue_chips": [
        "600000.SH",  # 浦发银行
        "600036.SH",  # 招商银行
        "600519.SH",  # 贵州茅台
        "600887.SH",  # 伊利股份
        "601318.SH",  # 中国平安
        "601398.SH",  # 工商银行
        "601857.SH",  # 中国石油
        "601939.SH",  # 建设银行
        "601988.SH",  # 中国银行
        "601288.SH",  # 农业银行
        "600030.SH",  # 中信证券
        "601336.SH",  # 新华保险
        "601601.SH",  # 中国太保
        "601668.SH",  # 中国建筑
        "601628.SH",  # 中国人寿
        "600276.SH",  # 恒瑞医药
        "600028.SH",  # 中国石化
        "601088.SH",  # 中国神华
        "600900.SH",  # 长江电力
        "600031.SH",  # 三一重工
    ],

    # 上交所科创板
    "sse_star": [
        "688981.SH",  # 中芯国际
        "688111.SH",  # 金山办公
        "688009.SH",  # 中国通号
        "688012.SH",  # 中微公司
        "688036.SH",  # 传音控股
        "688187.SH",  # 时代电气
        "688223.SH",  # 晶科能源
        "688256.SH",  # 寒武纪
        "688303.SH",  # 大全能源
        "688499.SH",  # 丰山集团
    ],

    # 深交所主板
    "szse_blue_chips": [
        "000001.SZ",  # 平安银行
        "000002.SZ",  # 万科A
        "000333.SZ",  # 美的集团
        "000651.SZ",  # 格力电器
        "000858.SZ",  # 五粮液
        "000876.SZ",  # 新希望
        "001979.SZ",  # 招商蛇口
        "002594.SZ",  # 比亚迪
    ],

    # 深交所创业板
    "szse_chi_next": [
        "300015.SZ",  # 爱尔眼科
        "300750.SZ",  # 宁德时代
        "300059.SZ",  # 东方财富
        "300124.SZ",  # 汇川技术
        "300142.SZ",  # 沃森生物
        "300274.SZ",  # 阳光电源
        "300413.SZ",  # 芒果超媒
        "300760.SZ",  # 迈瑞医疗
        "300896.SZ",  # 爱美客
        "300999.SZ",  # 金龙鱼
    ],

    # 指数ETF
    "etf": [
        "510300.SH",  # 沪深300ETF
        "510500.SH",  # 中证500ETF
        "510050.SH",  # 上证50ETF
        "159915.SZ",  # 创业板ETF
        "159919.SZ",  # 沪深300ETF
        "512000.SH",  # 券商ETF
        "512480.SH",  # 半导体ETF
        "515050.SH",  # 5GETF
        "515880.SH",  # 通信ETF
        "516160.SH",  # 新能源车ETF
    ]
}


def batch_download(
    categories: list = None,
    start_date: str = "20200101",
    end_date: str = "20241231",
    use_sqlite: bool = True,
    use_csv: bool = True
):
    """
    批量下载股票数据

    Args:
        categories: 要下载的类别列表，None表示全部
        start_date: 开始日期
        end_date: 结束日期
        use_sqlite: 是否使用SQLite存储
        use_csv: 是否使用CSV存储
    """
    print("=" * 80)
    print("A股数据批量下载")
    print("=" * 80)

    # 初始化
    sqlite_storage = SQLiteStorage() if use_sqlite else None
    csv_storage = DataStorage() if use_csv else None

    fetcher = TushareDataFetcher()

    # 选择类别
    if categories is None:
        categories = list(STOCK_LISTS.keys())

    # 统计
    total_stocks = 0
    success_count = 0
    failed_stocks = []

    for category in categories:
        if category not in STOCK_LISTS:
            logger.warning(f"Unknown category: {category}")
            continue

        stocks = STOCK_LISTS[category]
        print(f"\n[{category}] {len(stocks)} 只股票")

        for i, ts_code in enumerate(stocks, 1):
            print(f"  [{i}/{len(stocks)}] {ts_code}...", end=" ", flush=True)

            try:
                # 获取数据
                df = fetcher.get_daily_price(ts_code, start_date, end_date)

                if df is None or df.empty:
                    print("✗ 无数据")
                    failed_stocks.append(ts_code)
                    continue

                # 保存到SQLite
                if sqlite_storage:
                    sqlite_storage.save_daily_prices(df, ts_code)

                # 保存到CSV
                if csv_storage:
                    # 判断交易所
                    exchange = Exchange.SSE if ts_code.endswith('.SH') else Exchange.SZSE
                    csv_storage.save_daily_price(df, ts_code, exchange)

                print(f"✓ {len(df)} 行")
                success_count += 1
                total_stocks += 1

                # 避免频率限制
                time.sleep(0.5)

            except Exception as e:
                print(f"✗ 失败: {e}")
                failed_stocks.append(ts_code)
                logger.error(f"Failed to download {ts_code}: {e}")

    # 统计结果
    print("\n" + "=" * 80)
    print("下载完成")
    print("=" * 80)

    print(f"\n总计: {total_stocks} 只股票成功")
    print(f"失败: {len(failed_stocks)} 只")

    if failed_stocks:
        print("\n失败列表:")
        for code in failed_stocks:
            print(f"  - {code}")

    # SQLite统计
    if sqlite_storage:
        stats = sqlite_storage.get_stats()
        print(f"\nSQLite数据库:")
        print(f"  文件: {sqlite_storage.db_path}")
        print(f"  大小: {stats.get('db_size_mb', 0):.2f} MB")
        print(f"  股票数: {stats['stock_count']}")
        print(f"  数据行: {stats['total_rows']:,}")
        print(f"  日期范围: {stats['date_range']}")

    # CSV统计
    if csv_storage:
        print(f"\nCSV文件:")
        print(f"  目录: {csv_storage.base_path}")

        # 统计CSV文件数量
        csv_count = 0
        for exchange_dir in csv_storage.base_path.iterdir():
            if exchange_dir.is_dir():
                daily_dir = exchange_dir / "stocks" / "daily"
                if daily_dir.exists():
                    csv_count += len(list(daily_dir.glob("*.csv")))

        print(f"  文件数: {csv_count}")


def quick_download():
    """快速下载：主要蓝筹股"""
    print("快速下载：A股主要蓝筹股（50只）")

    batch_download(
        categories=["sse_blue_chips", "szse_blue_chips"],
        start_date="20200101",
        end_date="20241231"
    )


def full_download():
    """全量下载：所有类别"""
    print("全量下载：所有股票和ETF")

    batch_download(
        categories=None,  # 全部
        start_date="20200101",
        end_date="20241231"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量下载A股数据")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="下载模式: quick(50只) 或 full(全部)")
    parser.add_argument("--start", default="20200101", help="开始日期")
    parser.add_argument("--end", default="20241231", help="结束日期")
    parser.add_argument("--no-sqlite", action="store_true", help="不使用SQLite")
    parser.add_argument("--no-csv", action="store_true", help="不使用CSV")

    args = parser.parse_args()

    if args.mode == "quick":
        batch_download(
            categories=["sse_blue_chips", "szse_blue_chips"],
            start_date=args.start,
            end_date=args.end,
            use_sqlite=not args.no_sqlite,
            use_csv=not args.no_csv
        )
    else:
        batch_download(
            categories=None,
            start_date=args.start,
            end_date=args.end,
            use_sqlite=not args.no_sqlite,
            use_csv=not args.no_csv
        )
