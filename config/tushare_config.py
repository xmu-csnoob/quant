"""
Tushare API 配置

支持官方API和代理API（无频率限制）
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage.sqlite_storage import SQLiteStorage
from data.fetchers.tushare import TushareDataFetcher
import time


# API配置
TUSHARE_CONFIG = {
    # 使用代理API（无频率限制）
    "proxy": {
        "token": "464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976",
        "proxy_url": "http://lianghua.nanyangqiankun.top",
    },

    # 官方API（需要申请token，有频率限制）
    "official": {
        "token": os.getenv("TUSHARE_TOKEN", ""),
        "proxy_url": None,
    },
}


def create_fetcher(use_proxy: bool = True) -> TushareDataFetcher:
    """
    创建Tushare数据获取器

    Args:
        use_proxy: 是否使用代理API（无频率限制）

    Returns:
        TushareDataFetcher实例
    """
    if use_proxy:
        config = TUSHARE_CONFIG["proxy"]
        return TushareDataFetcher(
            token=config["token"],
            proxy_url=config["proxy_url"]
        )
    else:
        config = TUSHARE_CONFIG["official"]
        if not config["token"]:
            raise ValueError("官方API需要设置TUSHARE_TOKEN环境变量")
        return TushareDataFetcher(token=config["token"])


def batch_download_all_stocks(use_proxy: bool = True):
    """
    批量下载所有A股数据

    使用代理API可以无限制下载
    """
    print("=" * 80)
    print("批量下载所有A股数据")
    print("=" * 80)

    fetcher = create_fetcher(use_proxy=use_proxy)
    storage = SQLiteStorage()

    # 先获取股票列表
    print("\n[1] 获取股票列表...")
    try:
        stock_list = fetcher.get_stock_list()
        print(f"  ✓ 获取到 {len(stock_list)} 只股票")
    except Exception as e:
        print(f"  ✗ 获取失败: {e}")
        return

    # 过滤股票（排除ST、退市等）
    stock_list = stock_list[
        (~stock_list['name'].str.contains('ST')) &
        (~stock_list['name'].str.contains('退')) &
        (stock_list['list_date'] < '20240101')  # 排除新股
    ]

    print(f"  过滤后: {len(stock_list)} 只股票")

    # 批量下载
    print(f"\n[2] 开始下载...")

    success = 0
    failed = 0

    for i, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']

        print(f"  [{i+1}/{len(stock_list)}] {ts_code} {name}...", end=" ", flush=True)

        try:
            df = fetcher.get_daily_price(ts_code, "20200101", "20241231")

            if df is None or df.empty:
                print("✗ 无数据")
                failed += 1
                continue

            rows = storage.save_daily_prices(df, ts_code)
            print(f"✓ {rows} 行")
            success += 1

            # 使用代理可以快速下载，无需等待
            if not use_proxy:
                time.sleep(0.5)

        except Exception as e:
            print(f"✗ {e}")
            failed += 1

    # 统计
    print("\n" + "=" * 80)
    print("下载完成")
    print("=" * 80)
    print(f"成功: {success} 只")
    print(f"失败: {failed} 只")

    stats = storage.get_stats()
    print(f"\n数据库:")
    print(f"  股票数: {stats['stock_count']}")
    print(f"  数据行: {stats['total_rows']:,}")
    print(f"  大小: {stats['db_size_mb']:.2f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tushare数据下载")
    parser.add_argument("--mode", choices=["proxy", "official"], default="proxy",
                       help="API模式: proxy(无限制) 或 official(有频率限制)")
    parser.add_argument("--test", action="store_true", help="测试模式：下载单只股票")

    args = parser.parse_args()

    if args.test:
        print("测试模式：下载单只股票")
        fetcher = create_fetcher(use_proxy=(args.mode == "proxy"))

        df = fetcher.get_daily_price("000001.SZ", "20240101", "20240131")
        print(f"\n获取到 {len(df)} 行数据:")
        print(df.head())

    else:
        batch_download_all_stocks(use_proxy=(args.mode == "proxy"))
