"""
使用代理API批量下载所有A股数据

无频率限制，可以快速下载
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.fetchers.tushare import TushareDataFetcher


def download_all_a_stocks():
    """下载所有A股数据"""
    print("=" * 80)
    print("批量下载所有A股数据（使用代理API，无频率限制）")
    print("=" * 80)

    # 使用代理API
    fetcher = TushareDataFetcher(
        token="464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976",
        proxy_url="http://lianghua.nanyangqiankun.top"
    )

    storage = SQLiteStorage()

    # 获取股票列表
    print("\n[1] 获取股票列表...")
    try:
        stock_list = fetcher.get_stock_list()
        print(f"  ✓ 获取到 {len(stock_list)} 只股票")
    except Exception as e:
        print(f"  ✗ 获取失败: {e}")
        return

    # 过滤股票
    print("\n[2] 过滤股票...")
    original_count = len(stock_list)

    # 排除ST、退市、新股
    stock_list = stock_list[
        (~stock_list['name'].str.contains('ST')) &
        (~stock_list['name'].str.contains('退')) &
        (stock_list['list_date'] < '20240101')
    ].copy()

    print(f"  原始: {original_count} 只")
    print(f"  过滤后: {len(stock_list)} 只")

    # 按市场分组
    sse_stocks = stock_list[stock_list['ts_code'].str.endswith('.SH')]
    szse_stocks = stock_list[stock_list['ts_code'].str.endswith('.SZ')]

    print(f"\n[3] 市场分布:")
    print(f"  上交所: {len(sse_stocks)} 只")
    print(f"  深交所: {len(szse_stocks)} 只")

    # 下载
    print(f"\n[4] 开始下载...")

    success = 0
    failed = 0
    failed_list = []

    total = len(stock_list)

    for i, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']

        # 每100只显示进度
        if (i + 1) % 100 == 0:
            print(f"\n  进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

        print(f"  [{i+1:4d}/{total}] {ts_code} {name}...", end=" ", flush=True)

        try:
            df = fetcher.get_daily_price(ts_code, "20200101", "20241231")

            if df is None or df.empty:
                print("✗ 无数据")
                failed += 1
                failed_list.append((ts_code, name, "无数据"))
                continue

            rows = storage.save_daily_prices(df, ts_code)
            print(f"✓ {rows} 行")
            success += 1

        except Exception as e:
            print(f"✗ {str(e)[:30]}")
            failed += 1
            failed_list.append((ts_code, name, str(e)[:50]))

    # 统计
    print("\n" + "=" * 80)
    print("下载完成")
    print("=" * 80)
    print(f"成功: {success} 只")
    print(f"失败: {failed} 只")

    if failed_list:
        print(f"\n失败列表（前10个）:")
        for code, name, reason in failed_list[:10]:
            print(f"  - {code} {name}: {reason}")

    stats = storage.get_stats()
    print(f"\n数据库统计:")
    print(f"  文件: {storage.db_path}")
    print(f"  大小: {stats['db_size_mb']:.2f} MB")
    print(f"  股票数: {stats['stock_count']}")
    print(f"  数据行: {stats['total_rows']:,}")
    print(f"  日期范围: {stats['date_range']}")


def download_index_data():
    """下载指数数据"""
    print("\n" + "=" * 80)
    print("下载指数数据")
    print("=" * 80)

    fetcher = TushareDataFetcher(
        token="464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976",
        proxy_url="http://lianghua.nanyangqiankun.top"
    )

    storage = SQLiteStorage()

    # 主要指数
    indices = [
        "000001.SH",  # 上证指数
        "399001.SZ",  # 深证成指
        "399006.SZ",  # 创业板指
        "000300.SH",  # 沪深300
        "000905.SH",  # 中证500
        "000016.SH",  # 上证50
        "000688.SH",  # 科创50
        "000852.SH",  # 中证1000
    ]

    print(f"\n下载 {len(indices)} 个指数...")

    for i, ts_code in enumerate(indices, 1):
        print(f"  [{i}/{len(indices)}] {ts_code}...", end=" ", flush=True)

        try:
            df = fetcher.get_daily_price(ts_code, "20200101", "20241231")

            if df is not None and not df.empty:
                rows = storage.save_daily_prices(df, ts_code)
                print(f"✓ {rows} 行")
            else:
                print("✗ 无数据")

        except Exception as e:
            print(f"✗ {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量下载A股数据")
    parser.add_argument("--index-only", action="store_true", help="仅下载指数数据")
    parser.add_argument("--limit", type=int, help="限制下载数量（测试用）")

    args = parser.parse_args()

    if args.index_only:
        download_index_data()
    else:
        download_all_a_stocks()

        # 同时下载指数
        download_index_data()
