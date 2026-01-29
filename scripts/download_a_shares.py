"""
使用代理API批量下载所有A股数据

无频率限制，可以快速下载全市场数据
"""

import sys
from pathlib import Path
import time
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.sqlite_storage import SQLiteStorage
from data.fetchers.tushare import TushareDataFetcher
from data.fetchers.base import Exchange


def main():
    """主函数"""
    print("=" * 80)
    print("批量下载所有A股数据（使用代理API，无频率限制）")
    print("=" * 80)

    # 使用代理API
    print("\n初始化代理API...")
    fetcher = TushareDataFetcher(
        token="464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976",
        proxy_url="http://lianghua.nanyangqiankun.top"
    )

    storage = SQLiteStorage()

    # 获取股票列表
    print("\n[1/4] 获取股票列表...")
    try:
        # 分别获取上交所和深交所股票
        sse_list = fetcher.get_stock_list(Exchange.SSE)
        szse_list = fetcher.get_stock_list(Exchange.SZSE)

        stock_list = pd.concat([sse_list, szse_list], ignore_index=True)
        print(f"      获取到 {len(stock_list)} 只股票")
    except Exception as e:
        print(f"      获取失败: {e}")
        return

    # 过滤股票
    print("\n[2/4] 过滤股票...")
    original_count = len(stock_list)

    # 排除ST、退市、新股
    stock_list = stock_list[
        (~stock_list['name'].str.contains('ST')) &
        (~stock_list['name'].str.contains('退')) &
        (~stock_list['name'].str.contains('停')) &
        (stock_list['list_date'] < '20240101')
    ].copy()

    print(f"      原始: {original_count} 只")
    print(f"      过滤后: {len(stock_list)} 只")

    # 统计市场分布
    sse_count = len(stock_list[stock_list['ts_code'].str.endswith('.SH')])
    szse_count = len(stock_list[stock_list['ts_code'].str.endswith('.SZ')])

    print(f"\n      市场分布:")
    print(f"        上交所: {sse_count} 只")
    print(f"        深交所: {szse_count} 只")

    # 开始下载
    print(f"\n[3/4] 开始下载...")

    success = 0
    failed = 0
    skipped = 0
    failed_list = []

    total = len(stock_list)
    start_time = time.time()

    for i, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']

        # 每100只显示进度
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (total - i - 1) / speed
            print(f"\n      进度: {i+1}/{total} ({(i+1)/total*100:.1f}%) | "
                  f"速度: {speed:.1f}只/秒 | ETA: {eta/60:.1f}分钟")

        print(f"      [{i+1:4d}/{total}] {ts_code} {name}...", end=" ", flush=True)

        try:
            df = fetcher.get_daily_price(ts_code, "20200101", "20241231")

            if df is None or df.empty:
                print("✗ 无数据")
                failed += 1
                failed_list.append((ts_code, name, "无数据"))
                continue

            rows = storage.save_daily_prices(df, ts_code)

            if rows == 0:
                print("○ 已存在")
                skipped += 1
            else:
                print(f"✓ {rows} 行")
                success += 1

        except Exception as e:
            error_msg = str(e)[:30]
            print(f"✗ {error_msg}")
            failed += 1
            failed_list.append((ts_code, name, error_msg))

    # 最终统计
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("下载完成")
    print("=" * 80)
    print(f"\n耗时: {elapsed/60:.1f} 分钟")
    print(f"\n结果统计:")
    print(f"  新增: {success} 只")
    print(f"  已存在: {skipped} 只")
    print(f"  失败: {failed} 只")

    if failed_list:
        print(f"\n失败列表（前20个）:")
        for code, name, reason in failed_list[:20]:
            print(f"  - {code} {name}: {reason}")

    # 数据库统计
    stats = storage.get_stats()
    print(f"\n数据库统计:")
    print(f"  文件: {storage.db_path}")
    print(f"  大小: {stats['db_size_mb']:.2f} MB")
    print(f"  股票数: {stats['stock_count']}")
    print(f"  数据行: {stats['total_rows']:,}")
    print(f"  日期范围: {stats['date_range']}")


if __name__ == "__main__":
    main()
