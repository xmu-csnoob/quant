"""
查看A股下载进度
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.sqlite_storage import SQLiteStorage


def check_progress():
    """查看下载进度"""
    storage = SQLiteStorage()
    stats = storage.get_stats()

    print("=" * 60)
    print("A股数据下载进度")
    print("=" * 60)

    print(f"\n数据库:")
    print(f"  文件: {storage.db_path}")
    print(f"  大小: {stats['db_size_mb']:.2f} MB")
    print(f"  股票数: {stats['stock_count']}")
    print(f"  数据行: {stats['total_rows']:,}")
    print(f"  日期范围: {stats['date_range']}")
    print(f"  交易记录: {stats['trade_count']}")

    # 计算进度
    target = 4838
    current = stats['stock_count']
    progress = current / target * 100

    print(f"\n进度: {current}/{target} ({progress:.1f}%)")

    if progress < 100:
        remaining = target - current
        print(f"剩余: {remaining} 只股票")

    # 查看最新下载的股票
    stocks = sorted(storage.get_all_stocks())
    if stocks:
        print(f"\n最新股票: {stocks[-5:]}")


if __name__ == "__main__":
    check_progress()
