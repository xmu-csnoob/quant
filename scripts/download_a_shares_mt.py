"""
使用代理API多线程批量下载所有A股数据

多线程下载，大幅提升速度
"""

import sys
from pathlib import Path
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.sqlite_storage import SQLiteStorage
from data.fetchers.tushare import TushareDataFetcher
from data.fetchers.base import Exchange


class MultiThreadDownloader:
    """多线程下载器"""

    def __init__(self, max_workers=10, progress_tty="/dev/pts/0"):
        """
        初始化

        Args:
            max_workers: 线程数，默认10
            progress_tty: 进度输出终端
        """
        self.fetcher = TushareDataFetcher(
            token="464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976",
            proxy_url="http://lianghua.nanyangqiankun.top"
        )
        self.storage = SQLiteStorage()
        self.max_workers = max_workers
        self.progress_tty = progress_tty

        # 尝试打开终端进行实时进度输出
        self.tty_file = None
        if progress_tty and os.path.exists(progress_tty):
            try:
                self.tty_file = open(progress_tty, 'w')
            except:
                pass

        # 统计
        self.success = 0
        self.failed = 0
        self.skipped = 0
        self.lock = Lock()
        self.start_time = None
        self.total = 0

    def print_progress(self, message):
        """输出进度到终端和tty"""
        print(message)
        if self.tty_file:
            self.tty_file.write(message + "\n")
            self.tty_file.flush()

    def update_progress_display(self):
        """更新进度显示"""
        if not self.start_time:
            return

        with self.lock:
            elapsed = time.time() - self.start_time
            total_done = self.success + self.failed + self.skipped
            progress = total_done / self.total * 100 if self.total > 0 else 0
            speed = total_done / elapsed if elapsed > 0 else 0
            eta = (self.total - total_done) / speed if speed > 0 else 0

            stats = self.storage.get_stats()

            progress_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           A股数据下载实时进度                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  进度: {total_done:5d}/{self.total:5d} ({progress:6.2f}%)   速度: {speed:5.2f}只/秒   ETA: {eta/60:5.1f}分钟  ║
║  新增: {self.success:5d}  已存在: {self.skipped:5d}  失败: {self.failed:4d}                              ║
║  数据库: {stats['db_size_mb']:6.2f} MB  股票: {stats['stock_count']:5d}  行: {stats['total_rows']:8,}            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            # 清除tty并输出（使用ANSI转移码）
            if self.tty_file:
                self.tty_file.write("\033[2J\033[H")  # 清屏并移到顶部
                self.tty_file.write(progress_msg)
                self.tty_file.flush()

    def download_one_stock(self, ts_code, name):
        """
        下载单只股票

        Args:
            ts_code: 股票代码
            name: 股票名称

        Returns:
            (ts_code, name, status, rows)
        """
        try:
            df = self.fetcher.get_daily_price(ts_code, "20200101", "20241231")

            if df is None or df.empty:
                return (ts_code, name, "failed", 0, "无数据")

            rows = self.storage.save_daily_prices(df, ts_code)

            if rows == 0:
                return (ts_code, name, "skipped", 0, "")
            else:
                return (ts_code, name, "success", rows, "")

        except Exception as e:
            return (ts_code, name, "failed", 0, str(e)[:30])

    def update_stats(self, status, rows=0):
        """更新统计"""
        with self.lock:
            if status == "success":
                self.success += 1
            elif status == "failed":
                self.failed += 1
            elif status == "skipped":
                self.skipped += 1

    def download_all(self, stock_list):
        """
        下载所有股票

        Args:
            stock_list: 股票列表 DataFrame
        """
        self.total = len(stock_list)
        self.start_time = time.time()

        self.print_progress(f"\n[3/4] 开始下载（{self.max_workers}线程）...")

        # 使用线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {}
            for i, row in stock_list.iterrows():
                ts_code = row['ts_code']
                name = row['name']
                future = executor.submit(self.download_one_stock, ts_code, name)
                futures[future] = (i, ts_code, name)

            # 处理完成的任务
            for future in as_completed(futures):
                idx, ts_code, name = futures[future]

                try:
                    result = future.result()
                    ts_code, name, status, rows, error = result

                    # 更新统计
                    self.update_stats(status, rows)

                    # 每10只更新一次进度显示
                    total_done = self.success + self.failed + self.skipped
                    if total_done % 10 == 0:
                        self.update_progress_display()

                except Exception as e:
                    self.print_progress(f"  ✗ [{idx+1:4d}/{self.total}] {ts_code} 异常: {e}")
                    self.failed += 1

        # 最终更新
        self.update_progress_display()

        elapsed = time.time() - self.start_time
        return elapsed


def main():
    """主函数"""
    print("=" * 80)
    print("多线程批量下载所有A股数据")
    print("=" * 80)

    # 创建下载器（10线程）
    downloader = MultiThreadDownloader(max_workers=10)

    # 获取股票列表
    print("\n[1/4] 获取股票列表...")
    try:
        sse_list = downloader.fetcher.get_stock_list(Exchange.SSE)
        szse_list = downloader.fetcher.get_stock_list(Exchange.SZSE)
        stock_list = pd.concat([sse_list, szse_list], ignore_index=True)
        print(f"      获取到 {len(stock_list)} 只股票")
    except Exception as e:
        print(f"      获取失败: {e}")
        return

    # 过滤股票
    print("\n[2/4] 过滤股票...")
    original_count = len(stock_list)
    stock_list = stock_list[
        (~stock_list['name'].str.contains('ST')) &
        (~stock_list['name'].str.contains('退')) &
        (~stock_list['name'].str.contains('停')) &
        (stock_list['list_date'] < '20240101')
    ].copy()

    print(f"      原始: {original_count} 只")
    print(f"      过滤后: {len(stock_list)} 只")

    # 下载
    elapsed = downloader.download_all(stock_list)

    # 统计
    print("\n" + "=" * 80)
    print("下载完成")
    print("=" * 80)
    print(f"\n耗时: {elapsed/60:.1f} 分钟")
    print(f"\n结果统计:")
    print(f"  新增: {downloader.success} 只")
    print(f"  已存在: {downloader.skipped} 只")
    print(f"  失败: {downloader.failed} 只")

    # 数据库统计
    stats = downloader.storage.get_stats()
    print(f"\n数据库统计:")
    print(f"  文件: {downloader.storage.db_path}")
    print(f"  大小: {stats['db_size_mb']:.2f} MB")
    print(f"  股票数: {stats['stock_count']}")
    print(f"  数据行: {stats['total_rows']:,}")
    print(f"  日期范围: {stats['date_range']}")


if __name__ == "__main__":
    main()
