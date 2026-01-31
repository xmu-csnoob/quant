"""
实时监控下载进度并输出到指定终端
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage


def monitor_progress(tty="/dev/pts/0", interval=5):
    """
    监控下载进度

    Args:
        tty: 终端设备
        interval: 更新间隔（秒）
    """
    storage = SQLiteStorage()
    total = 4838
    last_count = 0

    try:
        tty_file = open(tty, 'w')
    except:
        print(f"无法打开终端: {tty}")
        return

    print(f"开始监控下载进度（每{interval}秒更新一次）...")
    print(f"进度将显示在: {tty}")
    print("按 Ctrl+C 停止监控\n")

    try:
        while True:
            stats = storage.get_stats()
            current = stats['stock_count']
            progress = current / total * 100
            remaining = total - current

            # 计算速度
            if last_count > 0:
                speed = (current - last_count) / interval
                eta = remaining / speed / 60 if speed > 0 else 0
            else:
                speed = 0
                eta = 0

            last_count = current

            # 构建进度条
            bar_width = 40
            filled = int(bar_width * current / total)
            bar = "█" * filled + "░" * (bar_width - filled)

            # 格式化消息
            msg = f"""
\033[2J\033[H
╔══════════════════════════════════════════════════════════════════════════════╗
║                      A股数据下载 - 实时进度监控                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  进度条: [{bar}]
║  完成: {current:5d}/{total:5d} ({progress:6.2f}%)
║  速度: {speed:5.2f}只/秒    预计剩余: {eta:5.1f}分钟
╠══════════════════════════════════════════════════════════════════════════════╣
║  数据库大小: {stats['db_size_mb']:7.2f} MB
║  股票数量:   {stats['stock_count']:5d}
║  数据行数:   {stats['total_rows']:8,}
║  日期范围:   {stats['date_range']}
╚══════════════════════════════════════════════════════════════════════════════╝
最后更新: {time.strftime('%H:%M:%S')}
"""

            # 写入到tty
            tty_file.write(msg)
            tty_file.flush()

            # 检查是否完成
            if current >= total:
                print(f"\n下载完成！共 {current} 只股票")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n监控已停止")
    finally:
        tty_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="监控下载进度")
    parser.add_argument("--tty", default="/dev/pts/0", help="输出终端")
    parser.add_argument("--interval", type=int, default=5, help="更新间隔（秒）")

    args = parser.parse_args()

    monitor_progress(tty=args.tty, interval=args.interval)
