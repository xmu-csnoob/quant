"""
下载创业板和科创板数据
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.data.fetchers.tushare import TushareDataFetcher
import time

# 创业板龙头
CHI_NEXT = [
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
    "300122.SZ",  # 智飞生物
    "300271.SZ",  # 华海药业
    "300347.SZ",  # 泰格医药
    "300408.SZ",  # 三环集团
    "300433.SZ",  # 蓝思科技
    "300454.SZ",  # 深信服
    "300476.SZ",  # 胜宏科技
    "300496.SZ",  # 中科创达
    "300558.SZ",  # 贝达药业
    "300676.SZ",  # 华大基因
]

# 科创板龙头
STAR_MARKET = [
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
    "688169.SH",  # 石头科技
    "688599.SH",  # 天合光能
    "688981.SH",  # 中芯国际
    "688363.SH",  # 华熙生物
    "688188.SH",  # 柏楚电子
    "688396.SH",  # 华润微
    "688126.SH",  # 沪硅产业
    "688298.SH",  # 东方生物
    "688122.SH",  # 西部超导
    "688185.SH",  # 康希诺
]

def download_batch(stocks, name):
    """批量下载"""
    print(f"\n下载 {name}...")

    storage = SQLiteStorage()
    fetcher = TushareDataFetcher()

    for i, ts_code in enumerate(stocks, 1):
        print(f"  [{i}/{len(stocks)}] {ts_code}...", end=" ", flush=True)

        try:
            df = fetcher.get_daily_price(ts_code, "20200101", "20241231")

            if df is None or df.empty:
                print("✗ 无数据")
                continue

            rows = storage.save_daily_prices(df, ts_code)
            print(f"✓ {rows} 行")

            time.sleep(0.5)

        except Exception as e:
            print(f"✗ {e}")

    stats = storage.get_stats()
    print(f"\n数据库统计:")
    print(f"  股票数: {stats['stock_count']}")
    print(f"  数据行: {stats['total_rows']:,}")
    print(f"  大小: {stats['db_size_mb']:.2f} MB")


if __name__ == "__main__":
    print("=" * 60)
    print("下载创业板和科创板数据")
    print("=" * 60)

    download_batch(CHI_NEXT, "创业板龙头")
    download_batch(STAR_MARKET, "科创板龙头")

    print("\n完成！")
