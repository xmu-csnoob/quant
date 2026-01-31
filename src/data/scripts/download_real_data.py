"""
下载真实A股历史数据

使用Tushare下载蓝筹股的多年历史数据
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import pandas as pd
from tqdm import tqdm

from src.data.fetchers.tushare import TushareDataFetcher
from src.data.storage.storage import DataStorage
from src.data.fetchers.base import Exchange


def download_blue_chips_historical():
    """下载蓝筹股多年历史数据"""
    print("=" * 70)
    print("下载真实A股历史数据")
    print("=" * 70)

    # 蓝筹股列表
    stocks = [
        # 银行股
        ("600000.SH", Exchange.SSE, "浦发银行"),
        ("600036.SH", Exchange.SSE, "招商银行"),
        ("601398.SH", Exchange.SSE, "工商银行"),
        ("601939.SH", Exchange.SSE, "建设银行"),
        ("601288.SH", Exchange.SSE, "农业银行"),
        # 保险
        ("601318.SH", Exchange.SSE, "中国平安"),
        ("601601.SH", Exchange.SSE, "中国太保"),
        ("601336.SH", Exchange.SSE, "新华保险"),
        # 科技
        ("688981.SH", Exchange.SSE, "中芯国际"),
        ("688009.SH", Exchange.SSE, "澜起科技"),
        # 消费
        ("600519.SH", Exchange.SSE, "贵州茅台"),
        ("600887.SH", Exchange.SSE, "伊利股份"),
        ("600809.SH", Exchange.SSE, "山西汾酒"),
        # 地产
        ("000002.SZ", Exchange.SZSE, "万科A"),
        ("000001.SZ", Exchange.SZSE, "平安银行"),
        ("000063.SZ", Exchange.SZSE, "中兴通讯"),
        ("000858.SZ", Exchange.SZSE, "五粮液"),
        ("002415.SZ", Exchange.SZSE, "海康威视"),
        # 新能源
        ("300750.SZ", Exchange.SZSE, "宁德时代"),
        ("002594.SZ", Exchange.SZSE, "比亚迪"),
        ("300014.SZ", Exchange.SZSE, "亿纬锂能"),
        # 医药
        ("000661.SZ", Exchange.SZSE, "长春高新"),
        ("300760.SZ", Exchange.SZSE, "迈瑞医疗"),
        ("300347.SZ", Exchange.SZSE, "泰格医药"),
    ]

    print(f"计划下载 {len(stocks)} 只股票")
    print(f"时间范围: 2019-2024 (6年)")
    print(f"预计数据量: {len(stocks)} × 6年 × 250天 ≈ {len(stocks) * 6 * 250:,} 条")
    print()

    # 检查Token
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("❌ TUSHARE_TOKEN 环境变量未设置")
        print("   请设置: export TUSHARE_TOKEN=your_token")
        return

    # 初始化
    fetcher = TushareDataFetcher()
    storage = DataStorage()

    results = []

    # 下载每只股票
    for stock_code, exchange, name in stocks:
        print(f"\n下载 {stock_code} ({name})...")

        all_data = []
        success_years = []

        try:
            for year in range(2019, 2025):
                start_date = f"{year}0101"
                end_date = f"{year}1231"

                try:
                    # 使用fetcher获取数据
                    df = fetcher.get_daily_price(stock_code, start_date, end_date)

                    if df is not None and len(df) > 0:
                        all_data.append(df)
                        success_years.append(year)
                        print(f"  {year}年: {len(df)} 条")
                    else:
                        print(f"  {year}年: 无数据")

                except Exception as e:
                    print(f"  {year}年: 失败 - {e}")
                    # 等待一下，避免频率限制
                    import time
                    time.sleep(1)

            if all_data:
                # 合并多年数据
                df_all = pd.concat(all_data, ignore_index=True)
                df_all = df_all.sort_values("trade_date").reset_index(drop=True)

                # 保存
                storage.save_daily_price(df_all, stock_code, exchange)

                total_count = len(df_all)
                years = len(success_years)

                print(f"  ✓ 总计: {total_count} 条 ({years} 年)")

                results.append({
                    "stock": stock_code,
                    "name": name,
                    "count": total_count,
                    "years": years,
                    "start": df_all.iloc[0]["trade_date"],
                    "end": df_all.iloc[-1]["trade_date"],
                })

        except Exception as e:
            print(f"  ✗ 下载失败: {e}")

    # 总结
    print("\n" + "=" * 70)
    print("下载完成")
    print("=" * 70)

    if results:
        print(f"成功下载: {len(results)}/{len(stocks)} 只")
        print()

        total_count = sum(r["count"] for r in results)
        print(f"总数据量: {total_count:,} 条")
        print()

        print("详情：")
        print("-" * 70)
        print(f"{'股票代码':<12} {'名称':<12} {'数据量':<10} {'年数':<6} {'起始日期':<12} {'结束日期':<12}")
        print("-" * 70)

        for r in results:
            print(f"{r['stock']:<12} {r['name']:<12} {r['count']:>10,} {r['years']:>6} {r['start']:<12} {r['end']:<12}")

    else:
        print("❌ 没有成功下载任何数据")


def download_index_data():
    """下载指数数据"""
    print("\n" + "=" * 70)
    print("下载指数数据")
    print("=" * 70)

    indices = [
        ("000300.SH", Exchange.SSE, "沪深300"),
        ("000905.SH", Exchange.SSE, "中证500"),
        ("399001.SZ", Exchange.SZSE, "深证成指"),
        ("399006.SZ", Exchange.SZSE, "创业板指"),
    ]

    print(f"计划下载 {len(indices)} 个指数")
    print(f"时间范围: 2019-2024")
    print()

    # 初始化
    fetcher = TushareDataFetcher()
    storage = DataStorage()

    results = []

    for index_code, exchange, name in indices:
        print(f"\n下载 {index_code} ({name})...")

        all_data = []

        try:
            for year in range(2019, 2025):
                start_date = f"{year}0101"
                end_date = f"{year}1231"

                try:
                    df = fetcher.get_daily_price(index_code, start_date, end_date)

                    if df is not None and len(df) > 0:
                        all_data.append(df)
                        print(f"  {year}年: {len(df)} 条")
                    else:
                        print(f"  {year}年: 无数据")

                except Exception as e:
                    print(f"  {year}年: 失败 - {e}")
                    import time
                    time.sleep(1)

            if all_data:
                df_all = pd.concat(all_data, ignore_index=True)
                df_all = df_all.sort_values("trade_date").reset_index(drop=True)

                storage.save_daily_price(df_all, index_code, exchange)

                print(f"  ✓ 总计: {len(df_all)} 条")

                results.append({
                    "stock": index_code,
                    "name": name,
                    "count": len(df_all),
                })

        except Exception as e:
            print(f"  ✗ 下载失败: {e}")

    print("\n" + "=" * 70)
    print(f"指数下载完成: {len(results)}/{len(indices)} 个")
    print("=" * 70)


if __name__ == "__main__":
    # 下载蓝筹股数据
    download_blue_chips_historical()

    # 下载指数数据
    download_index_data()
