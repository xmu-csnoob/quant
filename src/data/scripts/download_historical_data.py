"""
下载多年A股历史数据

使用Tushare获取真实A股数据
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime, timedelta
from src.data.fetchers.tushare import TushareDataFetcher
from src.data.storage.storage import DataStorage


def download_multi_year_data(
    stock_list: list[str],
    start_year: int = 2020,
    end_year: int = 2024,
):
    """
    下载多年历史数据

    Args:
        stock_list: 股票列表
        start_year: 开始年份
        end_year: 结束年份
    """
    # 检查Token
    import os

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("❌ 请先设置TUSHARE_TOKEN环境变量")
        print("   export TUSHARE_TOKEN=your_token_here")
        return

    print("=" * 70)
    print("下载A股历史数据")
    print("=" * 70)
    print(f"股票数量: {len(stock_list)}")
    print(f"时间范围: {start_year} - {end_year}")
    print(f"预计数据量: {len(stock_list)} × {(end_year - start_year + 1) * 250} ≈ {len(stock_list) * (end_year - start_year + 1) * 250:,} 条")
    print()

    # 初始化
    fetcher = TushareDataFetcher()
    storage = DataStorage()

    total_count = 0
    success_count = 0
    failed_stocks = []

    for stock_code in stock_list:
        try:
            print(f"正在下载 {stock_code}...")

            stock_data = []
            for year in range(start_year, end_year + 1):
                start_date = f"{year}0101"
                end_date = f"{year}1231"

                try:
                    df = fetcher.get_daily_price(stock_code, start_date, end_date)

                    if df is not None and len(df) > 0:
                        stock_data.append(df)
                        print(f"  {year}年: {len(df)} 条")
                    else:
                        print(f"  {year}年: 无数据")

                except Exception as e:
                    print(f"  {year}年: 下载失败 - {e}")

            if stock_data:
                # 合并多年数据
                df_all = pd.concat(stock_data, ignore_index=True)
                df_all = df_all.sort_values("trade_date").reset_index(drop=True)

                # 存储到本地
                storage.save_daily_price(df_all)

                count = len(df_all)
                total_count += count
                success_count += 1

                print(f"✓ {stock_code}: 总计 {count} 条")
            else:
                print(f"✗ {stock_code}: 下载失败")
                failed_stocks.append(stock_code)

        except Exception as e:
            print(f"✗ {stock_code}: 出错 - {e}")
            failed_stocks.append(stock_code)

        print()

    # 总结
    print("=" * 70)
    print("下载完成")
    print("=" * 70)
    print(f"成功: {success_count}/{len(stock_list)}")
    print(f"失败: {len(failed_stocks)}")
    print(f"总数据量: {total_count:,} 条")
    print()

    if failed_stocks:
        print("失败的股票:")
        for stock in failed_stocks:
            print(f"  - {stock}")


def download_index_data():
    """下载指数数据（沪深300、上证50等）"""
    import os

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("❌ 请先设置TUSHARE_TOKEN环境变量")
        return

    print("=" * 70)
    print("下载指数数据")
    print("=" * 70)

    # 主要指数
    indices = [
        "000300.SH",  # 沪深300
        "000016.SH",  # 上证50
        "399001.SZ",  # 深证成指
        "000905.SH",  # 中证500
    ]

    download_multi_year_data(
        stock_list=indices,
        start_year=2020,
        end_year=2024,
    )


def download_blue_chips():
    """下载蓝筹股数据"""
    import os

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("❌ 请先设置TUSHARE_TOKEN环境变量")
        return

    print("=" * 70)
    print("下载蓝筹股数据")
    print("=" * 70)

    # 沪深300成分股（部分）
    blue_chips = [
        "600000.SH",  # 浦发银行
        "600036.SH",  # 招商银行
        "600519.SH",  # 贵州茅台
        "600887.SH",  # 伊利股份
        "601318.SH",  # 中国平安
        "601398.SH",  # 工商银行
        "601857.SH",  # 中国石油
        "601988.SH",  # 中国银行
        "603259.SH",  # 药明康德
        "688981.SH",  # 中芯国际-U
    ]

    download_multi_year_data(
        stock_list=blue_chips,
        start_year=2019,
        end_year=2024,
    )


if __name__ == "__main__":
    import os

    if not os.getenv("TUSHARE_TOKEN"):
        print("请先设置TUSHARE_TOKEN环境变量：")
        print("  export TUSHARE_TOKEN=your_token_here")
        print()
        print("获取Token：")
        print("  1. 访问 https://tushare.pro/register")
        print("  2. 注册账号（免费）")
        print("  3. 获取Token")
        print("  4. 设置环境变量")
    else:
        # 下载蓝筹股数据
        download_blue_chips()

        # 下载指数数据
        download_index_data()
