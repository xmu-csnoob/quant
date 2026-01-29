"""
生成多年Mock数据

模拟不同市场环境：牛市、熊市、震荡市
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.fetchers.mock import MockDataFetcher
from data.storage.storage import DataStorage


def generate_multi_year_data(
    stock_code: str = "600000.SH",
    start_year: int = 2019,
    end_year: int = 2024,
):
    """
    生成多年Mock数据

    模拟不同市场环境：
    - 2019: 震荡市
    - 2020: 牛市（上半年下跌，下半年上涨）
    - 2021: 疯情后牛市
    - 2022: 熊市
    - 2023: 牛市
    - 2024: 震荡市
    """
    print("=" * 70)
    print("生成多年Mock数据")
    print("=" * 70)
    print(f"股票代码: {stock_code}")
    print(f"时间范围: {start_year} - {end_year}")
    print()

    storage = DataStorage()
    all_data = []

    # 每年的市场场景和参数
    scenarios = {
        2019: {"scenario": "sideways", "base_price": 15.0, "volatility": 0.02},
        2020: {"scenario": "bull", "base_price": 15.0, "volatility": 0.03},
        2021: {"scenario": "bull", "base_price": 20.0, "volatility": 0.025},
        2022: {"scenario": "bear", "base_price": 30.0, "volatility": 0.04},
        2023: {"scenario": "bull", "base_price": 20.0, "volatility": 0.03},
        2024: {"scenario": "sideways", "base_price": 40.0, "volatility": 0.02},
    }

    for year in range(start_year, end_year + 1):
        scenario_config = scenarios.get(year, {"scenario": "sideways", "base_price": 50.0, "volatility": 0.02})

        print(f"{year}年: {scenario_config['scenario']}")
        print(f"  场景: {scenario_config['scenario']}")
        print(f"  基价: {scenario_config['base_price']}")

        # 生成该年数据
        fetcher = MockDataFetcher(
            scenario=scenario_config["scenario"],
            base_price=scenario_config["base_price"],
            volatility=scenario_config["volatility"],
        )

        start_date = f"{year}0101"
        end_date = f"{year}1231"

        try:
            df = fetcher.get_daily_price(stock_code, start_date, end_date)
            if df is not None and len(df) > 0:
                all_data.append(df)
                print(f"  ✓ 生成 {len(df)} 条")
            else:
                print(f"  ✗ 生成失败")
        except Exception as e:
            print(f"  ✗ 错误: {e}")

        print()

    if all_data:
        # 合并所有年份数据
        df_all = pd.concat(all_data, ignore_index=True)
        df_all = df_all.sort_values("trade_date").reset_index(drop=True)

        # 存储到本地
        storage.save_daily_price(df_all)

        print("=" * 70)
        print("生成完成")
        print("=" * 70)
        print(f"总数据量: {len(df_all)} 条")
        print(f"日期范围: {df_all.iloc[0]['trade_date']} ~ {df_all.iloc[-1]['trade_date']}")
        print(f"年份: {df_all['trade_date'].dt.year.nunique()} 年")
        print()

        # 按年份统计
        print("各年份数据量：")
        year_counts = df_all['trade_date'].dt.year.value_counts().sort_index()
        for year, count in year_counts.items():
            scenario = scenarios.get(year, {}).get("scenario", "unknown")
            print(f"  {year}年 ({scenario}): {count} 条")
    else:
        print("❌ 生成失败")


def generate_multi_stock_data():
    """生成多只股票的Mock数据"""
    print("=" * 70)
    print("生成多只股票Mock数据")
    print("=" * 70)

    stock_list = [
        "600000.SH",  # 银行
        "600036.SH",  # 银行
        "600519.SH",  # 白酒
        "600887.SH",  # 乳业
        "601318.SH",  # 保险
        "000001.SZ",  # 平安银行
        "000002.SZ",  # 万科
        "000858.SZ",  # 五粮液
    ]

    print(f"股票数量: {len(stock_list)}")
    print(f"时间范围: 2019-2024（6年）")
    print(f"预计数据量: {len(stock_list)} × 1250 ≈ {len(stock_list) * 1250:,} 条")
    print()

    storage = DataStorage()
    total_count = 0

    for stock_code in stock_list:
        print(f"生成 {stock_code}...")

        # 生成多年数据
        all_data = []
        scenarios = {
            2019: {"scenario": "sideways", "base_price": 15.0 + np.random.randn() * 5, "volatility": 0.02},
            2020: {"scenario": "bull", "base_price": 15.0 + np.random.randn() * 5, "volatility": 0.03},
            2021: {"scenario": "bull", "base_price": 20.0 + np.random.randn() * 5, "volatility": 0.025},
            2022: {"scenario": "bear", "base_price": 30.0 + np.random.randn() * 5, "volatility": 0.04},
            2023: {"scenario": "bull", "base_price": 20.0 + np.random.randn() * 5, "volatility": 0.03},
            2024: {"scenario": "sideways", "base_price": 40.0 + np.random.randn() * 5, "volatility": 0.02},
        }

        for year, config in scenarios.items():
            fetcher = MockDataFetcher(
                scenario=config["scenario"],
                base_price=config["base_price"],
                volatility=config["volatility"],
            )

            start_date = f"{year}0101"
            end_date = f"{year}1231"

            try:
                df = fetcher.get_daily_price(stock_code, start_date, end_date)
                if df is not None and len(df) > 0:
                    all_data.append(df)
            except:
                pass

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            df_all = df_all.sort_values("trade_date").reset_index(drop=True)
            storage.save_daily_price(df_all)

            print(f"  ✓ {len(df_all)} 条")
            total_count += len(df_all)
        else:
            print(f"  ✗ 失败")

    print()
    print("=" * 70)
    print(f"总计: {total_count:,} 条")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "multi_stock":
        # 生成多只股票数据
        generate_multi_stock_data()
    else:
        # 生成单只股票多年数据
        generate_multi_year_data()
