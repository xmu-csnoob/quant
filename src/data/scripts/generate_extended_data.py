"""
生成扩展的Mock数据

生成多年、多市场的模拟数据
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.storage.storage import DataStorage
from src.data.fetchers.base import Exchange


def generate_extended_data():
    """生成扩展的Mock数据（多年、多市场环境）"""
    print("=" * 70)
    print("生成扩展Mock数据")
    print("=" * 70)

    storage = DataStorage()
    all_data = []

    # 定义不同年份的市场环境
    # 格式：(年份, 场景, 年化收益率, 波动率)
    market_scenarios = [
        (2019, "sideways", 0.05, 0.02),   # 震荡市，小幅上涨
        (2020, "bull", 0.30, 0.04),       # 牛市
        (2021, "bull", 0.25, 0.025),      # 疫情后牛市
        (2022, "bear", -0.20, 0.05),      # 熊市
        (2023, "bull", 0.35, 0.03),       # 大牛市
        (2024, "sideways", 0.08, 0.02),   # 震荡
    ]

    print(f"计划生成：{len(market_scenarios)} 年数据")
    print(f"股票代码：600000.SH")
    print()

    current_price = 15.0  # 起始价格

    for year, scenario, annual_return, volatility in market_scenarios:
        print(f"{year}年: {scenario}")
        print(f"  预期收益率: {annual_return*100:.0f}%")
        print(f"  波动率: {volatility*100:.1f}%")

        # 生成该年数据
        df_year = generate_year_data(
            year=year,
            scenario=scenario,
            start_price=current_price,
            annual_return=annual_return,
            volatility=volatility,
        )

        if df_year is not None and len(df_year) > 0:
            all_data.append(df_year)
            # 更新下一年起始价格
            current_price = df_year.iloc[-1]["close"]
            print(f"  ✓ 生成 {len(df_year)} 条")
            print(f"  价格: {df_year.iloc[0]['close']:.2f} → {df_year.iloc[-1]['close']:.2f}")
        else:
            print(f"  ✗ 生成失败")

        print()

    if all_data:
        # 合并所有年份数据
        df_all = pd.concat(all_data, ignore_index=True)
        df_all = df_all.sort_values("trade_date").reset_index(drop=True)

        # 存储到本地
        storage.save_daily_price(df_all, "600000.SH", Exchange.SSE)

        print("=" * 70)
        print("生成完成")
        print("=" * 70)
        print(f"总数据量: {len(df_all)} 条")
        print(f"日期范围: {df_all.iloc[0]['trade_date']} ~ {df_all.iloc[-1]['trade_date']}")
        print()

        # 统计
        print("各年份统计：")
        print("-" * 60)
        df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])
        for year, scenario, _, _ in market_scenarios:
            year_data = df_all[df_all["trade_date"].dt.year == year]
            if len(year_data) > 0:
                price_change = (year_data.iloc[-1]["close"] / year_data.iloc[0]["close"] - 1) * 100
                print(f"{year}年 ({scenario}): {len(year_data)} 条, 收益 {price_change:+.1f}%")
    else:
        print("❌ 生成失败")


def generate_year_data(
    year: int,
    scenario: str,
    start_price: float,
    annual_return: float,
    volatility: float,
) -> pd.DataFrame:
    """
    生成一年数据

    Args:
        year: 年份
        scenario: 市场场景
        start_price: 年初价格
        annual_return: 年化收益率
        volatility: 波动率
    """
    # 生成交易日历（简化版，实际应该获取交易日历）
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="B")
    # 去除节假日（简化处理）
    dates = dates[dates.weekday < 5][:250]  # 最多250个交易日

    n = len(dates)

    # 生成价格路径
    np.random.seed(year)  # 每年固定种子，保证可重复

    # 日收益率 = 年化收益率 / 252 + 随机波动
    daily_return = annual_return / 252 + volatility * np.random.randn(n)

    # 累计价格
    prices = start_price * (1 + daily_return).cumprod()

    # 生成OHLCV数据
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # 日内波动
        intraday_vol = close * volatility / np.sqrt(252) * np.abs(np.random.randn())

        open_price = close + np.random.randn() * intraday_vol * 0.5
        high_price = close + abs(np.random.randn() * intraday_vol * 0.5)
        low_price = close - abs(np.random.randn() * intraday_vol * 0.3)

        # 确保high >= close >= low
        high_price = max(high_price, close, open_price)
        low_price = min(low_price, close, open_price)

        # 成交量（随机）
        volume = np.random.randint(1000000, 10000000)

        # 涨跌幅限制
        max_change = 0.20  # 20%涨跌停
        change = (close - open_price) / open_price
        change = np.clip(change, -max_change, max_change)
        close = open_price * (1 + change)
        high_price = max(high_price, close, open_price)
        low_price = min(low_price, close, open_price)

        data.append({
            "ts_code": "600000.SH",
            "trade_date": date.strftime("%Y%m%d"),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close, 2),
            "volume": volume,
            "amount": round(close * volume, 2),
        })

    return pd.DataFrame(data)


def generate_multi_stock_data():
    """生成多只股票数据"""
    print("=" * 70)
    print("生成多只股票数据")
    print("=" * 70)

    stock_list = [
        ("600000.SH", "银行", 0.20),
        ("600036.SH", "银行", 0.15),
        ("600519.SH", "白酒", 0.25),
        ("600887.SH", "乳业", 0.22),
        ("601318.SH", "保险", 0.18),
        ("000001.SZ", "银行", 0.17),
        ("000002.SZ", "地产", 0.12),
        ("000858.SZ", "白酒", 0.23),
    ]

    print(f"股票数量: {len(stock_list)}")
    print(f"时间范围: 2019-2024（6年）")
    print()

    storage = DataStorage()
    total_count = 0

    for stock_code, sector, base_vol in stock_list:
        print(f"生成 {stock_code} ({sector})...")

        all_data = []
        current_price = 15.0 + np.random.randn() * 5  # 随机起始价格

        for year in range(2019, 2025):
            # 根据年份选择场景
            if year in [2020, 2021, 2023]:
                scenario = "bull"
                annual_return = 0.20 + np.random.randn() * 0.1
            elif year == 2022:
                scenario = "bear"
                annual_return = -0.15 + np.random.randn() * 0.1
            else:
                scenario = "sideways"
                annual_return = 0.05 + np.random.randn() * 0.05

            volatility = base_vol + np.random.randn() * 0.05

            df_year = generate_year_data(
                year=year,
                scenario=scenario,
                start_price=current_price,
                annual_return=annual_return,
                volatility=volatility,
            )

            if df_year is not None:
                all_data.append(df_year)
                current_price = df_year.iloc[-1]["close"]

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            df_all = df_all.sort_values("trade_date").reset_index(drop=True)

            # 修改股票代码
            df_all["ts_code"] = stock_code

            # 判断交易所
            exchange = Exchange.SSE if stock_code.endswith(".SH") else Exchange.SZSE

            storage.save_daily_price(df_all, stock_code, exchange)

            count = len(df_all)
            total_count += count
            print(f"  ✓ {count} 条")
        else:
            print(f"  ✗ 失败")

    print()
    print("=" * 70)
    print(f"总计: {total_count:,} 条")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    # 生成多年单只股票数据
    generate_extended_data()

    # 或者生成多只股票数据
    # generate_multi_stock_data()
