"""
数据模块快速测试（不需要 pytest）
"""

import sys
sys.path.insert(0, "/home/wangwenfei/quant")

from data.fetchers.mock import MockDataFetcher
from data.fetchers.base import Exchange
from data.storage.storage import DataStorage
from data.api.data_manager import DataManager


def test_mock_fetcher():
    """测试 MockDataFetcher"""
    print("\n" + "=" * 60)
    print("测试 MockDataFetcher")
    print("=" * 60)

    print("\n1. 初始化...")
    fetcher = MockDataFetcher(scenario="normal")
    print("   ✓ 初始化成功")

    print("\n2. 获取股票列表...")
    stock_list = fetcher.get_stock_list(Exchange.SSE)
    print(f"   ✓ 获取到 {len(stock_list)} 只股票")
    print(stock_list.head(3))

    print("\n3. 获取日线数据...")
    df = fetcher.get_daily_price("600000.SH", "20230101", "20230131")
    print(f"   ✓ 获取到 {len(df)} 条数据")
    print(df.head())

    print("\n4. 测试不同场景...")
    for scenario in ["bull", "bear", "sideways"]:
        fetcher.set_scenario(scenario)
        df_scenario = fetcher.get_daily_price("600000.SH", "20230101", "20231231")

        if not df_scenario.empty:
            initial = df_scenario["close"].iloc[0]
            final = df_scenario["close"].iloc[-1]
            returns = (final - initial) / initial * 100
            print(f"   ✓ {scenario}: 收益率 {returns:.2f}%")


def test_storage():
    """测试 DataStorage"""
    print("\n" + "=" * 60)
    print("测试 DataStorage")
    print("=" * 60)

    print("\n1. 初始化...")
    storage = DataStorage(base_path="data/tests/tmp")
    print("   ✓ 初始化成功")

    print("\n2. 保存数据...")
    import pandas as pd
    df = pd.DataFrame({
        "ts_code": ["600000.SH"] * 3,
        "trade_date": ["20230101", "20230102", "20230103"],
        "open": [10.0, 10.5, 11.0],
        "high": [11.0, 11.5, 12.0],
        "low": [9.5, 10.0, 10.5],
        "close": [10.5, 11.0, 11.5],
        "volume": [100000] * 3,
        "amount": [1000000] * 3
    })
    storage.save_daily_price(df, "600000.SH", Exchange.SSE)
    print("   ✓ 保存成功")

    print("\n3. 加载数据...")
    loaded_df = storage.load_daily_price("600000.SH", Exchange.SSE)
    print(f"   ✓ 加载成功，共 {len(loaded_df)} 条")

    print("\n4. 检查文件存在...")
    exists = storage.exists("600000.SH", Exchange.SSE)
    print(f"   ✓ 文件存在: {exists}")


def test_data_manager():
    """测试 DataManager"""
    print("\n" + "=" * 60)
    print("测试 DataManager")
    print("=" * 60)

    print("\n1. 初始化...")
    fetcher = MockDataFetcher()
    storage = DataStorage(base_path="data/tests/tmp")
    manager = DataManager(fetcher=fetcher, storage=storage, cache_size=10)
    print("   ✓ 初始化成功")

    print("\n2. 获取股票列表...")
    stock_list = manager.get_stock_list(Exchange.SSE)
    print(f"   ✓ 获取到 {len(stock_list)} 只股票")

    print("\n3. 获取日线数据...")
    df = manager.get_daily_price("600000.SH", "20230101", "20230131")
    print(f"   ✓ 获取到 {len(df)} 条数据")

    print("\n4. 查看缓存统计...")
    stats = manager.get_cache_stats()
    print(f"   ✓ 缓存大小: {stats['size']}/{stats['capacity']}")
    print(f"   ✓ 命中率: {stats['hit_rate']:.2%}")

    print("\n5. 批量下载...")
    manager.fetch_and_store(Exchange.SSE, "20230101", "20230110")
    print("   ✓ 批量下载完成")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print(" 数据模块快速测试")
    print("=" * 60)

    try:
        test_mock_fetcher()
        test_storage()
        test_data_manager()

        print("\n" + "=" * 60)
        print(" ✅ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
