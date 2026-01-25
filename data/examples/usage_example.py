"""
数据模块使用示例

演示如何使用数据模块获取 Mock 数据和真实数据
"""

import sys
sys.path.insert(0, "/home/wangwenfei/quant")

from loguru import logger
from data.fetchers.mock import MockDataFetcher
from data.fetchers.tushare import TushareDataFetcher
from data.fetchers.base import Exchange
from data.storage.storage import DataStorage
from data.api.data_manager import DataManager


# 配置日志
logger.add(
    "logs/data_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)


def example_1_use_mock_data():
    """示例 1: 使用 Mock 数据（开发阶段）"""
    print("\n" + "=" * 80)
    print("示例 1: 使用 Mock 数据")
    print("=" * 80)

    # 1. 创建 Mock 数据获取器
    fetcher = MockDataFetcher(scenario="normal")

    # 2. 创建存储
    storage = DataStorage(base_path="data/raw")

    # 3. 创建数据管理器
    manager = DataManager(fetcher=fetcher, storage=storage, cache_size=10)

    # 4. 获取股票列表
    print("\n1. 获取股票列表:")
    stock_list = manager.get_stock_list(Exchange.SSE)
    print(f"   上交所股票数量: {len(stock_list)}")
    print(stock_list.head(3))

    # 5. 获取单只股票数据
    print("\n2. 获取单只股票数据:")
    df = manager.get_daily_price("600000.SH", "20230101", "20230331")
    print(f"   浦发银行数据行数: {len(df)}")
    print(df.head())

    # 6. 批量下载（下载上交所所有股票）
    print("\n3. 批量下载:")
    manager.fetch_and_store(Exchange.SSE, "20230101", "20230131")

    # 7. 查看缓存统计
    print("\n4. 缓存统计:")
    stats = manager.get_cache_stats()
    print(f"   缓存大小: {stats['size']}/{stats['capacity']}")
    print(f"   命中率: {stats['hit_rate']:.2%}")

    print("\n✅ 示例 1 完成")


def example_2_test_different_scenarios():
    """示例 2: 测试不同市场场景"""
    print("\n" + "=" * 80)
    print("示例 2: 测试不同市场场景")
    print("=" * 80)

    fetcher = MockDataFetcher()
    storage = DataStorage(base_path="data/raw")
    manager = DataManager(fetcher=fetcher, storage=storage)

    scenarios = {
        "normal": "正常市场",
        "bull": "牛市",
        "bear": "熊市",
        "sideways": "横盘",
        "volatile": "高波动"
    }

    for scenario, name in scenarios.items():
        print(f"\n{name} ({scenario}):")

        # 切换场景
        fetcher.set_scenario(scenario)

        # 获取数据
        df = manager.get_daily_price("600000.SH", "20230101", "20231231")

        if not df.empty:
            # 计算收益率
            initial_price = df["close"].iloc[0]
            final_price = df["close"].iloc[-1]
            total_return = (final_price - initial_price) / initial_price * 100

            print(f"   初始价格: {initial_price:.2f}")
            print(f"   最终价格: {final_price:.2f}")
            print(f"   总收益率: {total_return:.2f}%")

    print("\n✅ 示例 2 完成")


def example_3_use_real_data():
    """示例 3: 使用真实 Tushare 数据"""
    print("\n" + "=" * 80)
    print("示例 3: 使用真实 Tushare 数据")
    print("=" * 80)

    import os

    # 检查是否设置了 Token
    token = os.getenv("TUSHARE_TOKEN")

    if not token:
        print("\n⚠️  未设置 TUSHARE_TOKEN 环境变量")
        print("   请先申请 Token: https://tushare.pro/register")
        print("   然后设置环境变量: export TUSHARE_TOKEN=your_token")
        return

    try:
        # 1. 创建 Tushare 数据获取器
        fetcher = TushareDataFetcher(token=token)

        # 2. 创建存储和管理器
        storage = DataStorage(base_path="data/raw")
        manager = DataManager(fetcher=fetcher, storage=storage)

        # 3. 获取真实股票列表
        print("\n1. 获取真实股票列表:")
        stock_list = manager.get_stock_list(Exchange.SSE)
        print(f"   上交所股票数量: {len(stock_list)}")
        print(stock_list.head(3))

        # 4. 获取真实行情数据
        print("\n2. 获取真实行情数据:")
        df = manager.get_daily_price("600000.SH", "20230101", "20230331")
        print(f"   浦发银行数据行数: {len(df)}")
        print(df.head())

        print("\n✅ 示例 3 完成")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("   请检查 Token 是否正确")


def example_4_switch_data_sources():
    """示例 4: 切换数据源"""
    print("\n" + "=" * 80)
    print("示例 4: 切换数据源")
    print("=" * 80)

    # 使用 Mock 数据
    print("\n1. 使用 Mock 数据:")
    mock_fetcher = MockDataFetcher()
    storage = DataStorage(base_path="data/raw")
    manager = DataManager(fetcher=mock_fetcher, storage=storage)

    df_mock = manager.get_daily_price("600000.SH", "20230101", "20230131")
    print(f"   Mock 数据行数: {len(df_mock)}")
    print(f"   平均价格: {df_mock['close'].mean():.2f}")

    # 切换到真实数据（如果有 Token）
    import os
    if os.getenv("TUSHARE_TOKEN"):
        print("\n2. 切换到真实数据:")
        real_fetcher = TushareDataFetcher()
        manager.fetcher = real_fetcher  # 替换数据源

        df_real = manager.get_daily_price("600000.SH", "20230101", "20230131")
        print(f"   真实数据行数: {len(df_real)}")
        print(f"   平均价格: {df_real['close'].mean():.2f}")
    else:
        print("\n2. 未设置 TUSHARE_TOKEN，跳过真实数据示例")

    print("\n✅ 示例 4 完成")


def example_5_batch_download():
    """示例 5: 批量下载全市场数据"""
    print("\n" + "=" * 80)
    print("示例 5: 批量下载")
    print("=" * 80)

    fetcher = MockDataFetcher()
    storage = DataStorage(base_path="data/raw")
    manager = DataManager(fetcher=fetcher, storage=storage)

    print("\n开始批量下载上交所股票数据...")
    print("（演示模式，只下载前 10 只股票）")

    # 批量下载
    manager.fetch_and_store(
        exchange=Exchange.SSE,
        start_date="20230101",
        end_date="20230131",
        force_update=False
    )

    print("\n✅ 批量下载完成")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" 数据模块使用示例")
    print("=" * 80)

    # 运行各个示例
    example_1_use_mock_data()
    example_2_test_different_scenarios()
    example_3_use_real_data()
    example_4_switch_data_sources()
    example_5_batch_download()

    print("\n" + "=" * 80)
    print(" 所有示例运行完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
