"""
特征提取模块使用示例

演示如何使用特征提取模块从OHLCV数据中提取特征
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data.fetchers.mock import MockDataFetcher
from src.data.storage.storage import DataStorage
from src.data.api.data_manager import DataManager

from src.utils.features import FeatureBuilder, TechnicalFeatureExtractor


def main():
    """主函数"""
    print("=" * 70)
    print("特征提取模块使用示例")
    print("=" * 70)
    print()

    # 步骤1：获取数据
    print("步骤1：获取数据")
    print("-" * 70)

    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)

    df = manager.get_daily_price("600000.SH", "20230101", "20231231")

    print(f"原始数据: {len(df)} 条")
    print(f"日期范围: {df.iloc[0]['trade_date']} ~ {df.iloc[-1]['trade_date']}")
    print(f"原始列: {list(df.columns)}")
    print()

    # 步骤2：使用默认特征构建器
    print("步骤2：使用默认特征构建器提取特征")
    print("-" * 70)

    builder = FeatureBuilder.create_default()
    df_with_features = builder.build(df)

    print(f"提取后列数: {len(df_with_features.columns)}")
    print(f"特征数量: {len(builder.get_feature_names())}")
    print()

    # 步骤3：查看提取的特征
    print("步骤3：提取的特征列表")
    print("-" * 70)

    feature_names = sorted(builder.get_feature_names())
    print(f"共提取 {len(feature_names)} 个特征：")
    for i, feature in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feature}")
    print()

    # 步骤4：查看特征数据
    print("步骤4：查看特征数据（最近5天）")
    print("-" * 70)

    # 选择一些关键特征展示
    key_features = [
        "trade_date",
        "close",
        "MA5_ratio",
        "MA20_slope",
        "DIF",
        "RSI",
        "BB_position",
        "ATR_ratio",
        "volume_ratio",
    ]

    display_cols = [col for col in key_features if col in df_with_features.columns]
    print(df_with_features[display_cols].tail().to_string(index=False))
    print()

    # 步骤5：获取特征矩阵（只包含特征列）
    print("步骤5：获取特征矩阵（只包含特征列）")
    print("-" * 70)

    feature_matrix = builder.get_feature_matrix(df)
    print(f"特征矩阵形状: {feature_matrix.shape}")
    print(f"特征矩阵列: {list(feature_matrix.columns[:10])}...（共{len(feature_matrix.columns)}列）")
    print()

    # 步骤6：自定义特征提取器
    print("步骤6：自定义特征提取器")
    print("-" * 70)

    custom_extractor = TechnicalFeatureExtractor(
        ma_periods=[5, 20, 60],  # 只用3个MA
        rsi_period=14,
        bb_period=20,
    )

    custom_builder = FeatureBuilder()
    custom_builder.add_extractor(custom_extractor)

    df_custom = custom_builder.build(df)

    print(f"自定义提取器特征数: {len(custom_builder.get_feature_names())}")
    print(f"自定义特征: {sorted(custom_builder.get_feature_names())}")
    print()

    # 步骤7：特征统计
    print("步骤7：特征统计信息")
    print("-" * 70)

    print("主要特征的统计信息：")
    stats_features = [
        "MA5_ratio",
        "DIF",
        "RSI",
        "BB_position",
        "ATR_ratio",
        "volume_ratio",
    ]

    stats_df = df_with_features[stats_features].describe()
    print(stats_df.to_string())
    print()

    # 步骤8：特征缺失值检查
    print("步骤8：特征缺失值检查")
    print("-" * 70)

    missing = df_with_features[builder.get_feature_names()].isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        print("包含缺失值的特征：")
        for feature, count in missing.items():
            print(f"  {feature}: {count} 个缺失值 ({count/len(df)*100:.1f}%)")
    else:
        print("✓ 所有特征都没有缺失值")
    print()

    print("=" * 70)
    print("特征提取示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
