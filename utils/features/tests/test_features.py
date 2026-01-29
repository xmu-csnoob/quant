"""
特征提取模块测试
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from utils.features import FeatureBuilder, TechnicalFeatureExtractor
from utils.features.base import BaseFeatureExtractor


class MockFeatureExtractor(BaseFeatureExtractor):
    """用于测试的模拟特征提取器"""

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取简单的测试特征"""
        df["test_feature"] = df["close"] / df["open"]
        self._register_feature("test_feature")
        return df


def create_test_data(n: int = 100) -> pd.DataFrame:
    """创建测试数据"""
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")

    # 生成随机价格数据（带趋势）
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 1, n)
    close = trend + noise

    df = pd.DataFrame(
        {
            "trade_date": dates,
            "open": close + np.random.uniform(-0.5, 0.5, n),
            "high": close + np.random.uniform(0, 1, n),
            "low": close - np.random.uniform(0, 1, n),
            "close": close,
            "volume": np.random.randint(1000000, 10000000, n),
        }
    )

    return df


def test_base_feature_extractor():
    """测试基础特征提取器"""
    print("\n测试基础特征提取器...")

    df = create_test_data(100)
    extractor = MockFeatureExtractor(name="test")

    result = extractor.extract(df)

    assert "test_feature" in result.columns
    assert "test_feature" in extractor.get_feature_names()

    print("✓ 基础特征提取器测试通过")


def test_technical_feature_extractor():
    """测试技术指标特征提取器"""
    print("\n测试技术指标特征提取器...")

    df = create_test_data(200)
    extractor = TechnicalFeatureExtractor()

    result = extractor.extract(df)

    # 检查是否有关键特征
    expected_features = [
        "MA5",
        "MA5_ratio",
        "DIF",
        "RSI",
        "BB_position",
        "ATR",
        "volume_ratio",
    ]

    for feature in expected_features:
        assert feature in result.columns, f"缺少特征: {feature}"

    feature_count = len(extractor.get_feature_names())
    print(f"✓ 技术指标特征提取器测试通过，提取了 {feature_count} 个特征")


def test_feature_builder():
    """测试特征构建器"""
    print("\n测试特征构建器...")

    df = create_test_data(200)

    # 测试默认构建器
    builder = FeatureBuilder.create_default()
    result = builder.build(df)

    feature_names = builder.get_feature_names()
    assert len(feature_names) > 0

    print(f"✓ 特征构建器测试通过，提取了 {len(feature_names)} 个特征")


def test_feature_matrix():
    """测试特征矩阵提取"""
    print("\n测试特征矩阵提取...")

    df = create_test_data(200)
    builder = FeatureBuilder.create_default()

    # 构建特征
    df_with_features = builder.build(df)

    # 获取特征矩阵
    feature_matrix = builder.get_feature_matrix(df)

    # 检查特征矩阵不包含原始列
    original_columns = ["open", "high", "low", "close", "volume", "trade_date"]
    for col in original_columns:
        assert col not in feature_matrix.columns

    print(f"✓ 特征矩阵测试通过，形状: {feature_matrix.shape}")


def test_custom_extractor():
    """测试自定义特征提取器"""
    print("\n测试自定义特征提取器...")

    df = create_test_data(200)

    # 创建自定义提取器
    custom_extractor = TechnicalFeatureExtractor(ma_periods=[5, 20])

    builder = FeatureBuilder()
    builder.add_extractor(custom_extractor)

    result = builder.build(df)

    # 检查是否有指定的MA
    assert "MA5" in result.columns
    assert "MA20" in result.columns

    # 检查不应该有MA60（因为我们只指定了[5, 20]）
    assert "MA60" not in result.columns

    print("✓ 自定义特征提取器测试通过")


def test_feature_values_range():
    """测试特征值范围是否合理"""
    print("\n测试特征值范围...")

    df = create_test_data(200)
    extractor = TechnicalFeatureExtractor()
    result = extractor.extract(df)

    # RSI应该在0-100之间
    rsi_values = result["RSI"].dropna()
    assert rsi_values.min() >= 0
    assert rsi_values.max() <= 100

    # 布林带位置应该在0-1之间（大部分情况下）
    bb_position = result["BB_position"].dropna()
    assert (bb_position >= -1).all()  # 允许一些异常值
    assert (bb_position <= 2).all()

    # 成交量比率应该为正
    volume_ratio = result["volume_ratio"].dropna()
    assert (volume_ratio > 0).all()

    print("✓ 特征值范围测试通过")


def test_no_data_leakage():
    """测试没有数据泄露"""
    print("\n测试数据泄露...")

    df = create_test_data(200)
    extractor = TechnicalFeatureExtractor()
    result = extractor.extract(df)

    # 检查特征是否只基于历史数据
    # 验证：需要差分的特征，第一行应该是NaN

    # 检查MA斜率（需要至少2天数据）
    # 第一行的斜率应该是NaN（因为需要前一天的数据）
    assert pd.isna(result["MA20_slope"].iloc[0])

    # 检查MACD斜率
    # 第一行应该是NaN
    assert pd.isna(result["MACD_slope"].iloc[0])

    # 检查价格加速度
    # 前两行应该是NaN（二阶差分）
    assert pd.isna(result["price_acceleration"].iloc[0])
    assert pd.isna(result["price_acceleration"].iloc[1])

    print("✓ 数据泄露测试通过，特征只基于历史数据")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("特征提取模块测试")
    print("=" * 70)

    test_base_feature_extractor()
    test_technical_feature_extractor()
    test_feature_builder()
    test_feature_matrix()
    test_custom_extractor()
    test_feature_values_range()
    test_no_data_leakage()

    print("\n" + "=" * 70)
    print("所有测试通过！✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
