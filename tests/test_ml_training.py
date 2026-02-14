"""
ML模型训练流水线测试

测试模型训练、评估和保存功能
"""

import sys
from pathlib import Path
import tempfile
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger

# 使用mock数据测试，避免依赖数据库
from src.data.fetchers.mock import MockDataFetcher
from src.data.storage.storage import DataStorage
from src.data.api.data_manager import DataManager


def test_feature_extraction():
    """测试特征提取"""
    print("=" * 60)
    print("测试特征提取")
    print("=" * 60)

    from src.utils.features.enhanced_features import EnhancedFeatureExtractor

    # 使用mock数据
    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)
    df = manager.get_daily_price("600000.SH", "20230101", "20231231")

    # 提取特征
    extractor = EnhancedFeatureExtractor(prediction_period=5)
    features = extractor.extract(df)

    # 验证
    feature_cols = [c for c in features.columns if c.startswith('f_')]
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  数据行数: {len(features)}")
    print(f"  前5个特征: {feature_cols[:5]}")

    assert len(feature_cols) > 30, "特征数量应该大于30"
    assert len(features) > 100, "数据行数应该大于100"

    print("  ✅ 特征提取测试通过")


def test_model_training():
    """测试模型训练"""
    print("\n" + "=" * 60)
    print("测试模型训练")
    print("=" * 60)

    from scripts.train_ml_model import MLTrainingPipeline
    from src.data.fetchers.mock import MockDataFetcher
    from src.data.storage.storage import DataStorage
    from src.data.api.data_manager import DataManager

    # 准备模拟数据
    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)

    # 获取数据并提取特征
    df = manager.get_daily_price("600000.SH", "20220101", "20231231")

    from src.utils.features.enhanced_features import EnhancedFeatureExtractor
    extractor = EnhancedFeatureExtractor(prediction_period=5)
    features = extractor.extract(df)

    # 计算标签
    features['future_return'] = features['close'].pct_change(5).shift(-5)
    features['label'] = (features['future_return'] > 0).astype(int)

    # 获取特征列
    feature_cols = [c for c in features.columns if c.startswith('f_')]

    # 删除NaN
    data = features.dropna(subset=feature_cols + ['label'])

    # 分割数据
    X = data[feature_cols]
    y = data['label']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 创建训练流水线
    pipeline = MLTrainingPipeline(prediction_period=5)
    pipeline.feature_cols = feature_cols

    # 训练模型（使用较少的迭代次数）
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc'],
        'max_depth': 4,
        'n_estimators': 50,
        'learning_rate': 0.1,
        'seed': 42,
    }

    metrics = pipeline.train(X_train, y_train, X_test, y_test, params=params)

    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    assert 'auc' in metrics, "应该返回AUC指标"
    assert 'accuracy' in metrics, "应该返回accuracy指标"

    print("  ✅ 模型训练测试通过")

    return pipeline


def test_model_save_load():
    """测试模型保存和加载"""
    print("\n" + "=" * 60)
    print("测试模型保存和加载")
    print("=" * 60)

    from scripts.train_ml_model import MLTrainingPipeline

    # 训练一个简单模型
    pipeline = test_model_training()

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    try:
        # 保存模型
        pipeline.save_model(model_dir=temp_dir, version="test_v1")
        print(f"  模型保存到: {temp_dir}")

        # 验证文件存在
        import os
        assert os.path.exists(os.path.join(temp_dir, "xgboost_model.json"))
        assert os.path.exists(os.path.join(temp_dir, "feature_cols.json"))
        assert os.path.exists(os.path.join(temp_dir, "model_info.json"))

        # 加载模型
        pipeline.load_model(os.path.join(temp_dir, "xgboost_model.json"))
        print("  模型加载成功")

        print("  ✅ 模型保存/加载测试通过")

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_feature_extraction()
    test_model_training()
    test_model_save_load()

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)
