"""
使用现有A股数据训练ML预测模型

使用下载的1423只股票数据训练XGBoost模型
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor
from src.strategies.base import SignalType
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from loguru import logger


def prepare_data():
    """准备训练数据"""
    print("=" * 80)
    print("准备训练数据")
    print("=" * 80)

    # 初始化
    storage = SQLiteStorage()
    feature_extractor = EnhancedFeatureExtractor()

    # 获取所有股票
    print("\n[1/5] 获取股票列表...")
    stocks = storage.get_all_stocks()
    print(f"  共 {len(stocks)} 只股票")

    # 选择部分股票（避免内存问题）
    sample_size = min(300, len(stocks))
    sampled_stocks = sorted(stocks)[:sample_size]
    print(f"  使用 {sample_size} 只股票训练")

    # 提取特征和标签
    print(f"\n[2/5] 提取特征和标签...")

    all_data = []

    for i, ts_code in enumerate(sampled_stocks):
        if (i + 1) % 50 == 0:
            print(f"  处理中: {i+1}/{sample_size}")

        try:
            # 获取数据
            df = storage.get_daily_prices(ts_code, "20200101", "20241231")

            if df is None or len(df) < 100:
                continue

            # 提取特征
            features = feature_extractor.extract(df)

            # 计算未来5日收益率作为标签
            features['future_return_5'] = features['close'].pct_change(5).shift(-5)
            features['direction'] = (features['future_return_5'] > 0).astype(int)

            # 添加股票代码
            features['ts_code'] = ts_code

            all_data.append(features)

        except Exception as e:
            logger.warning(f"处理 {ts_code} 失败: {e}")
            continue

    print(f"  ✓ 成功处理 {len(all_data)} 只股票")

    # 合并所有数据
    print(f"\n[3/5] 合并数据...")
    df = pd.concat(all_data, ignore_index=True)

    # 删除包含NaN的行
    df = df.dropna()

    print(f"  总样本数: {len(df)}")
    print(f"  特征数: {len([c for c in df.columns if c.startswith('f_')])}")

    return df


def train_model(df):
    """训练模型"""
    print(f"\n[4/5] 训练模型...")

    # 准备特征和标签
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feature_cols].values

    # 回归标签（未来收益率）
    y_reg = df['future_return_5'].values

    # 分类标签（涨跌方向）
    y_clf = df['direction'].values

    print(f"  特征维度: {X.shape}")
    print(f"  样本数: {len(y_reg)}")
    print(f"  上涨比例: {y_clf.mean():.2%}")

    # 划分训练集和测试集
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 训练回归模型（预测收益率）
    print(f"\n  训练回归模型...")
    dtrain_reg = xgb.DMatrix(X_train, label=y_train_reg)
    dtest_reg = xgb.DMatrix(X_test, label=y_test_reg)

    params_reg = {
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'min_child_weight': 3,
    }

    watchlist_reg = [(dtrain_reg, 'train'), (dtest_reg, 'eval')]

    model_reg = xgb.train(
        params_reg,
        dtrain_reg,
        num_boost_round=200,
        evals=watchlist_reg,
        early_stopping_rounds=30,
        verbose_eval=False
    )

    # 评估回归模型
    y_pred_reg = model_reg.predict(dtest_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)

    print(f"    MSE: {mse:.6f}")
    print(f"    R²: {r2:.4f}")

    # 训练分类模型（预测涨跌）
    print(f"\n  训练分类模型...")
    dtrain_clf = xgb.DMatrix(X_train, label=y_train_clf)
    dtest_clf = xgb.DMatrix(X_test, label=y_test_clf)

    params_clf = {
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'min_child_weight': 3,
    }

    watchlist_clf = [(dtrain_clf, 'train'), (dtest_clf, 'eval')]

    model_clf = xgb.train(
        params_clf,
        dtrain_clf,
        num_boost_round=200,
        evals=watchlist_clf,
        early_stopping_rounds=30,
        verbose_eval=False
    )

    # 评估分类模型
    y_pred_prob = model_clf.predict(dtest_clf)
    y_pred_clf = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test_clf, y_pred_clf)

    print(f"    准确率: {accuracy:.4f}")

    # 特征重要性
    importance = model_reg.get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n[5/5] 特征重要性 Top 15:")
    feature_names = [c.replace('f_', '') for c in feature_cols]
    feature_idx_map = {f'f{i}': name for i, name in enumerate(feature_names)}

    for i, (fid, score) in enumerate(sorted_importance[:15]):
        fname = feature_idx_map.get(fid, fid)
        print(f"  {i+1}. {fname}: {score}")

    return model_reg, model_clf, {
        'mse': mse,
        'r2': r2,
        'accuracy': accuracy,
        'importance': sorted_importance,
    }


def main():
    """主函数"""
    print("=" * 80)
    print("A股ML模型训练")
    print("数据: 1700只股票 (2020-2024)")
    print("=" * 80)

    # 准备数据
    df = prepare_data()

    if df is None or df.empty:
        print("数据准备失败")
        return

    # 训练模型
    model_reg, model_clf, metrics = train_model(df)

    # 保存模型
    print(f"\n保存模型...")
    import os
    os.makedirs('models', exist_ok=True)
    model_reg.save_model('models/xgboost_regression.json')
    model_clf.save_model('models/xgboost_classification.json')
    print(f"  ✓ 模型已保存到 models/")

    # 总结
    print(f"\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    print(f"\n模型性能:")
    print(f"  回归 R²: {metrics['r2']:.4f}")
    print(f"  分类准确率: {metrics['accuracy']:.4f}")
    print(f"\n模型文件:")
    print(f"  models/xgboost_regression.json")
    print(f"  models/xgboost_classification.json")


if __name__ == "__main__":
    main()
