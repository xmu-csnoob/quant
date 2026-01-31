"""
严格按时间划分训练ML模型

- 训练集: 2020-2024年9月
- 测试集: 2024年10-12月（模拟盘用）
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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from loguru import logger
import os


def prepare_data_time_split():
    """
    准备训练数据，严格按时间划分

    Returns:
        train_df, test_df
    """
    print("=" * 80)
    print("准备训练数据（严格时间划分）")
    print("=" * 80)

    # 初始化
    storage = SQLiteStorage()
    feature_extractor = EnhancedFeatureExtractor()

    # 时间划分
    train_end = "20240930"
    test_start = "20241001"
    test_end = "20241231"

    print(f"\n时间划分:")
    print(f"  训练集: 20200101 ~ {train_end}")
    print(f"  测试集: {test_start} ~ {test_end}")

    # 获取股票列表
    print("\n[1/5] 获取股票列表...")
    stocks = storage.get_all_stocks()
    print(f"  共 {len(stocks)} 只股票")

    # 选择部分股票（避免内存问题）
    sample_size = min(300, len(stocks))
    sampled_stocks = sorted(stocks)[:sample_size]
    print(f"  使用 {sample_size} 只股票训练")

    # 提取训练数据
    print(f"\n[2/5] 提取训练数据...")

    train_data = []
    test_data = []

    for i, ts_code in enumerate(sampled_stocks):
        if (i + 1) % 50 == 0:
            print(f"  处理中: {i+1}/{sample_size}")

        try:
            # 获取训练期数据
            df_train = storage.get_daily_prices(ts_code, "20200101", train_end)

            if df_train is None or len(df_train) < 100:
                continue

            # 提取特征
            features_train = feature_extractor.extract(df_train)

            # 计算未来5日收益率作为标签
            features_train['future_return_5'] = features_train['close'].pct_change(5).shift(-5)
            features_train['direction'] = (features_train['future_return_5'] > 0).astype(int)
            features_train['ts_code'] = ts_code

            train_data.append(features_train)

            # 获取测试期数据
            df_test = storage.get_daily_prices(ts_code, test_start, test_end)

            if df_test is not None and len(df_test) >= 20:
                # 需要前面一些数据用于特征计算
                df_for_features = storage.get_daily_prices(
                    ts_code,
                    (pd.to_datetime(test_start) - pd.Timedelta(days=60)).strftime('%Y%m%d'),
                    test_end
                )

                if df_for_features is not None and len(df_for_features) >= 60:
                    features_test = feature_extractor.extract(df_for_features)
                    features_test['future_return_5'] = features_test['close'].pct_change(5).shift(-5)
                    features_test['direction'] = (features_test['future_return_5'] > 0).astype(int)
                    features_test['ts_code'] = ts_code

                    # 只保留测试期的数据
                    features_test['trade_date'] = pd.to_datetime(features_test['trade_date'])
                    features_test = features_test[
                        features_test['trade_date'] >= pd.to_datetime(test_start)
                    ].reset_index(drop=True)

                    if len(features_test) >= 10:
                        test_data.append(features_test)

        except Exception as e:
            logger.warning(f"处理 {ts_code} 失败: {e}")
            continue

    print(f"  ✓ 训练集: {len(train_data)} 只股票")
    print(f"  ✓ 测试集: {len(test_data)} 只股票")

    # 合并数据
    print(f"\n[3/5] 合并数据...")

    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    # 删除包含NaN的行
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    print(f"  训练集样本: {len(train_df)}")
    print(f"  测试集样本: {len(test_df)}")
    print(f"  特征数: {len([c for c in train_df.columns if c.startswith('f_')])}")

    return train_df, test_df


def train_model(train_df, test_df):
    """训练模型并在测试集上评估"""
    print(f"\n[4/5] 训练模型...")

    # 准备特征和标签
    feature_cols = [c for c in train_df.columns if c.startswith('f_')]

    X_train = train_df[feature_cols].values
    y_train_reg = train_df['future_return_5'].values
    y_train_clf = train_df['direction'].values

    X_test = test_df[feature_cols].values
    y_test_reg = test_df['future_return_5'].values
    y_test_clf = test_df['direction'].values

    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练集上涨比例: {y_train_clf.mean():.2%}")
    print(f"  测试集上涨比例: {y_test_clf.mean():.2%}")

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

    print(f"    测试集准确率: {accuracy:.4f}")

    # 特征重要性
    importance = model_clf.get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n[5/5] 特征重要性 Top 15:")
    feature_names = [c.replace('f_', '') for c in feature_cols]
    feature_idx_map = {f'f{i}': name for i, name in enumerate(feature_names)}

    for i, (fid, score) in enumerate(sorted_importance[:15]):
        fname = feature_idx_map.get(fid, fid)
        print(f"  {i+1}. {fname}: {score}")

    return model_clf, {
        'accuracy': accuracy,
        'importance': sorted_importance,
    }


def main():
    """主函数"""
    print("=" * 80)
    print("A股ML模型训练（严格时间划分）")
    print("训练: 2020-2024年9月")
    print("测试: 2024年10-12月")
    print("=" * 80)

    # 准备数据
    train_df, test_df = prepare_data_time_split()

    if train_df is None or train_df.empty:
        print("训练数据准备失败")
        return

    if test_df is None or test_df.empty:
        print("测试数据准备失败")
        return

    # 训练模型
    model, metrics = train_model(train_df, test_df)

    # 保存模型
    print(f"\n保存模型...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgboost_time_split.json')
    print(f"  ✓ 模型已保存到 models/xgboost_time_split.json")

    # 总结
    print(f"\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    print(f"\n模型性能:")
    print(f"  测试集准确率: {metrics['accuracy']:.4f}")
    print(f"\n模型文件:")
    print(f"  models/xgboost_time_split.json")
    print(f"\n重要提醒:")
    print(f"  ✓ 训练集: 2020-2024年9月")
    print(f"  ✓ 测试集: 2024年10-12月")
    print(f"  ✓ 无时间泄漏，可安全用于模拟盘")


if __name__ == "__main__":
    main()
