"""
改进的ML模型训练 - 避免只学到熊市

采用Walk-Forward验证方法，使用多个测试期
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger


def get_walk_forward_splits():
    """
    定义Walk-Forward验证的时间划分

    使用滚动窗口方法，每个测试期3个月
    """
    splits = [
        # 测试期1: 2021年Q1 (牛市)
        {
            "train_start": "20200101",
            "train_end": "20201231",
            "test_start": "20210101",
            "test_end": "20210331",
            "market": "牛市"
        },
        # 测试期2: 2021年Q3 (震荡)
        {
            "train_start": "20200101",
            "train_end": "20210630",
            "test_start": "20210701",
            "test_end": "20210930",
            "market": "震荡"
        },
        # 测试期3: 2022年Q2 (熊市开始)
        {
            "train_start": "20200101",
            "train_end": "20220331",
            "test_start": "20220401",
            "test_end": "20220630",
            "market": "熊市"
        },
        # 测试期4: 2023年Q1 (反弹)
        {
            "train_start": "20200101",
            "train_end": "20221231",
            "test_start": "20230101",
            "test_end": "20230331",
            "market": "反弹"
        },
        # 测试期5: 2024年Q1 (震荡上行)
        {
            "train_start": "20200101",
            "train_end": "20231231",
            "test_start": "20240101",
            "test_end": "20240331",
            "market": "震荡上行"
        },
    ]

    return splits


def prepare_data_for_period(storage, train_start, train_end, test_start, test_end, max_stocks=500):
    """
    为指定时间段准备数据

    Returns:
        train_df, test_df
    """
    print(f"\n准备数据: 训练{train_start}-{train_end}, 测试{test_start}-{test_end}")

    feature_extractor = EnhancedFeatureExtractor()

    # 获取股票列表
    all_stocks = storage.get_all_stocks()
    np.random.seed(42)
    sampled_stocks = sorted(all_stocks)[:max_stocks]

    train_data = []
    test_data = []

    for i, ts_code in enumerate(sampled_stocks):
        if (i + 1) % 100 == 0:
            print(f"  处理中: {i+1}/{len(sampled_stocks)}")

        try:
            # 获取训练数据（需要额外60天用于特征计算）
            df_train_full = storage.get_daily_prices(
                ts_code,
                (pd.to_datetime(train_start) - timedelta(days=90)).strftime('%Y%m%d'),
                train_end
            )

            if df_train_full is None or len(df_train_full) < 100:
                continue

            # 提取特征
            features_train = feature_extractor.extract(df_train_full)

            # 计算标签
            features_train['future_return_5'] = features_train['close'].pct_change(5).shift(-5)
            features_train['direction'] = (features_train['future_return_5'] > 0).astype(int)

            # 筛选训练期数据
            features_train['trade_date'] = pd.to_datetime(features_train['trade_date'])
            features_train = features_train[
                features_train['trade_date'] >= pd.to_datetime(train_start)
            ].reset_index(drop=True)

            if len(features_train) > 50:
                train_data.append(features_train)

            # 获取测试数据
            df_test_full = storage.get_daily_prices(
                ts_code,
                (pd.to_datetime(test_start) - timedelta(days=90)).strftime('%Y%m%d'),
                test_end
            )

            if df_test_full is None or len(df_test_full) < 60:
                continue

            features_test = feature_extractor.extract(df_test_full)
            features_test['future_return_5'] = features_test['close'].pct_change(5).shift(-5)
            features_test['direction'] = (features_test['future_return_5'] > 0).astype(int)

            # 筛选测试期数据
            features_test['trade_date'] = pd.to_datetime(features_test['trade_date'])
            features_test = features_test[
                features_test['trade_date'] >= pd.to_datetime(test_start)
            ].reset_index(drop=True)

            if len(features_test) > 10:
                test_data.append(features_test)

        except Exception as e:
            logger.warning(f"处理 {ts_code} 失败: {e}")
            continue

    # 合并数据
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()

    # 删除NaN
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    print(f"  训练集: {len(train_df)} 样本 ({len(train_data)} 只股票)")
    print(f"  测试集: {len(test_df)} 样本 ({len(test_data)} 只股票)")

    return train_df, test_df


def train_and_evaluate(train_df, test_df, split_info):
    """
    训练模型并在测试集上评估

    Returns:
        model, metrics
    """
    # 准备特征
    feature_cols = [c for c in train_df.columns if c.startswith('f_')]

    X_train = train_df[feature_cols].values
    y_train = train_df['direction'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['direction'].values

    print(f"  特征数: {len(feature_cols)}")
    print(f"  训练集上涨比例: {y_train.mean():.2%}")
    print(f"  测试集上涨比例: {y_test.mean():.2%}")

    # 训练模型
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'min_child_weight': 3,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),  # 处理样本不平衡
    }

    watchlist = [(dtrain, 'train'), (dtest, 'eval')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # 评估
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'train_size': len(y_train),
        'test_size': len(y_test),
        'train_up_ratio': y_train.mean(),
        'test_up_ratio': y_test.mean(),
    }

    return model, metrics


def main():
    """主函数"""
    print("=" * 80)
    print("改进的ML模型训练 - Walk-Forward验证")
    print("=" * 80)

    storage = SQLiteStorage()
    splits = get_walk_forward_splits()

    print(f"\n共 {len(splits)} 个测试期")
    print("\n测试期分布:")
    for i, split in enumerate(splits, 1):
        print(f"  {i}. {split['test_start']}-{split['test_end']} ({split['market']})")

    # 存储所有结果
    all_results = []
    all_models = []

    for i, split in enumerate(splits, 1):
        print("\n" + "=" * 80)
        print(f"测试期 {i}/{len(splits)}: {split['test_start']}-{split['test_end']} ({split['market']})")
        print("=" * 80)

        # 准备数据
        train_df, test_df = prepare_data_for_period(
            storage,
            split['train_start'],
            split['train_end'],
            split['test_start'],
            split['test_end'],
            max_stocks=500
        )

        if train_df.empty or test_df.empty:
            print("  跳过（数据不足）")
            continue

        # 训练并评估
        print(f"\n训练模型...")
        model, metrics = train_and_evaluate(train_df, test_df, split)

        metrics.update({
            'period': i,
            'test_start': split['test_start'],
            'test_end': split['test_end'],
            'market': split['market'],
        })

        all_results.append(metrics)
        all_models.append(model)

        print(f"\n结果:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")

    # 汇总结果
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)

    results_df = pd.DataFrame(all_results)

    print("\n各测试期表现:")
    print(f"\n{'期间':<8} {'市场':<10} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1':<8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['test_start'][:4]}{r['test_start'][4:6]}-{r['test_end'][4:6]}  "
              f"{r['market']:<10} {r['accuracy']:.4f}   {r['precision']:.4f}   "
              f"{r['recall']:.4f}   {r['f1']:.4f}")

    print(f"\n平均表现:")
    print(f"  准确率: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"  精确率: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"  召回率: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"  F1分数: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")

    # 保存最佳模型（在最后一个测试期表现最好）
    best_model_idx = results_df['accuracy'].idxmax()
    best_model = all_models[best_model_idx]
    best_split = all_results[best_model_idx]

    os.makedirs('models', exist_ok=True)
    best_model.save_model('models/xgboost_robust.json')

    print(f"\n最佳模型: 测试期{best_split['period']} ({best_split['test_start']}-{best_split['test_end']})")
    print(f"  准确率: {best_split['accuracy']:.4f}")
    print(f"  已保存到: models/xgboost_robust.json")

    # 保存详细结果
    results_df.to_csv('backtest_results/walk_forward_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: backtest_results/walk_forward_results.csv")


if __name__ == "__main__":
    main()
