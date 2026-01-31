#!/usr/bin/env python3
"""
使用最新数据训练ML模型

数据划分:
- 训练集: 2024全年 + 2025上半年 (2024-01-01 ~ 2025-06-30)
- 测试集: 2025下半旬 + 2026 (2025-07-01 ~ 2026-12-31)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from loguru import logger


def prepare_data(storage, train_start, train_end, test_start, test_end, max_stocks=1000):
    """
    准备训练和测试数据

    Returns:
        train_df, test_df
    """
    print(f"\n{'='*60}")
    print(f"数据划分")
    print(f"{'='*60}")
    print(f"训练集: {train_start} ~ {train_end}")
    print(f"测试集: {test_start} ~ {test_end}")
    print(f"{'='*60}\n")

    feature_extractor = EnhancedFeatureExtractor()

    # 获取股票列表
    all_stocks = storage.get_all_stocks()
    np.random.seed(42)
    sampled_stocks = sorted(all_stocks)[:max_stocks]

    print(f"股票池: {len(sampled_stocks)} 只\n")

    train_data = []
    test_data = []

    for i, ts_code in enumerate(sampled_stocks):
        if (i + 1) % 100 == 0:
            print(f"  处理中: {i+1}/{len(sampled_stocks)}")

        try:
            # 获取训练数据（需要额外90天用于特征计算）
            df_train_full = storage.get_daily_prices(
                ts_code,
                (pd.to_datetime(train_start) - timedelta(days=120)).strftime('%Y%m%d'),
                train_end
            )

            if df_train_full is None or len(df_train_full) < 100:
                continue

            # 提取特征
            features_train = feature_extractor.extract(df_train_full)

            # 计算标签（5日后涨跌）
            features_train['future_return_5'] = features_train['close'].pct_change(5).shift(-5)
            features_train['direction'] = (features_train['future_return_5'] > 0).astype(int)

            # 筛选训练期数据
            features_train['trade_date'] = pd.to_datetime(features_train['trade_date'])
            features_train = features_train[
                (features_train['trade_date'] >= pd.to_datetime(train_start)) &
                (features_train['trade_date'] <= pd.to_datetime(train_end))
            ].reset_index(drop=True)

            if len(features_train) > 50:
                train_data.append(features_train)

            # 获取测试数据
            df_test_full = storage.get_daily_prices(
                ts_code,
                (pd.to_datetime(test_start) - timedelta(days=120)).strftime('%Y%m%d'),
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
                (features_test['trade_date'] >= pd.to_datetime(test_start)) &
                (features_test['trade_date'] <= pd.to_datetime(test_end))
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

    print(f"\n数据统计:")
    print(f"  训练集: {len(train_df)} 样本 ({len(train_data)} 只股票)")
    print(f"  测试集: {len(test_df)} 样本 ({len(test_data)} 只股票)")
    print(f"  训练集上涨比例: {train_df['direction'].mean():.2%}")
    print(f"  测试集上涨比例: {test_df['direction'].mean():.2%}")

    return train_df, test_df


def train_model(train_df, test_df):
    """
    训练XGBoost模型

    Returns:
        model, metrics
    """
    print(f"\n{'='*60}")
    print(f"训练模型")
    print(f"{'='*60}\n")

    # 准备特征
    feature_cols = [c for c in train_df.columns if c.startswith('f_')]

    X_train = train_df[feature_cols].values
    y_train = train_df['direction'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['direction'].values

    print(f"特征数: {len(feature_cols)}")

    # 训练模型
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'min_child_weight': 3,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    }

    print(f"模型参数:")
    for k, v in params.items():
        if k != 'scale_pos_weight':
            print(f"  {k}: {v}")
    print(f"  scale_pos_weight: {params['scale_pos_weight']:.2f}\n")

    watchlist = [(dtrain, 'train'), (dtest, 'eval')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # 评估
    print(f"\n{'='*60}")
    print(f"模型评估")
    print(f"{'='*60}\n")

    y_pred_prob_train = model.predict(dtrain)
    y_pred_train = (y_pred_prob_train > 0.5).astype(int)

    y_pred_prob_test = model.predict(dtest)
    y_pred_test = (y_pred_prob_test > 0.5).astype(int)

    # 训练集指标
    train_acc = accuracy_score(y_train, y_pred_train)
    train_prec = precision_score(y_train, y_pred_train, zero_division=0)
    train_rec = recall_score(y_train, y_pred_train, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    train_auc = roc_auc_score(y_train, y_pred_prob_train)

    print(f"训练集:")
    print(f"  准确率: {train_acc:.4f}")
    print(f"  精确率: {train_prec:.4f}")
    print(f"  召回率: {train_rec:.4f}")
    print(f"  F1分数: {train_f1:.4f}")
    print(f"  AUC: {train_auc:.4f}")

    # 测试集指标
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_auc = roc_auc_score(y_test, y_pred_prob_test)

    print(f"\n测试集:")
    print(f"  准确率: {test_acc:.4f}")
    print(f"  精确率: {test_prec:.4f}")
    print(f"  召回率: {test_rec:.4f}")
    print(f"  F1分数: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")

    metrics = {
        'train': {
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_rec,
            'f1': train_f1,
            'auc': train_auc,
            'size': len(y_train),
        },
        'test': {
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1,
            'auc': test_auc,
            'size': len(y_test),
        },
    }

    return model, metrics


def backtest_strategy(model, test_df, buy_threshold=0.52, sell_threshold=0.48):
    """
    简单回测策略
    """
    print(f"\n{'='*60}")
    print(f"策略回测 (买入阈值={buy_threshold}, 卖出阈值={sell_threshold})")
    print(f"{'='*60}\n")

    feature_cols = [c for c in test_df.columns if c.startswith('f_')]
    X_test = test_df[feature_cols].values
    y_test = test_df['direction'].values
    returns = test_df['future_return_5'].values

    dtest = xgb.DMatrix(X_test)
    probs = model.predict(dtest)

    # 交易信号
    buy_signals = probs > buy_threshold
    sell_signals = probs < sell_threshold

    # 只在买入信号时持仓
    position_returns = returns[buy_signals]
    hold_returns = returns[probs >= 0.5]

    print(f"信号统计:")
    print(f"  买入信号: {buy_signals.sum()} ({buy_signals.sum()/len(probs)*100:.1f}%)")
    print(f"  卖出信号: {sell_signals.sum()} ({sell_signals.sum()/len(probs)*100:.1f}%)")
    neutral = ((probs >= sell_threshold) & (probs <= buy_threshold)).sum()
    print(f"  观望信号: {neutral}")

    print(f"\n收益统计:")
    print(f"  买入信号平均5日收益: {position_returns.mean()*100:.2f}%")
    print(f"  持有(概率>=0.5)平均5日收益: {hold_returns.mean()*100:.2f}%")
    print(f"  全市场平均5日收益: {returns.mean()*100:.2f}%")

    if len(position_returns) > 0:
        print(f"  买入信号胜率: {(position_returns > 0).sum()/len(position_returns)*100:.1f}%")


def main():
    """主函数"""
    print("="*60)
    print("使用最新数据训练ML模型")
    print("="*60)
    print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    storage = SQLiteStorage()

    # 数据划分 - 扩大训练集范围
    train_start = "20220101"  # 从2022年开始训练
    train_end = "20250630"
    test_start = "20250701"
    test_end = "20261231"

    # 准备数据 - 使用更多股票
    train_df, test_df = prepare_data(
        storage,
        train_start,
        train_end,
        test_start,
        test_end,
        max_stocks=2000
    )

    if train_df.empty or test_df.empty:
        print("\n错误: 数据不足")
        return

    # 训练模型
    model, metrics = train_model(train_df, test_df)

    # 策略回测
    backtest_strategy(model, test_df)

    # 保存模型
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgboost_2022_2026.json')

    print(f"\n{'='*60}")
    print(f"模型已保存到: models/xgboost_2022_2026.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
