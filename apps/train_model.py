#!/usr/bin/env python3
"""
ML模型训练脚本

功能：
1. 从SQLite数据库加载历史数据
2. 提取60+个技术特征
3. 训练XGBoost分类模型（预测5日涨跌方向）
4. 评估模型性能
5. 保存模型

数据划分：
- 训练集: 2021-02-01 ~ 2024-12-31
- 测试集: 2025-01-01 ~ 2026-02-02

使用方法:
    python apps/train_model.py
    python apps/train_model.py --stocks 500  # 限制股票数量
    python apps/train_model.py --train-start 20220101 --train-end 20241231
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from loguru import logger


def prepare_data(
    storage: SQLiteStorage,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    max_stocks: int = 1000,
    prediction_period: int = 5
):
    """
    准备训练和测试数据

    Args:
        storage: 数据存储
        train_start: 训练集开始日期
        train_end: 训练集结束日期
        test_start: 测试集开始日期
        test_end: 测试集结束日期
        max_stocks: 最大股票数量
        prediction_period: 预测周期（天）

    Returns:
        train_df, test_df, feature_cols
    """
    print(f"\n{'='*60}")
    print(f"数据准备")
    print(f"{'='*60}")
    print(f"训练集: {train_start} ~ {train_end}")
    print(f"测试集: {test_start} ~ {test_end}")
    print(f"预测周期: {prediction_period}天")
    print(f"{'='*60}\n")

    feature_extractor = EnhancedFeatureExtractor(prediction_period)

    # 获取股票列表
    all_stocks = storage.get_all_stocks()
    np.random.seed(42)
    np.random.shuffle(all_stocks)
    sampled_stocks = all_stocks[:max_stocks]

    print(f"股票池: {len(sampled_stocks)} / {len(all_stocks)} 只\n")

    train_data = []
    test_data = []
    feature_cols = None

    for i, ts_code in enumerate(sampled_stocks):
        if (i + 1) % 200 == 0:
            print(f"  处理进度: {i+1}/{len(sampled_stocks)} "
                  f"(训练: {sum(len(d) for d in train_data):,}, "
                  f"测试: {sum(len(d) for d in test_data):,})")

        try:
            # 获取完整数据（需要额外数据用于特征计算）
            df_full = storage.get_daily_prices(
                ts_code,
                (pd.to_datetime(train_start) - pd.Timedelta(days=150)).strftime('%Y%m%d'),
                test_end
            )

            if df_full is None or len(df_full) < 200:
                continue

            # 提取特征
            features_df = feature_extractor.extract(df_full)

            # 计算标签（未来N日涨跌）
            features_df['future_return'] = features_df['close'].pct_change(prediction_period).shift(-prediction_period)
            features_df['label'] = (features_df['future_return'] > 0).astype(int)

            # 筛选训练期数据
            features_df['trade_date'] = pd.to_datetime(features_df['trade_date'])
            train_mask = (
                (features_df['trade_date'] >= pd.to_datetime(train_start)) &
                (features_df['trade_date'] <= pd.to_datetime(train_end))
            )
            test_mask = (
                (features_df['trade_date'] >= pd.to_datetime(test_start)) &
                (features_df['trade_date'] <= pd.to_datetime(test_end))
            )

            train_subset = features_df[train_mask].copy()
            test_subset = features_df[test_mask].copy()

            if len(train_subset) > 50:
                train_data.append(train_subset)
            if len(test_subset) > 10:
                test_data.append(test_subset)

            # 记录特征列
            if feature_cols is None:
                feature_cols = [c for c in features_df.columns if c.startswith('f_')]

        except Exception as e:
            logger.debug(f"处理 {ts_code} 跳过: {e}")
            continue

    # 合并数据
    if not train_data or not test_data:
        print("\n错误: 数据不足，无法训练")
        return None, None, None

    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    # 删除含NaN的行（特征列和标签列）
    cols_to_check = feature_cols + ['label', 'future_return']
    train_df = train_df.dropna(subset=cols_to_check)
    test_df = test_df.dropna(subset=cols_to_check)

    print(f"\n数据统计:")
    print(f"  训练集: {len(train_df):,} 样本 ({len(train_data)} 只股票)")
    print(f"  测试集: {len(test_df):,} 样本 ({len(test_data)} 只股票)")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  训练集上涨比例: {train_df['label'].mean():.2%}")
    print(f"  测试集上涨比例: {test_df['label'].mean():.2%}")

    return train_df, test_df, feature_cols


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    """
    训练XGBoost模型

    Args:
        train_df: 训练数据
        test_df: 测试数据
        feature_cols: 特征列名

    Returns:
        model, metrics
    """
    print(f"\n{'='*60}")
    print(f"训练模型")
    print(f"{'='*60}\n")

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    print(f"特征数: {len(feature_cols)}")
    print(f"训练样本: {len(X_train):,}")
    print(f"测试样本: {len(X_test):,}")

    # 计算类别权重
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # 模型参数
    params = {
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'min_child_weight': 3,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42,
    }

    print(f"\n模型参数:")
    for k, v in params.items():
        if k != 'scale_pos_weight':
            print(f"  {k}: {v}")
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 训练
    print(f"\n开始训练...")
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

    # 预测
    y_pred_prob_train = model.predict(dtrain)
    y_pred_train = (y_pred_prob_train > 0.5).astype(int)

    y_pred_prob_test = model.predict(dtest)
    y_pred_test = (y_pred_prob_test > 0.5).astype(int)

    # 训练集指标
    print("训练集:")
    print(f"  准确率: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"  精确率: {precision_score(y_train, y_pred_train, zero_division=0):.4f}")
    print(f"  召回率: {recall_score(y_train, y_pred_train, zero_division=0):.4f}")
    print(f"  F1分数: {f1_score(y_train, y_pred_train, zero_division=0):.4f}")
    print(f"  AUC: {roc_auc_score(y_train, y_pred_prob_train):.4f}")

    # 测试集指标
    print("\n测试集:")
    print(f"  准确率: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"  精确率: {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
    print(f"  召回率: {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
    print(f"  F1分数: {f1_score(y_test, y_pred_test, zero_division=0):.4f}")
    print(f"  AUC: {roc_auc_score(y_test, y_pred_prob_test):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n混淆矩阵 (测试集):")
    print(f"  预测跌  预测涨")
    print(f"  实际跌  {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"  实际涨  {cm[1,0]:>6}  {cm[1,1]:>6}")

    # 特征重要性
    print(f"\n特征重要性 Top 10:")
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feat, score) in enumerate(sorted_importance):
        feat_idx = int(feat.replace('f', ''))
        feat_name = feature_cols[feat_idx] if feat_idx < len(feature_cols) else feat
        print(f"  {i+1}. {feat_name}: {score:.2f}")

    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'auc': roc_auc_score(y_train, y_pred_prob_train),
            'size': len(y_train),
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'auc': roc_auc_score(y_test, y_pred_prob_test),
            'size': len(y_test),
        },
        'best_iteration': model.best_iteration,
    }

    return model, metrics, feature_cols


def backtest_strategy(model, test_df: pd.DataFrame, feature_cols: list,
                      buy_threshold: float = 0.55, sell_threshold: float = 0.45):
    """
    策略回测

    Args:
        model: 训练好的模型
        test_df: 测试数据
        feature_cols: 特征列
        buy_threshold: 买入阈值
        sell_threshold: 卖出阈值
    """
    print(f"\n{'='*60}")
    print(f"策略回测")
    print(f"买入阈值: {buy_threshold:.2f}, 卖出阈值: {sell_threshold:.2f}")
    print(f"{'='*60}\n")

    X_test = test_df[feature_cols].values
    returns = test_df['future_return'].values

    dtest = xgb.DMatrix(X_test)
    probs = model.predict(dtest)

    # 信号统计
    buy_signals = probs > buy_threshold
    sell_signals = probs < sell_threshold
    neutral = ((probs >= sell_threshold) & (probs <= buy_threshold))

    print(f"信号分布:")
    print(f"  买入信号: {buy_signals.sum():,} ({buy_signals.sum()/len(probs)*100:.1f}%)")
    print(f"  卖出信号: {sell_signals.sum():,} ({sell_signals.sum()/len(probs)*100:.1f}%)")
    print(f"  观望信号: {neutral.sum():,} ({neutral.sum()/len(probs)*100:.1f}%)")

    # 收益分析
    buy_returns = returns[buy_signals]
    hold_returns = returns[probs >= 0.5]
    all_returns = returns

    print(f"\n收益统计 (未来5日):")
    print(f"  买入信号平均收益: {buy_returns.mean()*100:.2f}%" if len(buy_returns) > 0 else "  买入信号: 无")
    print(f"  概率>=0.5平均收益: {hold_returns.mean()*100:.2f}%")
    print(f"  全市场平均收益: {all_returns.mean()*100:.2f}%")

    if len(buy_returns) > 0:
        win_rate = (buy_returns > 0).sum() / len(buy_returns)
        print(f"  买入信号胜率: {win_rate*100:.1f}%")
        print(f"  买入信号盈亏比: {(buy_returns[buy_returns > 0].mean() / abs(buy_returns[buy_returns < 0].mean())):.2f}"
              if (buy_returns < 0).sum() > 0 else "  盈亏比: N/A")


def main():
    parser = argparse.ArgumentParser(description='训练ML模型')
    parser.add_argument('--stocks', type=int, default=1000, help='股票数量限制')
    parser.add_argument('--train-start', type=str, default='20210201', help='训练开始日期')
    parser.add_argument('--train-end', type=str, default='20241231', help='训练结束日期')
    parser.add_argument('--test-start', type=str, default='20250101', help='测试开始日期')
    parser.add_argument('--test-end', type=str, default='20261231', help='测试结束日期')
    parser.add_argument('--prediction-period', type=int, default=5, help='预测周期')
    args = parser.parse_args()

    print("="*60)
    print("ML模型训练")
    print("="*60)
    print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 初始化存储
    storage = SQLiteStorage()

    # 准备数据
    train_df, test_df, feature_cols = prepare_data(
        storage,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        max_stocks=args.stocks,
        prediction_period=args.prediction_period
    )

    if train_df is None:
        return

    # 训练模型
    model, metrics, feature_cols = train_model(train_df, test_df, feature_cols)

    # 策略回测
    backtest_strategy(model, test_df, feature_cols)

    # 保存模型
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgboost_model.json'
    model.save_model(model_path)

    # 保存特征列表
    import json
    with open('models/feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)

    print(f"\n{'='*60}")
    print(f"模型已保存:")
    print(f"  模型文件: {model_path}")
    print(f"  特征列表: models/feature_cols.json")
    print(f"  最佳迭代: {metrics['best_iteration']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
