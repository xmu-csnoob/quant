"""
机器学习交易模型训练脚本 v3

使用增强特征（60+个特征）
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tqdm import tqdm

from src.data.storage.storage import DataStorage
from src.data.fetchers.base import Exchange
from src.utils.features.enhanced_features import EnhancedFeatureExtractor
from src.utils.labels import LabelGenerator


def clip_outliers(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """截断异常值"""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower_bound, upper_bound)
from src.backtesting.simple_backtester import SimpleBacktester


def train_ml_model_v3():
    """训练ML交易模型 v3 - 增强特征"""
    print("=" * 80)
    print("机器学习交易模型训练 v3 (增强特征)")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    storage = DataStorage()

    stock_list = [
        ("600000.SH", Exchange.SSE),
        ("600036.SH", Exchange.SSE),
        ("601398.SH", Exchange.SSE),
        ("600519.SH", Exchange.SSE),
        ("600887.SH", Exchange.SSE),
        ("601318.SH", Exchange.SSE),
        ("000001.SZ", Exchange.SZSE),
        ("000002.SZ", Exchange.SZSE),
        ("000858.SZ", Exchange.SZSE),
    ]

    all_data = []
    for stock_code, exchange in stock_list:
        df = storage.load_daily_price(stock_code, exchange)
        if len(df) > 0:
            df["ts_code"] = stock_code
            all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    print(f"  ✓ 加载 {len(stock_list)} 只股票，共 {len(df_all)} 条数据")
    print(f"  日期范围: {df_all['trade_date'].min()} ~ {df_all['trade_date'].max()}")

    # 2. 增强特征工程
    print("\n[2] 增强特征工程...")
    feature_extractor = EnhancedFeatureExtractor(prediction_period=5)
    df_features = feature_extractor.extract(df_all)

    feature_cols = [c for c in df_features.columns if c.startswith("f_")]
    print(f"  ✓ 提取 {len(feature_cols)} 个增强特征")

    # 3. 生成标签
    print("\n[3] 生成标签...")
    label_gen = LabelGenerator(
        prediction_period=5,
        task_type="regression",
        threshold=0.02,
    )
    df_labeled = label_gen.generate(df_features)

    # 截断异常值
    df_labeled["label"] = clip_outliers(df_labeled["label"], 0.01, 0.99)

    stats = label_gen.get_label_distribution(df_labeled)
    print(f"  标签统计（截断后）:")
    print(f"    均值: {stats['mean']:.4f}")
    print(f"    标准差: {stats['std']:.4f}")
    print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # 4. 准备训练数据
    print("\n[4] 准备训练数据（时间序列分割）...")
    df_clean = df_labeled.dropna(subset=feature_cols + ["label"]).copy()
    df_clean = df_clean.sort_values("trade_date").reset_index(drop=True)

    train_size = int(len(df_clean) * 0.6)
    val_size = int(len(df_clean) * 0.2)

    df_train = df_clean.iloc[:train_size].copy()
    df_val = df_clean.iloc[train_size:train_size + val_size].copy()
    df_test = df_clean.iloc[train_size + val_size:].copy()

    print(f"  训练集: {len(df_train)} 条")
    print(f"  验证集: {len(df_val)} 条")
    print(f"  测试集: {len(df_test)} 条")

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["label"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values

    # 5. 训练模型
    print("\n[5] 训练XGBoost模型（增强特征）...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'max_depth': 5,
        'min_child_weight': 3,
        'gamma': 0.05,
        'reg_alpha': 0.3,
        'reg_lambda': 1.5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'learning_rate': 0.03,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
    }

    results = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False,
        evals_result=results,
    )

    print(f"  ✓ 训练完成（{bst.best_iteration} 棵树）")

    # 6. 特征选择（保留重要特征）
    print("\n[6] 特征选择...")

    importance = bst.get_score(importance_type='gain')
    importance_dict = {feature_cols[int(k[1:])]: v for k, v in importance.items()}

    # 保留前20个重要特征
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    selected_features = [f for f, _ in sorted_features[:20]]

    print(f"  ✓ 特征数量: {len(feature_cols)} → {len(selected_features)}")

    # 重新训练
    selected_indices = [feature_cols.index(f) for f in selected_features]

    X_train_sel = X_train[:, selected_indices]
    X_val_sel = X_val[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    dtrain_sel = xgb.DMatrix(X_train_sel, label=y_train)
    dval_sel = xgb.DMatrix(X_val_sel, label=y_val)

    bst_final = xgb.train(
        params,
        dtrain_sel,
        num_boost_round=500,
        evals=[(dtrain_sel, 'train'), (dval_sel, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    print(f"  ✓ 重新训练完成（{bst_final.best_iteration} 棵树, {len(selected_features)} 个特征）")

    # 7. 评估模型
    print("\n[7] 模型评估...")

    def evaluate(X, y, name):
        dmat = xgb.DMatrix(X)
        pred = bst_final.predict(dmat)
        mse = mean_squared_error(y, pred)
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        direction_acc = np.mean((y > 0) == (pred > 0))

        print(f"\n  {name}:")
        print(f"    MSE:  {mse:.6f}")
        print(f"    MAE:  {mae:.6f}")
        print(f"    R²:   {r2:.4f}")
        print(f"    方向准确率: {direction_acc:.2%}")

        return pred

    pred_train = evaluate(X_train_sel, y_train, "训练集")
    pred_val = evaluate(X_val_sel, y_val, "验证集")
    pred_test = evaluate(X_test_sel, y_test, "测试集")

    # 8. 特征重要性
    print("\n[8] 特征重要性 (Top 15):")
    importance_final = bst_final.get_score(importance_type='gain')
    importance_dict = {selected_features[int(k[1:])]: v for k, v in importance_final.items()}
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    for i, (feat, imp) in enumerate(sorted_importance[:15]):
        print(f"  {i+1}. {feat:25s} {imp:.4f}")

    # 9. 回测评估
    print("\n" + "=" * 80)
    print("[9] 回测评估")
    print("=" * 80)

    from src.strategies.ml_strategy import MLStrategy

    # 创建模型包装器
    class XGBWrapper:
        def __init__(self, bst, selected_indices):
            self.bst = bst
            self.selected_indices = selected_indices
            self.__class__.__name__ = "XGBRegressor"

        def predict(self, X):
            if X.shape[1] > len(self.selected_indices):
                X = X[:, self.selected_indices]
            dmat = xgb.DMatrix(X)
            return self.bst.predict(dmat)

    model_final = XGBWrapper(bst_final, selected_indices)

    ml_strategy = MLStrategy(
        model=model_final,
        feature_extractor=feature_extractor,
        threshold=0.01,  # 1%阈值
    )

    results = []

    for stock_code, exchange in [
        ("600000.SH", Exchange.SSE),
        ("600036.SH", Exchange.SSE),
        ("600519.SH", Exchange.SSE),
        ("601318.SH", Exchange.SSE),
        ("000001.SZ", Exchange.SZSE),
        ("000002.SZ", Exchange.SZSE),
        ("000858.SZ", Exchange.SZSE),
    ]:
        df = storage.load_daily_price(stock_code, exchange)
        if len(df) < 200:
            continue

        # 只用测试期数据
        test_start = df_test["trade_date"].min()
        test_end = df_test["trade_date"].max()
        df = df[(df["trade_date"] >= test_start) & (df["trade_date"] <= test_end)]

        if len(df) < 50:
            continue

        backtester = SimpleBacktester(initial_capital=100000)
        result = backtester.run(ml_strategy, df)

        ml_return = result.total_return * 100
        buy_hold_return = (df.iloc[-1]["close"] / df.iloc[0]["close"] - 1) * 100

        results.append({
            "stock": stock_code,
            "ml_return": ml_return,
            "buy_hold": buy_hold_return,
            "signals": result.trade_count * 2,
        })

    # 结果汇总
    print("\n" + "-" * 80)
    print(f"{'股票':<12} {'ML策略':>10} {'买入持有':>10} {'信号数':>8} {'结果':>10}")
    print("-" * 80)

    ml_beats = 0
    for r in results:
        comparison = "✓" if r["ml_return"] > r["buy_hold"] else "✗"
        if r["ml_return"] > r["buy_hold"]:
            ml_beats += 1

        print(f"{r['stock']:<12} {r['ml_return']:>9.2f}% {r['buy_hold']:>10.2f}% {r['signals']:>8} {comparison:>10}")

    print("-" * 80)
    print(f"跑赢买入持有: {ml_beats}/{len(results)} ({ml_beats/len(results)*100:.1f}%)")

    if results:
        avg_ml = np.mean([r["ml_return"] for r in results])
        avg_bh = np.mean([r["buy_hold"] for r in results])
        print(f"平均收益: ML策略 {avg_ml:+.2f}% | 买入持有 {avg_bh:+.2f}%")

    # 保存模型
    print("\n[10] 保存模型...")
    model_path = project_root / "models" / "xgboost_trading_model_v3.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    bst_final.save_model(str(model_path))

    import json
    config_path = project_root / "models" / "xgboost_trading_model_v3_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'selected_features': selected_features,
            'feature_cols': feature_cols,
            'threshold': 0.01,
            'prediction_period': 5,
        }, f, indent=2)

    print(f"  ✓ 模型已保存: {model_path}")
    print(f"  ✓ 配置已保存: {config_path}")

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    train_ml_model_v3()
