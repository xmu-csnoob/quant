"""
机器学习交易模型训练脚本

使用XGBoost训练预测模型
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tqdm import tqdm

from src.data.storage.storage import DataStorage
from src.data.fetchers.base import Exchange
from src.utils.features.ml_features import MLFeatureExtractor
from src.utils.labels import LabelGenerator
from src.backtesting.simple_backtester import SimpleBacktester


def train_ml_model():
    """训练ML交易模型"""
    print("=" * 80)
    print("机器学习交易模型训练")
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

    # 2. 特征工程
    print("\n[2] 特征工程...")
    feature_extractor = MLFeatureExtractor(prediction_period=5)
    df_features = feature_extractor.extract(df_all)

    feature_cols = [c for c in df_features.columns if c.startswith("f_")]
    print(f"  ✓ 提取 {len(feature_cols)} 个特征")

    # 3. 生成标签
    print("\n[3] 生成标签...")
    label_gen = LabelGenerator(
        prediction_period=5,
        task_type="regression",
        threshold=0.02,
    )
    df_labeled = label_gen.generate(df_features)

    stats = label_gen.get_label_distribution(df_labeled)
    print(f"  标签统计:")
    print(f"    均值: {stats['mean']:.4f}")
    print(f"    标准差: {stats['std']:.4f}")
    print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # 4. 准备训练数据
    print("\n[4] 准备训练数据...")

    # 删除包含NaN的行
    df_clean = df_labeled.dropna(subset=feature_cols + ["label"]).copy()

    # 时间序列分割（重要！不能用随机分割）
    train_size = int(len(df_clean) * 0.6)
    val_size = int(len(df_clean) * 0.2)

    df_train = df_clean.iloc[:train_size].copy()
    df_val = df_clean.iloc[train_size:train_size + val_size].copy()
    df_test = df_clean.iloc[train_size + val_size:].copy()

    print(f"  训练集: {len(df_train)} 条 ({df_train['trade_date'].min()} ~ {df_train['trade_date'].max()})")
    print(f"  验证集: {len(df_val)} 条 ({df_val['trade_date'].min()} ~ {df_val['trade_date'].max()})")
    print(f"  测试集: {len(df_test)} 条 ({df_test['trade_date'].min()} ~ {df_test['trade_date'].max()})")

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["label"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values

    # 5. 训练模型
    print("\n[5] 训练XGBoost模型...")

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print("  ✓ 训练完成")

    # 6. 评估模型
    print("\n[6] 模型评估...")

    def evaluate(X, y, name):
        pred = model.predict(X)
        mse = mean_squared_error(y, pred)
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)

        # 方向准确率
        direction_acc = np.mean((y > 0) == (pred > 0))

        print(f"\n  {name}:")
        print(f"    MSE:  {mse:.6f}")
        print(f"    MAE:  {mae:.6f}")
        print(f"    R²:   {r2:.4f}")
        print(f"    方向准确率: {direction_acc:.2%}")

        return pred

    pred_train = evaluate(X_train, y_train, "训练集")
    pred_val = evaluate(X_val, y_val, "验证集")
    pred_test = evaluate(X_test, y_test, "测试集")

    # 7. 特征重要性
    print("\n[7] 特征重要性 (Top 10):")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    for i in range(min(10, len(feature_cols))):
        print(f"  {i+1}. {feature_cols[indices[i]]:20s} {importance[indices[i]]:.4f}")

    # 8. 回测评估
    print("\n" + "=" * 80)
    print("[8] 回测评估")
    print("=" * 80)

    from src.strategies.ml_strategy import MLStrategy

    ml_strategy = MLStrategy(
        model=model,
        feature_extractor=feature_extractor,
        threshold=0.02,  # 预测收益率>2%时买入
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

        # 使用backtester运行完整回测
        backtester = SimpleBacktester(initial_capital=100000)
        result = backtester.run(ml_strategy, df)

        ml_return = result.total_return * 100

        # 买入持有
        buy_hold_return = (df.iloc[-1]["close"] / df.iloc[0]["close"] - 1) * 100

        results.append({
            "stock": stock_code,
            "ml_return": ml_return,
            "buy_hold": buy_hold_return,
            "signals": result.trade_count * 2,  # 每笔交易有买入和卖出
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

    avg_ml = np.mean([r["ml_return"] for r in results])
    avg_bh = np.mean([r["buy_hold"] for r in results])
    print(f"平均收益: ML策略 {avg_ml:+.2f}% | 买入持有 {avg_bh:+.2f}%")

    # 保存模型
    print("\n[9] 保存模型...")
    model_path = project_root / "models" / "xgboost_trading_model.joblib"
    ml_strategy.save_model(str(model_path))
    print(f"  ✓ 模型已保存: {model_path}")

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    train_ml_model()
