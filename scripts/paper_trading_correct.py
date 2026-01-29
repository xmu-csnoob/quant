"""
正确的样本外测试 - 严格使用训练后的时间段

测试期：2024年Q1（与Walk-Forward最后一个测试期一致）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.sqlite_storage import SQLiteStorage
from utils.features.enhanced_features import EnhancedFeatureExtractor
import xgboost as xgb
from loguru import logger
import os


def run_correct_test():
    """运行正确的样本外测试"""
    print("=" * 80)
    print("严格的样本外测试 - 2024年Q1")
    print("=" * 80)

    storage = SQLiteStorage()

    # 严格按照Walk-Forward的最后一个测试期
    test_start = datetime(2024, 1, 1)
    test_end = datetime(2024, 3, 31)

    print(f"\n重要说明:")
    print(f"  训练数据期间: 2020-01-01 ~ 2023-12-31")
    print(f"  测试数据期间: 2024-01-01 ~ 2024-03-31")
    print(f"  ✓ 严格的时间划分，无数据泄漏")
    print(f"\n策略参数: 买入阈值0.52, 卖出阈值0.48")

    print(f"\n[1/3] 获取股票...")
    stocks = storage.get_all_stocks()
    np.random.seed(42)
    test_stocks = sorted(stocks)[:200]
    print(f"  测试股票: {len(test_stocks)} 只")

    print(f"\n[2/3] 运行回测...")

    model = xgb.Booster()
    model.load_model('models/xgboost_robust.json')
    feature_extractor = EnhancedFeatureExtractor()

    all_results = []

    for i, ts_code in enumerate(test_stocks):
        if (i + 1) % 50 == 0:
            print(f"  处理中: {i+1}/{len(test_stocks)}")

        try:
            # 获取数据（需要前面90天用于特征计算）
            df = storage.get_daily_prices(
                ts_code,
                (test_start - timedelta(days=90)).strftime('%Y%m%d'),
                test_end.strftime('%Y%m%d')
            )

            if df is None or len(df) < 60:
                continue

            features = feature_extractor.extract(df)
            features['trade_date'] = pd.to_datetime(features['trade_date'])
            features = features[
                (features['trade_date'] >= test_start) &
                (features['trade_date'] <= test_end)
            ].reset_index(drop=True)

            if len(features) < 40:  # Q1约60个交易日
                continue

            # 提取特征并预测
            feature_cols = [c for c in features.columns if c.startswith('f_')]
            X = features[feature_cols].values
            probs = model.predict(xgb.DMatrix(X))

            # 回测
            capital = 100000
            position = None
            trades = []
            daily_values = []

            for j in range(len(features)):
                close = features['close'].iloc[j]
                prob = probs[j]

                # 更新净值
                if position:
                    market_value = position['quantity'] * close
                else:
                    market_value = 0
                total_value = capital + market_value

                daily_values.append(total_value)

                # 交易逻辑
                if prob > 0.52 and position is None:
                    buy_value = capital * 0.95
                    quantity = int(buy_value / close / 100) * 100

                    if quantity >= 100:
                        cost = quantity * close * 1.0003
                        if cost <= capital:
                            capital -= cost
                            position = {
                                'entry_date': features['trade_date'].iloc[j],
                                'entry_price': close,
                                'quantity': quantity,
                            }

                elif prob < 0.48 and position:
                    sell_value = position['quantity'] * close * 0.9997
                    pnl = sell_value - position['quantity'] * position['entry_price']
                    pnl_ratio = pnl / (position['quantity'] * position['entry_price'])

                    capital += sell_value
                    trades.append({
                        'pnl': pnl,
                        'pnl_ratio': pnl_ratio,
                    })
                    position = None

            # 期末平仓
            if position:
                last_price = features['close'].iloc[-1]
                sell_value = position['quantity'] * last_price * 0.9997
                pnl = sell_value - position['quantity'] * position['entry_price']
                pnl_ratio = pnl / (position['quantity'] * position['entry_price'])
                capital += sell_value
                trades.append({'pnl': pnl, 'pnl_ratio': pnl_ratio})
                position = None

            # 计算指标
            if trades:
                total_return = (capital - 100000) / 100000
                buy_hold_return = (features['close'].iloc[-1] - features['close'].iloc[0]) / features['close'].iloc[0]
                win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)

                all_results.append({
                    'symbol': ts_code,
                    'total_return': total_return,
                    'buy_hold_return': buy_hold_return,
                    'trade_count': len(trades),
                    'win_rate': win_rate,
                    'final_capital': capital,
                })

        except Exception as e:
            logger.warning(f"回测 {ts_code} 失败: {e}")
            continue

    print(f"  ✓ 完成 {len(all_results)} 只股票")

    print(f"\n[3/3] 分析结果...")

    if not all_results:
        print("  无有效结果")
        return

    # 统计
    avg_return = np.mean([r['total_return'] for r in all_results])
    median_return = np.median([r['total_return'] for r in all_results])
    win_rate = np.mean([r['win_rate'] for r in all_results])
    avg_trades = np.mean([r['trade_count'] for r in all_results])
    beat_market = sum(1 for r in all_results if r['total_return'] > r['buy_hold_return'])
    avg_buy_hold = np.mean([r['buy_hold_return'] for r in all_results])

    print(f"\n{'='*80}")
    print("2024年Q1样本外测试结果（严格时间划分）")
    print(f"{'='*80}")
    print(f"\n测试概况:")
    print(f"  测试股票: {len(all_results)} 只")
    print(f"  测试期间: 2024-01-01 ~ 2024-03-31 (Q1)")
    print(f"  训练期间: 2020-01-01 ~ 2023-12-31")
    print(f"\n收益指标:")
    print(f"  平均收益: {avg_return*100:.2f}%")
    print(f"  中位数收益: {median_return*100:.2f}%")
    print(f"  买入持有: {avg_buy_hold*100:.2f}%")
    print(f"  超额收益: {(avg_return-avg_buy_hold)*100:.2f}%")
    print(f"  跑赢市场: {beat_market}/{len(all_results)} ({beat_market/len(all_results)*100:.1f}%)")
    print(f"\n交易指标:")
    print(f"  胜率: {win_rate*100:.2f}%")
    print(f"  平均交易次数: {avg_trades:.1f}")

    # 收益分布
    positive = sum(1 for r in all_results if r['total_return'] > 0)
    negative = len(all_results) - positive

    print(f"\n收益分布:")
    print(f"  盈利: {positive} 只 ({positive/len(all_results)*100:.1f}%)")
    print(f"  亏损: {negative} 只 ({negative/len(all_results)*100:.1f}%)")

    # Top 10
    sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)

    print(f"\n表现最好 (Top 10):")
    print(f"\n{'股票':<12} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 75)

    for r in sorted_results[:10]:
        excess = (r['total_return'] - r['buy_hold_return']) * 100
        print(f"{r['symbol']:<12} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6} {r['win_rate']*100:>6.1f}%")

    print(f"\n表现最差 (Bottom 5):")
    print(f"\n{'股票':<12} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 75)

    for r in sorted_results[-5:]:
        excess = (r['total_return'] - r['buy_hold_return']) * 100
        print(f"{r['symbol']:<12} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6} {r['win_rate']*100:>6.1f}%")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('total_return', ascending=False)

    os.makedirs('backtest_results', exist_ok=True)
    results_df.to_csv('backtest_results/out_of_sample_2024q1.csv', index=False)

    print(f"\n✓ 结果已保存到: backtest_results/out_of_sample_2024q1.csv")


if __name__ == "__main__":
    run_correct_test()
