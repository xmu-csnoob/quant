"""
使用训练好的ML模型进行回测

用现有A股数据测试ML策略性能
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.sqlite_storage import SQLiteStorage
from utils.features.enhanced_features import EnhancedFeatureExtractor
from strategies.base import Signal, SignalType
import xgboost as xgb
from backtesting.simple_backtester import SimpleBacktester
from risk.manager import RiskManager, PositionSizer
from loguru import logger
import os


class MLBacktestStrategy:
    """ML回测策略"""

    def __init__(self, model_path='models/xgboost_classification.json'):
        self.model_path = model_path
        self.model = None
        self.feature_extractor = EnhancedFeatureExtractor()
        self.name = "ML_Strategy"

    def load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        print(f"✓ 加载模型: {self.model_path}")

    def generate_signals(self, df):
        """
        生成交易信号

        Args:
            df: OHLCV数据

        Returns:
            信号列表
        """
        if df is None or df.empty:
            return []

        # 提取特征
        features = self.feature_extractor.extract(df)

        # 计算未来收益率（用于标签）
        features['future_return_5'] = features['close'].pct_change(5).shift(-5)

        # 删除包含NaN的行
        features = features.dropna()

        if features.empty:
            return []

        # 准备特征
        feature_cols = [c for c in features.columns if c.startswith('f_')]
        X = features[feature_cols].values

        # 预测涨跌概率
        probs = self.model.predict(xgb.DMatrix(X))

        # 生成信号
        signals = []
        for i in range(len(features)):
            date = features['trade_date'].iloc[i]
            prob = probs[i]

            # 只有高置信度才交易
            if prob > 0.6:  # 买入
                signals.append(Signal(
                    date=date.strftime('%Y%m%d'),
                    signal_type=SignalType.BUY,
                    price=features['close'].iloc[i],
                    reason=f"ML预测上涨概率={prob:.2f}",
                    confidence=prob
                ))
            elif prob < 0.4:  # 卖出
                signals.append(Signal(
                    date=date.strftime('%Y%m%d'),
                    signal_type=SignalType.SELL,
                    price=features['close'].iloc[i],
                    reason=f"ML预测下跌概率={1-prob:.2f}",
                    confidence=1-prob
                ))

        return signals


def run_ml_backtest():
    """运行ML回测"""
    print("=" * 80)
    print("ML策略回测")
    print("=" * 80)

    # 初始化
    storage = SQLiteStorage()
    strategy = MLBacktestStrategy()

    # 加载模型
    print("\n[1/4] 加载训练好的模型...")
    try:
        strategy.load_model()
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        print("\n  请先运行: python scripts/train_ml_with_current_data.py")
        return

    # 获取测试数据
    print("\n[2/4] 获取测试数据...")
    stocks = storage.get_all_stocks()
    print(f"  可用股票: {len(stocks)} 只")

    # 选择一些股票进行回测（使用训练时没用过的股票）
    test_stocks = sorted(stocks)[300:350]  # 跳过前300只（可能用于训练）
    print(f"  测试股票: {len(test_stocks)} 只")

    # 运行回测
    print(f"\n[3/4] 运行回测...")

    all_results = []

    for i, ts_code in enumerate(test_stocks):
        if (i + 1) % 10 == 0:
            print(f"  处理中: {i+1}/{len(test_stocks)}")

        try:
            # 获取数据
            df = storage.get_daily_prices(ts_code, "20200101", "20241231")

            if df is None or len(df) < 100:
                continue

            # 生成信号
            signals = strategy.generate_signals(df)

            if not signals:
                continue

            # 回测
            backtester = SimpleBacktester(initial_capital=100000)
            result = backtester.run(strategy, df)

            if result and result.trade_count > 0:
                all_results.append({
                    'symbol': ts_code,
                    'total_return': result.total_return,
                    'win_rate': result.win_rate,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'num_trades': result.trade_count,
                    'final_capital': result.final_capital,
                })

        except Exception as e:
            logger.warning(f"回测 {ts_code} 失败: {e}")
            continue

    print(f"  ✓ 完成 {len(all_results)} 只股票的回测")

    # 统计结果
    print(f"\n[4/4] 分析结果...")

    if not all_results:
        print("  无有效结果")
        return

    # 计算汇总
    total_return = np.mean([r['total_return'] for r in all_results])
    win_rate = np.mean([r['win_rate'] for r in all_results])
    max_drawdown = np.mean([r['max_drawdown'] for r in all_results])
    sharpe = np.mean([r.get('sharpe_ratio', 0) for r in all_results])

    # 找出表现最好的股票
    best_result = max(all_results, key=lambda x: x['total_return'])

    print(f"\n整体表现:")
    print(f"  平均收益率: {total_return:.2%}")
    print(f"  平均胜率: {win_rate:.2%}")
    print(f"  平均最大回撤: {max_drawdown:.2%}")
    print(f"  平均夏普比率: {sharpe:.2f}")

    print(f"\n表现最好的股票:")
    print(f"  {best_result['symbol']}: {best_result['total_return']:.2%}")
    print(f"  胜率: {best_result['win_rate']:.2%}")
    print(f"  交易次数: {best_result['num_trades']}")

    # 显示部分详细结果
    print(f"\n部分详细结果 (按收益率排序):")
    sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)

    print(f"\n{'股票':<12} {'收益率':<10} {'胜率':<8} {'交易次数':<8}")
    print("-" * 40)

    for r in sorted_results[:10]:
        print(f"{r['symbol']:<12} {r['total_return']:>8.2%} {r['win_rate']:>7.1%} {r['num_trades']:>8}")

    # 与买入持有对比
    print(f"\n与买入持有对比:")
    buy_hold_count = sum(1 for r in all_results if r['total_return'] > 0)
    print(f"  跑赢买入持有: {buy_hold_count}/{len(all_results)} ({buy_hold_count/len(all_results)*100:.1f}%)")

    # 保存详细结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('total_return', ascending=False)

    os.makedirs('backtest_results', exist_ok=True)
    results_df.to_csv('backtest_results/ml_backtest_results.csv', index=False)

    print(f"\n✓ 详细结果已保存到: backtest_results/ml_backtest_results.csv")


if __name__ == "__main__":
    run_ml_backtest()
