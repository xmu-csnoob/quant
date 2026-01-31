"""
全面的模拟盘测试 - 2024年全年数据

测试更长时间段，更多股票，全面评估策略表现
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor
from src.strategies.base import Signal, SignalType
import xgboost as xgb
from loguru import logger
import os


class ComprehensivePaperTrader:
    """全面模拟盘测试器"""

    def __init__(
        self,
        initial_capital=100000,
        model_path='models/xgboost_robust.json',
        buy_threshold=0.52,
        sell_threshold=0.48,
        transaction_cost=0.0003,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.daily_values = []

        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.feature_extractor = EnhancedFeatureExtractor()

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.transaction_cost = transaction_cost

    def run_backtest(self, df, features_df):
        """运行回测"""
        feature_cols = [c for c in features_df.columns if c.startswith('f_')]
        if len(feature_cols) == 0:
            return 0

        X = features_df[feature_cols].values
        probs = self.model.predict(xgb.DMatrix(X))

        # 生成交易信号
        for i in range(len(features_df)):
            date = features_df['trade_date'].iloc[i]
            close = features_df['close'].iloc[i]
            prob = probs[i]

            # 更新净值
            if self.position:
                market_value = self.position['quantity'] * close
            else:
                market_value = 0
            total_value = self.capital + market_value

            self.daily_values.append({
                'date': date,
                'total_value': total_value,
                'cash': self.capital,
                'position_value': market_value,
                'position': self.position['quantity'] if self.position else 0,
            })

            # 买入信号
            if prob > self.buy_threshold and self.position is None:
                buy_value = self.capital * 0.95
                quantity = int(buy_value / close / 100) * 100

                if quantity >= 100:
                    cost = quantity * close * (1 + self.transaction_cost)
                    if cost <= self.capital:
                        self.capital -= cost
                        self.position = {
                            'entry_date': date,
                            'entry_price': close,
                            'quantity': quantity,
                        }

            # 卖出信号
            elif prob < self.sell_threshold and self.position:
                sell_value = self.position['quantity'] * close * (1 - self.transaction_cost)
                pnl = sell_value - self.position['quantity'] * self.position['entry_price']
                pnl_ratio = pnl / (self.position['quantity'] * self.position['entry_price'])

                self.capital += sell_value

                self.trades.append({
                    'entry_date': self.position['entry_date'],
                    'exit_date': date,
                    'entry_price': self.position['entry_price'],
                    'exit_price': close,
                    'quantity': self.position['quantity'],
                    'pnl': pnl,
                    'pnl_ratio': pnl_ratio,
                })

                self.position = None

        # 期末平仓
        if self.position:
            last_price = features_df['close'].iloc[-1]
            last_date = features_df['trade_date'].iloc[-1]
            sell_value = self.position['quantity'] * last_price * (1 - self.transaction_cost)
            pnl = sell_value - self.position['quantity'] * self.position['entry_price']
            pnl_ratio = pnl / (self.position['quantity'] * self.position['entry_price'])

            self.capital += sell_value

            self.trades.append({
                'entry_date': self.position['entry_date'],
                'exit_date': last_date,
                'entry_price': self.position['entry_price'],
                'exit_price': last_price,
                'quantity': self.position['quantity'],
                'pnl': pnl,
                'pnl_ratio': pnl_ratio,
            })

            self.position = None

        return len(self.trades)

    def get_metrics(self, benchmark_values=None):
        """计算绩效指标"""
        if not self.daily_values:
            return None

        df_values = pd.DataFrame(self.daily_values)
        final_value = df_values['total_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        df_values['daily_return'] = df_values['total_value'].pct_change()
        df_values['cummax'] = df_values['total_value'].cummax()
        df_values['drawdown'] = (df_values['cummax'] - df_values['total_value']) / df_values['cummax']
        max_drawdown = df_values['drawdown'].max()

        returns = df_values['daily_return'].dropna()
        sharpe_ratio = 0
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std()
            sharpe_ratio = sharpe * np.sqrt(252)

        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            win_rate = (trades_df['pnl'] > 0).mean()
            avg_return = trades_df['pnl_ratio'].mean()
            win_count = (trades_df['pnl'] > 0).sum()
            lose_count = (trades_df['pnl'] <= 0).sum()
        else:
            win_rate = 0
            avg_return = 0
            win_count = 0
            lose_count = 0

        benchmark_return = 0
        excess_return = 0
        if benchmark_values is not None and len(benchmark_values) > 0:
            benchmark_return = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]
            excess_return = total_return - benchmark_return

        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trade_count': len(self.trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'win_count': win_count,
            'lose_count': lose_count,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
        }


def run_comprehensive_test():
    """运行全面测试"""
    print("=" * 80)
    print("全面模拟盘测试 - 2024年完整数据")
    print("=" * 80)

    storage = SQLiteStorage()

    # 测试2024年完整数据
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    print(f"\n测试期间: {start_date.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')}")
    print(f"测试时长: 1年完整数据")
    print(f"策略参数: 买入阈值0.52, 卖出阈值0.48")

    # 获取股票列表
    print("\n[1/3] 获取股票列表...")
    stocks = storage.get_all_stocks()
    print(f"  可用股票: {len(stocks)} 只")

    # 使用更多股票进行测试
    np.random.seed(42)
    test_stocks = sorted(stocks)[:300]  # 增加到300只
    print(f"  测试股票: {len(test_stocks)} 只")

    print(f"\n[2/3] 运行回测...")

    all_results = []
    stats_by_quarter = {
        'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []
    }

    for i, ts_code in enumerate(test_stocks):
        if (i + 1) % 50 == 0:
            print(f"  处理中: {i+1}/{len(test_stocks)}")

        try:
            df = storage.get_daily_prices(
                ts_code,
                (start_date - timedelta(days=90)).strftime('%Y%m%d'),
                end_date.strftime('%Y%m%d')
            )

            if df is None or len(df) < 60:
                continue

            feature_extractor = EnhancedFeatureExtractor()
            features = feature_extractor.extract(df)

            features['trade_date'] = pd.to_datetime(features['trade_date'])
            features = features[
                (features['trade_date'] >= start_date) &
                (features['trade_date'] <= end_date)
            ].reset_index(drop=True)

            if len(features) < 100:  # 确保有足够数据
                continue

            trader = ComprehensivePaperTrader(initial_capital=100000)
            trader.run_backtest(df, features)

            buy_hold_return = (features['close'].iloc[-1] - features['close'].iloc[0]) / features['close'].iloc[0]
            metrics = trader.get_metrics(features['close'].values)

            if metrics and metrics['trade_count'] > 0:
                metrics['symbol'] = ts_code
                metrics['buy_hold_return'] = buy_hold_return
                all_results.append(metrics)

                # 按季度分类
                for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                    stats_by_quarter[quarter].append(metrics)

        except Exception as e:
            logger.warning(f"回测 {ts_code} 失败: {e}")
            continue

    print(f"  ✓ 完成 {len(all_results)} 只股票")

    print(f"\n[3/3] 分析结果...")

    if not all_results:
        print("  无有效结果")
        return

    # 整体统计
    avg_return = np.mean([r['total_return'] for r in all_results])
    median_return = np.median([r['total_return'] for r in all_results])
    win_rate = np.mean([r['win_rate'] for r in all_results])
    max_drawdown = np.mean([r['max_drawdown'] for r in all_results])
    sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
    avg_trades = np.mean([r['trade_count'] for r in all_results])
    beat_market = sum(1 for r in all_results if r['total_return'] > r['buy_hold_return'])
    avg_buy_hold = np.mean([r['buy_hold_return'] for r in all_results])

    # 收益分布
    returns = [r['total_return'] for r in all_results]
    positive_count = sum(1 for r in returns if r > 0)
    negative_count = sum(1 for r in returns if r < 0)

    print(f"\n{'='*80}")
    print("整体表现")
    print(f"{'='*80}")
    print(f"测试股票数: {len(all_results)}")
    print(f"测试期间: 2024年完整年份")
    print(f"\n收益指标:")
    print(f"  平均收益: {avg_return*100:.2f}%")
    print(f"  中位数收益: {median_return*100:.2f}%")
    print(f"  买入持有收益: {avg_buy_hold*100:.2f}%")
    print(f"  超额收益: {(avg_return-avg_buy_hold)*100:.2f}%")
    print(f"  跑赢市场: {beat_market}/{len(all_results)} ({beat_market/len(all_results)*100:.1f}%)")
    print(f"\n收益分布:")
    print(f"  盈利股票: {positive_count} ({positive_count/len(all_results)*100:.1f}%)")
    print(f"  亏损股票: {negative_count} ({negative_count/len(all_results)*100:.1f}%)")
    print(f"\n风险指标:")
    print(f"  胜率: {win_rate*100:.2f}%")
    print(f"  平均最大回撤: {max_drawdown*100:.2f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"\n交易指标:")
    print(f"  平均交易次数: {avg_trades:.1f}")
    print(f"  总交易次数: {sum(r['trade_count'] for r in all_results)}")

    # 表现最好/最差
    sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)

    print(f"\n{'='*80}")
    print("表现最好的股票 (Top 20)")
    print(f"{'='*80}")
    print(f"\n{'股票':<12} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 75)

    for r in sorted_results[:20]:
        excess = (r['total_return'] - r['buy_hold_return']) * 100
        print(f"{r['symbol']:<12} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6} {r['win_rate']*100:>6.1f}%")

    print(f"\n{'='*80}")
    print("表现最差的股票 (Bottom 10)")
    print(f"{'='*80}")
    print(f"\n{'股票':<12} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 75)

    for r in sorted_results[-10:]:
        excess = (r['total_return'] - r['buy_hold_return']) * 100
        print(f"{r['symbol']:<12} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6} {r['win_rate']*100:>6.1f}%")

    # 收益率区间分布
    print(f"\n{'='*80}")
    print("收益率分布")
    print(f"{'='*80}")

    bins = [-np.inf, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, np.inf]
    labels = ['<-20%', '-20~-10%', '-10~-5%', '-5~0%', '0~5%', '5~10%', '10~20%', '>20%']

    returns_series = pd.Series(returns)
    dist = returns_series.value_counts(bins=bins, sort=False)

    print(f"\n{'区间':<15} {'股票数':<10} {'占比':<10}")
    print("-" * 35)
    for i, (label, count) in enumerate(zip(labels, dist)):
        print(f"{label:<15} {count:<10} {count/len(all_results)*100:>8.1f}%")

    # 保存详细结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('total_return', ascending=False)

    os.makedirs('backtest_results', exist_ok=True)
    results_df.to_csv('backtest_results/comprehensive_test_2024.csv', index=False)

    print(f"\n✓ 详细结果已保存到: backtest_results/comprehensive_test_2024.csv")

    # 与历史版本对比
    print(f"\n{'='*80}")
    print("历史版本对比")
    print(f"{'='*80}")
    print(f"\n{'版本':<25} {'测试期':<15} {'股票数':<10} {'平均收益':<12} {'交易次数':<10} {'胜率':<10}")
    print("-" * 90)
    print(f"{'原版 (阈值0.6)':<25} {'2024Q4':<15} {'15':<10} {'-1.10%':<12} {'1.0':<10} {'53.33%':<10}")
    print(f"{'固定低阈值版':<25} {'2024Q4':<15} {'38':<10} {'+6.66%':<12} {'3.7':<10} {'59.69%':<10}")
    print(f"{'全面测试版 (当前)':<25} {'2024全年':<15} {f'{len(all_results)}':<10} {f'{avg_return*100:.2f}%':<12} {f'{avg_trades:.1f}':<10} {f'{win_rate*100:.2f}%':<10}")


if __name__ == "__main__":
    run_comprehensive_test()
