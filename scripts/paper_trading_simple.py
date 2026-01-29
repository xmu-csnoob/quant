"""
简化优化版模拟盘 - 固定低阈值策略

优化：
1. 降低固定阈值（0.52）
2. 简化交易逻辑
3. 加入交易成本
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
from strategies.base import Signal, SignalType
import xgboost as xgb
from loguru import logger
import os


class SimpleOptimizedTrader:
    """简化优化版交易器"""

    def __init__(
        self,
        initial_capital=100000,
        model_path='models/xgboost_robust.json',
        buy_threshold=0.52,  # 固定买入阈值
        sell_threshold=0.48,  # 固定卖出阈值
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

    def generate_signals(self, features_df):
        """生成所有信号"""
        feature_cols = [c for c in features_df.columns if c.startswith('f_')]
        if len(feature_cols) == 0:
            return []

        X = features_df[feature_cols].values
        probs = self.model.predict(xgb.DMatrix(X))

        signals = []
        for i in range(len(features_df)):
            prob = probs[i]

            if prob > self.buy_threshold:
                signals.append({
                    'index': i,
                    'date': features_df['trade_date'].iloc[i],
                    'type': 'BUY',
                    'price': features_df['close'].iloc[i],
                    'prob': prob,
                })
            elif prob < self.sell_threshold:
                signals.append({
                    'index': i,
                    'date': features_df['trade_date'].iloc[i],
                    'type': 'SELL',
                    'price': features_df['close'].iloc[i],
                    'prob': prob,
                })

        return signals

    def run_backtest(self, df, features_df):
        """运行回测"""
        signals = self.generate_signals(features_df)

        # 按日期排序
        signals = sorted(signals, key=lambda x: x['index'])

        for signal in signals:
            date = signal['date']
            close = signal['price']

            # 更新净值
            if self.position:
                market_value = self.position['quantity'] * close
                total_value = self.capital + market_value
            else:
                total_value = self.capital
                market_value = 0

            self.daily_values.append({
                'date': date,
                'total_value': total_value,
                'cash': self.capital,
                'position_value': market_value,
                'position': self.position['quantity'] if self.position else 0,
            })

            # 执行交易
            if signal['type'] == 'BUY' and self.position is None:
                # 买入
                buy_value = self.capital * 0.95  # 95%仓位
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

            elif signal['type'] == 'SELL' and self.position:
                # 卖出
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


def run_simple_optimized():
    """运行简化优化版"""
    print("=" * 80)
    print("简化优化版ML策略模拟盘")
    print("=" * 80)

    storage = SQLiteStorage()

    start_date = datetime(2024, 10, 1)
    end_date = datetime(2024, 12, 31)
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    print(f"\n测试期间: {start_str} ~ {end_str}")
    print(f"策略参数:")
    print(f"  买入阈值: 0.52 (降低以增加交易)")
    print(f"  卖出阈值: 0.48")
    print(f"  仓位: 95%")
    print(f"  交易成本: 0.03%")

    print("\n[1/3] 获取股票...")
    stocks = storage.get_all_stocks()
    np.random.seed(42)
    test_stocks = sorted(stocks)[:100]
    print(f"  测试股票: {len(test_stocks)} 只")

    print(f"\n[2/3] 运行回测...")

    all_results = []

    for i, ts_code in enumerate(test_stocks):
        if (i + 1) % 10 == 0:
            print(f"  处理中: {i+1}/{len(test_stocks)}")

        try:
            df = storage.get_daily_prices(
                ts_code,
                (start_date - timedelta(days=90)).strftime('%Y%m%d'),
                end_str
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

            if len(features) < 20:
                continue

            trader = SimpleOptimizedTrader(initial_capital=100000)
            trader.run_backtest(df, features)

            buy_hold_return = (features['close'].iloc[-1] - features['close'].iloc[0]) / features['close'].iloc[0]
            metrics = trader.get_metrics(features['close'].values)

            if metrics and metrics['trade_count'] > 0:
                metrics['symbol'] = ts_code
                metrics['buy_hold_return'] = buy_hold_return
                all_results.append(metrics)

        except Exception as e:
            logger.warning(f"回测 {ts_code} 失败: {e}")
            continue

    print(f"  ✓ 完成 {len(all_results)} 只股票")

    print(f"\n[3/3] 分析结果...")

    if not all_results:
        print("  无有效结果")
        return

    avg_return = np.mean([r['total_return'] for r in all_results])
    avg_trades = np.mean([r['trade_count'] for r in all_results])
    win_rate = np.mean([r['win_rate'] for r in all_results])
    beat_market = sum(1 for r in all_results if r['total_return'] > r['buy_hold_return'])
    avg_buy_hold = np.mean([r['buy_hold_return'] for r in all_results])

    print(f"\n整体表现 ({len(all_results)} 只股票):")
    print(f"  平均收益: {avg_return*100:.2f}%")
    print(f"  买入持有: {avg_buy_hold*100:.2f}%")
    print(f"  超额收益: {(avg_return-avg_buy_hold)*100:.2f}%")
    print(f"  跑赢市场: {beat_market}/{len(all_results)} ({beat_market/len(all_results)*100:.1f}%)")
    print(f"  平均胜率: {win_rate*100:.2f}%")
    print(f"  平均交易次数: {avg_trades:.1f}")

    sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)

    print(f"\n表现最好 (Top 10):")
    print(f"\n{'股票':<12} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 70)

    for r in sorted_results[:10]:
        excess = (r['total_return'] - r['buy_hold_return']) * 100
        print(f"{r['symbol']:<12} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6} {r['win_rate']*100:>6.1f}%")

    print(f"\n{'='*70}")
    print("版本对比")
    print(f"{'='*70}")
    print(f"{'版本':<20} {'交易次数':<12} {'胜率':<12} {'超额收益':<12}")
    print("-" * 50)
    print(f"{'原版 (阈值0.6)':<20} {'1.0':<12} {'53.33%':<12} {'+5.89%':<12}")
    print(f"{'动态阈值版':<20} {'1.0':<12} {'45.24%':<12} {'+4.78%':<12}")
    print(f"{'固定低阈值版':<20} {f'{avg_trades:.1f}':<12} {f'{win_rate*100:.2f}%':<12} {f'{(avg_return-avg_buy_hold)*100:+.2f}%':<12}")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('total_return', ascending=False)

    os.makedirs('backtest_results', exist_ok=True)
    results_df.to_csv('backtest_results/paper_trading_simple_optimized.csv', index=False)

    print(f"\n✓ 详细结果已保存到: backtest_results/paper_trading_simple_optimized.csv")


if __name__ == "__main__":
    run_simple_optimized()
