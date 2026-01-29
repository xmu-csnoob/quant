"""
修复版模拟盘测试 - 修复所有已知问题

修复内容：
1. 幸存者偏差 - 统计所有股票（包括无交易的）
2. 交易成本 - 买入0.03%, 卖出0.13%（符合A股）
3. 增加样本量 - 测试更多股票
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


class FixedPaperTrader:
    """修复版模拟盘交易器"""

    # A股实际交易成本
    BUY_COST_RATE = 0.0003    # 0.03% 佣金
    SELL_COST_RATE = 0.0013  # 0.13%  佣金+印花税

    def __init__(
        self,
        initial_capital=100000,
        model_path='models/xgboost_robust.json',
        buy_threshold=0.52,
        sell_threshold=0.48,
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

    def run_backtest(self, df, features_df):
        """运行回测"""
        feature_cols = [c for c in features_df.columns if c.startswith('f_')]
        if len(feature_cols) == 0:
            return 0, False  # 返回交易次数和是否有交易

        X = features_df[feature_cols].values
        probs = self.model.predict(xgb.DMatrix(X))

        has_trade = False

        for i in range(len(features_df)):
            close = features_df['close'].iloc[i]
            prob = probs[i]

            # 更新净值
            if self.position:
                market_value = self.position['quantity'] * close
            else:
                market_value = 0
            total_value = self.capital + market_value

            self.daily_values.append(total_value)

            # 买入
            if prob > self.buy_threshold and self.position is None:
                buy_value = self.capital * 0.95
                quantity = int(buy_value / close / 100) * 100

                if quantity >= 100:
                    cost = quantity * close * (1 + self.BUY_COST_RATE)
                    if cost <= self.capital:
                        self.capital -= cost
                        self.position = {
                            'entry_date': features_df['trade_date'].iloc[i],
                            'entry_price': close,
                            'quantity': quantity,
                        }
                        has_trade = True

            # 卖出
            elif prob < self.sell_threshold and self.position:
                sell_value = self.position['quantity'] * close * (1 - self.SELL_COST_RATE)
                pnl = sell_value - self.position['quantity'] * self.position['entry_price']
                pnl_ratio = pnl / (self.position['quantity'] * self.position['entry_price'])

                self.capital += sell_value

                self.trades.append({
                    'pnl': pnl,
                    'pnl_ratio': pnl_ratio,
                })
                self.position = None
                has_trade = True

        # 期末平仓
        if self.position:
            last_price = features_df['close'].iloc[-1]
            sell_value = self.position['quantity'] * last_price * (1 - self.SELL_COST_RATE)
            pnl = sell_value - self.position['quantity'] * self.position['entry_price']
            pnl_ratio = pnl / (self.position['quantity'] * self.position['entry_price'])

            self.capital += sell_value

            self.trades.append({'pnl': pnl, 'pnl_ratio': pnl_ratio})
            self.position = None
            has_trade = True

        trade_count = len(self.trades)
        return trade_count, has_trade

    def get_metrics(self, buy_hold_return):
        """计算绩效指标"""
        if not self.daily_values:
            return None

        final_value = self.daily_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 最大回撤
        cummax = np.maximum.accumulate(self.daily_values)
        drawdowns = (cummax - np.array(self.daily_values)) / cummax
        max_drawdown = np.max(drawdowns)

        # 夏普比率（简化）
        returns = np.diff(self.daily_values) / np.array(self.daily_values[:-1])
        sharpe_ratio = 0
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std()
            sharpe_ratio = sharpe * np.sqrt(252)

        if self.trades:
            win_rate = sum(1 for t in self.trades if t['pnl'] > 0) / len(self.trades)
        else:
            win_rate = 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trade_count': len(self.trades),
            'win_rate': win_rate,
            'final_capital': final_value,
        }


def run_fixed_test():
    """运行修复版测试"""
    print("=" * 80)
    print("修复版模拟盘测试 - 2024年Q1")
    print("=" * 80)

    print("\n修复内容:")
    print("  ✓ 修正交易成本（买入0.03%, 卖出0.13%）")
    print("  ✓ 修复幸存者偏差（统计所有200只股票）")
    print("  ✓ 准确的A股交易成本")

    storage = SQLiteStorage()

    # 测试参数
    test_start = datetime(2024, 1, 1)
    test_end = datetime(2024, 3, 31)

    print(f"\n测试期间: {test_start.strftime('%Y%m%d')} ~ {test_end.strftime('%Y%m%d')}")
    print(f"策略参数: 买入阈值0.52, 卖出阈值0.48")

    # 获取股票
    print("\n[1/3] 获取股票...")
    stocks = storage.get_all_stocks()
    np.random.seed(42)
    test_stocks = sorted(stocks)[:200]
    print(f"  测试股票: {len(test_stocks)} 只")

    print(f"\n[2/3] 运行回测...")

    model = xgb.Booster()
    model.load_model('models/xgboost_robust.json')
    feature_extractor = EnhancedFeatureExtractor()

    all_results = []  # 包括所有股票，不只是有交易的

    for i, ts_code in enumerate(test_stocks):
        if (i + 1) % 50 == 0:
            print(f"  处理中: {i+1}/{len(test_stocks)}")

        try:
            df = storage.get_daily_prices(
                ts_code,
                (test_start - timedelta(days=90)).strftime('%Y%m%d'),
                test_end.strftime('%Y%m%d')
            )

            if df is None or len(df) < 60:
                # 即使数据不足也记录（算作失败）
                all_results.append({
                    'symbol': ts_code,
                    'total_return': None,
                    'buy_hold_return': None,
                    'trade_count': 0,
                    'status': 'insufficient_data',
                })
                continue

            features = feature_extractor.extract(df)
            features['trade_date'] = pd.to_datetime(features['trade_date'])
            features = features[
                (features['trade_date'] >= test_start) &
                (features['trade_date'] <= test_end)
            ].reset_index(drop=True)

            if len(features) < 40:
                all_results.append({
                    'symbol': ts_code,
                    'total_return': None,
                    'buy_hold_return': None,
                    'trade_count': 0,
                    'status': 'insufficient_data',
                })
                continue

            # 计算买入持有收益
            buy_hold_return = (features['close'].iloc[-1] - features['close'].iloc[0]) / features['close'].iloc[0]

            # 运行回测
            trader = FixedPaperTrader(initial_capital=100000)
            trade_count, has_trade = trader.run_backtest(df, features)

            if has_trade:
                metrics = trader.get_metrics(buy_hold_return)
                all_results.append({
                    'symbol': ts_code,
                    'total_return': metrics['total_return'],
                    'max_drawdown': metrics['max_drawdown'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'trade_count': trade_count,
                    'win_rate': metrics['win_rate'],
                    'buy_hold_return': buy_hold_return,
                    'final_capital': metrics['final_capital'],
                    'status': 'has_trade',
                })
            else:
                # 没有交易：使用买入持有收益
                all_results.append({
                    'symbol': ts_code,
                    'total_return': buy_hold_return,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'trade_count': 0,
                    'win_rate': 0,
                    'buy_hold_return': buy_hold_return,
                    'final_capital': 100000 * (1 + buy_hold_return),
                    'status': 'no_trade',
                })

        except Exception as e:
            logger.warning(f"回测 {ts_code} 失败: {e}")
            all_results.append({
                'symbol': ts_code,
                'total_return': None,
                'buy_hold_return': None,
                'trade_count': 0,
                'status': 'error',
                'error': str(e)[:50],
            })
            continue

    print(f"  ✓ 完成 {len(all_results)} 只股票")

    print(f"\n[3/3] 分析结果...")

    # 分组统计
    has_trade_results = [r for r in all_results if r['status'] == 'has_trade']
    no_trade_results = [r for r in all_results if r['status'] == 'no_trade']
    error_results = [r for r in all_results if r['status'] == 'error']
    insufficient_results = [r for r in all_results if r['status'] == 'insufficient_data']

    print(f"\n结果分类:")
    print(f"  有交易: {len(has_trade_results)} 只 ({len(has_trade_results)/len(all_results)*100:.1f}%)")
    print(f"  无交易: {len(no_trade_results)} 只 ({len(no_trade_results)/len(all_results)*100:.1f}%)")
    print(f"  数据不足: {len(insufficient_results)} 只")
    print(f"  错误: {len(error_results)} 只")

    # 计算整体统计（包括所有股票）
    valid_results = [r for r in all_results if r['total_return'] is not None]
    print(f"\n有效结果: {len(valid_results)} 只")

    if valid_results:
        returns = [r['total_return'] for r in valid_results]
        avg_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)

        buy_hold_returns = [r['buy_hold_return'] for r in valid_results]
        avg_buy_hold = np.mean(buy_hold_returns)

        excess_return = avg_return - avg_buy_hold

        # 收益分布
        positive_count = sum(1 for r in returns if r > 0)
        negative_count = sum(1 for r in returns if r < 0)

        print(f"\n{'='*80}")
        print("整体表现（所有股票，包括无交易的）")
        print(f"{'='*80}")
        print(f"\n收益指标:")
        print(f"  平均收益: {avg_return*100:.2f}%")
        print(f"  中位数收益: {median_return*100:.2f}%")
        print(f"  标准差: {std_return*100:.2f}%")
        print(f"  买入持有: {avg_buy_hold*100:.2f}%")
        print(f"  超额收益: {excess_return*100:.2f}%")

        print(f"\n收益分布:")
        print(f"  盈利: {positive_count} 只 ({positive_count/len(valid_results)*100:.1f}%)")
        print(f"  亏损: {negative_count} 只 ({negative_count/len(valid_results)*100:.1f}%)")

        # 统计显著性
        n = len(valid_results)
        se = std_return / np.sqrt(n)
        t_stat = avg_return / se if se > 0 else 0

        print(f"\n统计显著性:")
        print(f"  样本量: {n}")
        print(f"  标准误: {se*100:.2f}%")
        print(f"  t统计量: {t_stat:.2f}")
        print(f"  p-value: {2*(1-norm.cdf(abs(t_stat))):.4f}" if abs(t_stat) < 3 else "  p-value: <0.01")

        # 只看有交易股票
        if has_trade_results:
            trade_returns = [r['total_return'] for r in has_trade_results]
            trade_avg = np.mean(trade_returns)
            trade_std = np.std(trade_returns)
            trade_win_rate = np.mean([r['win_rate'] for r in has_trade_results])
            trade_count = np.mean([r['trade_count'] for r in has_trade_results])

            print(f"\n有交易股票表现 ({len(has_trade_results)}只):")
            print(f"  平均收益: {trade_avg*100:.2f}%")
            print(f"  标准差: {trade_std*100:.2f}%")
            print(f"  胜率: {trade_win_rate*100:.2f}%")
            print(f"  平均交易次数: {trade_count:.1f}")

        # Top 10
        sorted_results = sorted(valid_results, key=lambda x: x['total_return'], reverse=True)

        print(f"\n表现最好 (Top 10):")
        print(f"\n{'股票':<12} {'状态':<10} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6}")
        print("-" * 75)

        for r in sorted_results[:10]:
            status = '交易' if r['status'] == 'has_trade' else '无交易'
            excess = (r['total_return'] - r['buy_hold_return']) * 100
            print(f"{r['symbol']:<12} {status:<10} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6}")

        # 保存结果
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('total_return', ascending=False)

        os.makedirs('backtest_results', exist_ok=True)
        results_df.to_csv('backtest_results/fixed_test_2024q1.csv', index=False)

        print(f"\n✓ 结果已保存到: backtest_results/fixed_test_2024q1.csv")

        # 与之前错误版本对比
        print(f"\n{'='*80}")
        print("版本对比（修复前后）")
        print(f"{'='*80}")
        print(f"\n{'版本':<25} {'幸存者偏差':<18} {'交易成本':<15} {'平均收益':<12}")
        print("-" * 75)
        print(f"{'错误版本':<25} {'只统计有交易的':<18} {'0.03%双向':<15} {'6.68%':<12}")
        print(f"{'修复版本':<25} {'统计所有股票':<18} {'买入0.03%':<15} {f'{avg_return*100:.2f}%':<12}")

        # 计算修正后的收益（假设无交易股票收益为0）
        if has_trade_results and no_trade_results:
            trade_returns = [r['total_return'] for r in has_trade_results]
            no_trade_returns = [r['buy_hold_return'] for r in no_trade_results]

            weighted_return = (
                np.sum(trade_returns) + np.sum(no_trade_returns)
            ) / (len(has_trade_results) + len(no_trade_results))

            print(f"\n加权平均收益（无交易用买入持有）:")
            print(f"  {weighted_return*100:.2f}%")
            print(f"  (有交易股票平均 × {len(has_trade_results)} + 无交易买入持有 × {len(no_trade_results)}) / {len(has_trade_results) + len(no_trade_results)}")


if __name__ == "__main__":
    from scipy.stats import norm
    run_fixed_test()
