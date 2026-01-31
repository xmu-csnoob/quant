"""
优化版模拟盘测试

改进：
1. 动态置信度阈值
2. 市场环境判断
3. 仓位管理
4. T+1限制
5. 交易成本
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


class OptimizedPaperTrader:
    """优化版模拟盘交易器"""

    def __init__(
        self,
        initial_capital=100000,
        model_path='models/xgboost_robust.json',
        base_threshold=0.55,  # 降低基础阈值
        transaction_cost=0.0003,  # 0.03%交易成本（佣金+印花税）
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.daily_values = []

        # 加载模型
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.feature_extractor = EnhancedFeatureExtractor()

        # 参数
        self.base_threshold = base_threshold
        self.transaction_cost = transaction_cost

        logger.info(f"优化模拟盘初始化: 初始资金={initial_capital:,.2f}, 阈值={base_threshold}")

    def get_market_regime(self, df, window=20):
        """
        判断市场环境

        Returns:
            regime: 'bull' (牛市), 'bear' (熊市), 'neutral' (中性)
            strength: 0-1, 趋势强度
        """
        if len(df) < window:
            return 'neutral', 0.5

        recent = df.tail(window)

        # 计算趋势
        start_price = recent['close'].iloc[0]
        end_price = recent['close'].iloc[-1]
        total_return = (end_price - start_price) / start_price

        # 计算波动率
        returns = recent['close'].pct_change().dropna()
        volatility = returns.std()

        # 判断趋势强度
        trend_strength = min(abs(total_return) / (volatility * np.sqrt(window) + 0.01), 1.0)

        if total_return > 0.02:  # 上涨超过2%
            return 'bull', trend_strength
        elif total_return < -0.02:  # 下跌超过2%
            return 'bear', trend_strength
        else:
            return 'neutral', trend_strength

    def get_dynamic_threshold(self, market_regime, trend_strength):
        """
        根据市场环境动态调整阈值

        牛市: 降低阈值，积极参与
        熊市: 提高阈值，更加保守
        """
        if market_regime == 'bull':
            # 牛市降低阈值，最低0.5
            return max(0.50, self.base_threshold - 0.05 * trend_strength)
        elif market_regime == 'bear':
            # 熊市提高阈值，最高0.65
            return min(0.65, self.base_threshold + 0.05 * trend_strength)
        else:
            return self.base_threshold

    def calculate_position_size(self, confidence, market_regime, trend_strength):
        """
        根据置信度和市场环境计算仓位

        Returns:
            position_ratio: 0-1, 仓位比例
        """
        base_ratio = 0.95  # 默认95%仓位

        # 根据置信度调整
        if market_regime == 'bull':
            # 牛市高置信度可以满仓
            if confidence > 0.7:
                return 1.0
            else:
                return base_ratio * confidence
        elif market_regime == 'bear':
            # 熊市降低仓位
            return base_ratio * 0.7 * confidence
        else:
            return base_ratio * confidence

    def generate_signal(self, features_df, i, market_regime, trend_strength):
        """生成交易信号"""
        # 准备特征
        feature_cols = [c for c in features_df.columns if c.startswith('f_')]
        if len(feature_cols) == 0:
            return None

        X = features_df[feature_cols].iloc[i:i+1].values

        # 预测
        prob = self.model.predict(xgb.DMatrix(X))[0]

        # 动态阈值
        buy_threshold = self.get_dynamic_threshold(market_regime, trend_strength)
        sell_threshold = 1 - buy_threshold

        # 生成信号
        if prob > buy_threshold:
            return Signal(
                date=features_df['trade_date'].iloc[i].strftime('%Y%m%d'),
                signal_type=SignalType.BUY,
                price=features_df['close'].iloc[i],
                reason=f"ML预测上涨概率={prob:.2f} (阈值={buy_threshold:.2f})",
                confidence=prob
            )
        elif prob < sell_threshold:
            return Signal(
                date=features_df['trade_date'].iloc[i].strftime('%Y%m%d'),
                signal_type=SignalType.SELL,
                price=features_df['close'].iloc[i],
                reason=f"ML预测下跌概率={1-prob:.2f} (阈值={sell_threshold:.2f})",
                confidence=1-prob
            )
        return None

    def run_daily(self, df, features_df):
        """按日运行模拟交易"""
        trade_count = 0
        last_signal_date = None  # T+1限制

        for i in range(len(features_df)):
            date = features_df['trade_date'].iloc[i]
            close = features_df['close'].iloc[i]

            # 计算当前净值
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

            # 判断市场环境（使用前60天数据）
            market_df = features_df.iloc[max(0, i-60):i]
            market_regime, trend_strength = self.get_market_regime(market_df)

            # 生成信号（T+1限制：需要间隔一天）
            can_trade = (last_signal_date is None or
                         (date - last_signal_date).days >= 2)

            if i >= 50 and can_trade:
                signal = self.generate_signal(features_df, i, market_regime, trend_strength)

                if signal:
                    if signal.signal_type == SignalType.BUY and self.position is None:
                        # 计算仓位
                        position_ratio = self.calculate_position_size(
                            signal.confidence, market_regime, trend_strength
                        )

                        # 计算买入数量
                        buy_value = self.capital * position_ratio
                        quantity = int(buy_value / signal.price / 100) * 100  # 整手

                        if quantity >= 100:
                            cost = quantity * signal.price * (1 + self.transaction_cost)
                            if cost <= self.capital:
                                self.capital -= cost
                                self.position = {
                                    'entry_date': signal.date,
                                    'entry_price': signal.price,
                                    'quantity': quantity,
                                }
                                last_signal_date = date
                                trade_count += 1
                                logger.debug(f"买入: {date} {quantity}股 @{signal.price:.2f} 仓位={position_ratio:.1%}")

                    elif signal.signal_type == SignalType.SELL and self.position:
                        # 卖出
                        sell_value = self.position['quantity'] * signal.price * (1 - self.transaction_cost)
                        pnl = sell_value - self.position['quantity'] * self.position['entry_price']
                        pnl_ratio = pnl / (self.position['quantity'] * self.position['entry_price'])

                        self.capital += sell_value

                        self.trades.append({
                            'entry_date': self.position['entry_date'],
                            'exit_date': signal.date,
                            'entry_price': self.position['entry_price'],
                            'exit_price': signal.price,
                            'quantity': self.position['quantity'],
                            'pnl': pnl,
                            'pnl_ratio': pnl_ratio,
                            'market_regime': market_regime,
                        })

                        logger.debug(f"卖出: {date} {self.position['quantity']}股 @{signal.price:.2f} PNL={pnl:.2f} ({pnl_ratio*100:.2f}%)")

                        self.position = None
                        last_signal_date = date

        # 期末平仓
        if self.position:
            last_date = features_df['trade_date'].iloc[-1]
            last_price = features_df['close'].iloc[-1]
            sell_value = self.position['quantity'] * last_price * (1 - self.transaction_cost)
            pnl = sell_value - self.position['quantity'] * self.position['entry_price']
            pnl_ratio = pnl / (self.position['quantity'] * self.position['entry_price'])

            self.capital += sell_value

            self.trades.append({
                'entry_date': self.position['entry_date'],
                'exit_date': last_date.strftime('%Y%m%d'),
                'entry_price': self.position['entry_price'],
                'exit_price': last_price,
                'quantity': self.position['quantity'],
                'pnl': pnl,
                'pnl_ratio': pnl_ratio,
                'market_regime': 'end',
            })

            logger.debug(f"期末平仓: {last_date} @{last_price:.2f} PNL={pnl:.2f} ({pnl_ratio*100:.2f}%)")

            self.position = None

        return trade_count

    def get_metrics(self, benchmark_values=None):
        """计算绩效指标"""
        if not self.daily_values:
            return None

        df_values = pd.DataFrame(self.daily_values)
        final_value = df_values['total_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 计算每日收益率
        df_values['daily_return'] = df_values['total_value'].pct_change()

        # 计算最大回撤
        df_values['cummax'] = df_values['total_value'].cummax()
        df_values['drawdown'] = (df_values['cummax'] - df_values['total_value']) / df_values['cummax']
        max_drawdown = df_values['drawdown'].max()

        # 计算夏普比率
        returns = df_values['daily_return'].dropna()
        sharpe_ratio = 0
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std()
            sharpe_ratio = sharpe * np.sqrt(252)

        # 交易统计
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

        # 与基准对比
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


def run_optimized_paper_trading():
    """运行优化版模拟盘"""
    print("=" * 80)
    print("优化版ML策略模拟盘测试")
    print("=" * 80)

    # 初始化
    storage = SQLiteStorage()

    # 测试期间
    start_date = datetime(2024, 10, 1)
    end_date = datetime(2024, 12, 31)

    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    print(f"\n测试期间: {start_str} ~ {end_str}")
    print(f"优化策略:")
    print(f"  - 动态置信度阈值 (基础: 0.55)")
    print(f"  - 市场环境判断 (牛市/熊市/中性)")
    print(f"  - 动态仓位管理")
    print(f"  - T+1交易限制")
    print(f"  - 交易成本 (0.03%)")

    # 获取股票列表
    print("\n[1/4] 获取股票列表...")
    stocks = storage.get_all_stocks()
    print(f"  可用股票: {len(stocks)} 只")

    # 测试股票
    np.random.seed(42)
    test_stocks = sorted(stocks)[:100]
    print(f"  测试股票: {len(test_stocks)} 只")

    # 存储结果
    all_results = []

    print(f"\n[2/4] 运行优化模拟盘...")

    for i, ts_code in enumerate(test_stocks):
        if (i + 1) % 10 == 0:
            print(f"  处理中: {i+1}/{len(test_stocks)}")

        try:
            # 获取数据
            df = storage.get_daily_prices(
                ts_code,
                (start_date - timedelta(days=90)).strftime('%Y%m%d'),
                end_str
            )

            if df is None or len(df) < 60:
                continue

            # 提取特征
            feature_extractor = EnhancedFeatureExtractor()
            features = feature_extractor.extract(df)

            # 筛选日期范围
            features['trade_date'] = pd.to_datetime(features['trade_date'])
            features = features[
                (features['trade_date'] >= start_date) &
                (features['trade_date'] <= end_date)
            ].reset_index(drop=True)

            if len(features) < 20:
                continue

            # 运行模拟盘
            trader = OptimizedPaperTrader(
                initial_capital=100000,
                base_threshold=0.55
            )
            trader.run_daily(df, features)

            # 计算基准
            buy_hold_return = (features['close'].iloc[-1] - features['close'].iloc[0]) / features['close'].iloc[0]

            # 获取指标
            metrics = trader.get_metrics(features['close'].values)

            if metrics and metrics['trade_count'] > 0:
                metrics['symbol'] = ts_code
                metrics['buy_hold_return'] = buy_hold_return
                all_results.append(metrics)

        except Exception as e:
            logger.warning(f"模拟盘 {ts_code} 失败: {e}")
            continue

    print(f"  ✓ 完成 {len(all_results)} 只股票的模拟盘")

    # 统计结果
    print(f"\n[3/4] 分析结果...")

    if not all_results:
        print("  无有效结果")
        return

    # 计算汇总
    avg_return = np.mean([r['total_return'] for r in all_results])
    median_return = np.median([r['total_return'] for r in all_results])
    win_rate = np.mean([r['win_rate'] for r in all_results])
    max_drawdown = np.mean([r['max_drawdown'] for r in all_results])
    sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
    avg_trades = np.mean([r['trade_count'] for r in all_results])

    # 与买入持有对比
    beat_market_count = sum(1 for r in all_results if r['total_return'] > r['buy_hold_return'])
    avg_buy_hold = np.mean([r['buy_hold_return'] for r in all_results])

    print(f"\n整体表现 ({len(all_results)} 只股票):")
    print(f"  平均收益: {avg_return*100:.2f}%")
    print(f"  中位数收益: {median_return*100:.2f}%")
    print(f"  买入持有收益: {avg_buy_hold*100:.2f}%")
    print(f"  超额收益: {(avg_return-avg_buy_hold)*100:.2f}%")
    print(f"  跑赢市场: {beat_market_count}/{len(all_results)} ({beat_market_count/len(all_results)*100:.1f}%)")
    print(f"  平均胜率: {win_rate*100:.2f}%")
    print(f"  平均最大回撤: {max_drawdown*100:.2f}%")
    print(f"  平均夏普比率: {sharpe:.2f}")
    print(f"  平均交易次数: {avg_trades:.1f}")

    # 表现最好和最差
    print(f"\n[4/4] 表现最好 (Top 10):")
    sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)

    print(f"\n{'股票':<12} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 70)

    for r in sorted_results[:10]:
        excess = (r['total_return'] - r['buy_hold_return']) * 100
        print(f"{r['symbol']:<12} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6} {r['win_rate']*100:>6.1f}%")

    print(f"\n表现最差 (Bottom 5):")
    print(f"\n{'股票':<12} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 70)

    for r in sorted_results[-5:]:
        excess = (r['total_return'] - r['buy_hold_return']) * 100
        print(f"{r['symbol']:<12} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6} {r['win_rate']*100:>6.1f}%")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('total_return', ascending=False)

    os.makedirs('backtest_results', exist_ok=True)
    results_df.to_csv('backtest_results/paper_trading_optimized.csv', index=False)

    print(f"\n✓ 详细结果已保存到: backtest_results/paper_trading_optimized.csv")

    # 对比分析
    print(f"\n{'='*80}")
    print("对比分析: 优化版 vs 原版")
    print(f"{'='*80}")
    print(f"{'指标':<20} {'原版':<15} {'优化版':<15} {'改进':<15}")
    print("-" * 70)
    print(f"{'交易次数':<20} {'1.0':<15} {avg_trades:<15.1f} {avg_trades-1.0:+.1f}")
    print(f"{'胜率':<20} {'53.33%':<15} {f'{win_rate*100:.2f}%':<15} {f'{(win_rate-0.5333)*100:+.2f}%'}")
    print(f"{'超额收益':<20} {'+5.89%':<15} {f'{(avg_return-avg_buy_hold)*100:+.2f}%':<15} {f'{((avg_return-avg_buy_hold)-0.0589)*100:+.2f}%'}")
    print(f"{'平均收益':<20} {'-1.10%':<15} {f'{avg_return*100:.2f}%':<15} {f'{(avg_return+0.011)*100:+.2f}%'}")


if __name__ == "__main__":
    run_optimized_paper_trading()
