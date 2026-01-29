"""
使用最近3个月真实数据进行模拟盘测试

验证ML策略在真实数据上的表现
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


class PaperTrader:
    """模拟盘交易器"""

    def __init__(self, initial_capital=100000, model_path='models/xgboost_robust.json'):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None  # 当前持仓
        self.trades = []  # 交易记录
        self.daily_values = []  # 每日净值

        # 加载模型
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.feature_extractor = EnhancedFeatureExtractor()

        logger.info(f"模拟盘初始化: 初始资金={initial_capital:,.2f}")

    def generate_signal(self, features_df, i):
        """
        根据模型预测生成单个交易信号

        Args:
            features_df: 特征DataFrame
            i: 行索引

        Returns:
            Signal或None
        """
        # 准备特征
        feature_cols = [c for c in features_df.columns if c.startswith('f_')]
        if len(feature_cols) == 0:
            return None

        X = features_df[feature_cols].iloc[i:i+1].values

        # 预测
        prob = self.model.predict(xgb.DMatrix(X))[0]

        # 只有高置信度才交易
        if prob > 0.6:  # 买入
            return Signal(
                date=features_df['trade_date'].iloc[i].strftime('%Y%m%d'),
                signal_type=SignalType.BUY,
                price=features_df['close'].iloc[i],
                reason=f"ML预测上涨概率={prob:.2f}",
                confidence=prob
            )
        elif prob < 0.4:  # 卖出
            return Signal(
                date=features_df['trade_date'].iloc[i].strftime('%Y%m%d'),
                signal_type=SignalType.SELL,
                price=features_df['close'].iloc[i],
                reason=f"ML预测下跌概率={1-prob:.2f}",
                confidence=1-prob
            )
        return None

    def run_daily(self, df, features_df):
        """
        按日运行模拟交易

        Args:
            df: 原始OHLCV数据
            features_df: 特征数据

        Returns:
            交易次数
        """
        trade_count = 0

        for i in range(len(features_df)):
            date = features_df['trade_date'].iloc[i]
            close = features_df['close'].iloc[i]

            # 计算当前净值（持仓市值 + 现金）
            if self.position:
                market_value = self.position['quantity'] * close
                total_value = self.capital + market_value
            else:
                total_value = self.capital

            self.daily_values.append({
                'date': date,
                'total_value': total_value,
                'cash': self.capital,
                'position_value': market_value if self.position else 0,
                'position': self.position['quantity'] if self.position else 0,
            })

            # 生成交易信号（使用之前的数据预测）
            if i >= 50:  # 确保有足够的历史数据
                signal = self.generate_signal(features_df, i)

                if signal:
                    if signal.signal_type == SignalType.BUY and self.position is None:
                        # 买入
                        quantity = int(self.capital / signal.price / 100) * 100  # 整手
                        if quantity >= 100:  # 至少一手
                            cost = quantity * signal.price
                            self.capital -= cost
                            self.position = {
                                'entry_date': signal.date,
                                'entry_price': signal.price,
                                'quantity': quantity,
                            }
                            trade_count += 1
                            logger.debug(f"买入: {date} {quantity}股 @{signal.price:.2f}")

                    elif signal.signal_type == SignalType.SELL and self.position:
                        # 卖出
                        pnl = (signal.price - self.position['entry_price']) * self.position['quantity']
                        pnl_ratio = (signal.price - self.position['entry_price']) / self.position['entry_price']

                        self.capital += self.position['quantity'] * signal.price

                        self.trades.append({
                            'entry_date': self.position['entry_date'],
                            'exit_date': signal.date,
                            'entry_price': self.position['entry_price'],
                            'exit_price': signal.price,
                            'quantity': self.position['quantity'],
                            'pnl': pnl,
                            'pnl_ratio': pnl_ratio,
                        })

                        logger.debug(f"卖出: {date} {self.position['quantity']}股 @{signal.price:.2f} PNL={pnl:.2f} ({pnl_ratio*100:.2f}%)")

                        self.position = None
                        trade_count += 1

        # 如果最后还有持仓，用最后一天的价格平仓
        if self.position:
            last_date = features_df['trade_date'].iloc[-1]
            last_price = features_df['close'].iloc[-1]
            pnl = (last_price - self.position['entry_price']) * self.position['quantity']
            pnl_ratio = (last_price - self.position['entry_price']) / self.position['entry_price']

            self.capital += self.position['quantity'] * last_price

            self.trades.append({
                'entry_date': self.position['entry_date'],
                'exit_date': last_date.strftime('%Y%m%d'),
                'entry_price': self.position['entry_price'],
                'exit_price': last_price,
                'quantity': self.position['quantity'],
                'pnl': pnl,
                'pnl_ratio': pnl_ratio,
                'reason': '期末平仓',
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
            sharpe_ratio = sharpe * np.sqrt(252)  # 年化

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

    def print_summary(self, metrics, symbol=None):
        """打印绩效摘要"""
        title = f"模拟盘报告: {symbol}" if symbol else "模拟盘报告"
        print("=" * 70)
        print(title)
        print("=" * 70)
        print(f"初始资金: {metrics['initial_capital']:,.2f}")
        print(f"最终资金: {metrics['final_value']:,.2f}")
        print(f"总收益率: {metrics['total_return']*100:.2f}%")
        if metrics['benchmark_return'] != 0:
            print(f"基准收益: {metrics['benchmark_return']*100:.2f}%")
            print(f"超额收益: {metrics['excess_return']*100:.2f}%")
        print("-" * 70)
        print(f"交易次数: {metrics['trade_count']}")
        print(f"盈利次数: {metrics['win_count']}")
        print(f"亏损次数: {metrics['lose_count']}")
        print(f"胜率: {metrics['win_rate']*100:.2f}%")
        print(f"平均收益: {metrics['avg_return']*100:.2f}%")
        print("-" * 70)
        print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print("=" * 70)


def run_paper_trading():
    """运行模拟盘"""
    print("=" * 80)
    print("ML策略模拟盘测试（最近3个月真实数据）")
    print("=" * 80)

    # 初始化
    storage = SQLiteStorage()

    # 使用最近可用的3个月数据（2024年10月-12月）
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2024, 10, 1)

    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    print(f"\n测试期间: {start_str} ~ {end_str}")
    print(f"测试时长: 约3个月")

    # 获取股票列表
    print("\n[1/4] 获取股票列表...")
    stocks = storage.get_all_stocks()
    print(f"  可用股票: {len(stocks)} 只")

    # 选择流动性好的股票（随机采样一部分）
    np.random.seed(42)
    test_stocks = sorted(stocks)[:100]  # 使用前100只股票
    print(f"  测试股票: {len(test_stocks)} 只")

    # 存储所有结果
    all_results = []

    print(f"\n[2/4] 运行模拟盘...")

    for i, ts_code in enumerate(test_stocks):
        if (i + 1) % 10 == 0:
            print(f"  处理中: {i+1}/{len(test_stocks)}")

        try:
            # 获取数据（需要额外50天用于特征计算）
            df = storage.get_daily_prices(
                ts_code,
                (start_date - timedelta(days=60)).strftime('%Y%m%d'),
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
            trader = PaperTrader(initial_capital=100000)
            trader.run_daily(df, features)

            # 计算基准（买入持有）
            buy_hold_return = (features['close'].iloc[-1] - features['close'].iloc[0]) / features['close'].iloc[0]

            # 获取指标
            metrics = trader.get_metrics(benchmark_values=features['close'].values)

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

    # 找出表现最好的股票
    print(f"\n[4/4] 表现最好的股票 (Top 10):")
    sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)

    print(f"\n{'股票':<12} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 70)

    for r in sorted_results[:10]:
        excess = (r['total_return'] - r['buy_hold_return']) * 100
        print(f"{r['symbol']:<12} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6} {r['win_rate']*100:>6.1f}%")

    # 找出表现最差的股票
    print(f"\n表现最差的股票 (Bottom 5):")
    print(f"\n{'股票':<12} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易':<6} {'胜率':<8}")
    print("-" * 70)

    for r in sorted_results[-5:]:
        excess = (r['total_return'] - r['buy_hold_return']) * 100
        print(f"{r['symbol']:<12} {r['total_return']*100:>10.2f}%   {r['buy_hold_return']*100:>10.2f}%   {excess:>+8.2f}%   {r['trade_count']:<6} {r['win_rate']*100:>6.1f}%")

    # 保存详细结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('total_return', ascending=False)

    os.makedirs('backtest_results', exist_ok=True)
    results_df.to_csv('backtest_results/paper_trading_3months.csv', index=False)

    print(f"\n✓ 详细结果已保存到: backtest_results/paper_trading_3months.csv")

    # 打印一份详细报告
    if sorted_results:
        best = sorted_results[0]
        print(f"\n详细报告 - 表现最好的股票 ({best['symbol']}):")
        print("=" * 70)
        print(f"测试期间: {start_str} ~ {end_str}")
        print(f"策略收益: {best['total_return']*100:.2f}%")
        print(f"基准收益: {best['buy_hold_return']*100:.2f}%")
        print(f"超额收益: {(best['total_return']-best['buy_hold_return'])*100:.2f}%")
        print(f"交易次数: {best['trade_count']}")
        print(f"胜率: {best['win_rate']*100:.2f}%")
        print(f"最大回撤: {best['max_drawdown']*100:.2f}%")
        print(f"夏普比率: {best['sharpe_ratio']:.2f}")
        print("=" * 70)


if __name__ == "__main__":
    run_paper_trading()
