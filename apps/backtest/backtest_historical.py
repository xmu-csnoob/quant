#!/usr/bin/env python3
"""
历史回测 - 从2025年1月1日开始
初始资金100万，确保无数据泄露
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor
import xgboost as xgb
from loguru import logger


class HistoricalBacktester:
    """历史回测器"""

    # A股交易成本
    BUY_COST_RATE = 0.0003    # 0.03% 佣金
    SELL_COST_RATE = 0.0013  # 0.13% 佣金+印花税

    # 策略参数
    BUY_THRESHOLD = 0.52
    SELL_THRESHOLD = 0.48
    MAX_POSITIONS = 3  # 最多同时持有3只股票
    POSITION_SIZE = 0.3  # 每只股票30%仓位

    def __init__(self, initial_capital=1000000):
        """
        初始化回测器

        Args:
            initial_capital: 初始资金，默认100万
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # {ts_code: {entry_date, entry_price, quantity}}
        self.trade_history = []
        self.daily_values = []

        # 加载模型
        self.model = xgb.Booster()
        self.model.load_model('models/xgboost_2022_2026.json')
        self.feature_extractor = EnhancedFeatureExtractor()

        # 数据存储
        self.storage = SQLiteStorage()

        # 获取股票池
        self.universe = sorted(self.storage.get_all_stocks())[:100]  # 前100只股票

        logger.info(f"回测初始化完成")
        logger.info(f"  初始资金: {initial_capital:,.2f} 元")
        logger.info(f"  股票池: {len(self.universe)} 只")
        logger.info(f"  回测期间: 2025-01-01 ~ 2025-01-29")

    def get_trading_dates(self, start_date, end_date):
        """获取所有交易日"""
        # 从第一只股票获取交易日列表
        df = self.storage.get_daily_prices(self.universe[0], start_date, end_date)
        if df is None or df.empty:
            return []
        return sorted(df['trade_date'].unique().tolist())

    def check_signals(self, ts_code, current_date):
        """
        检查交易信号（无数据泄露）

        Args:
            ts_code: 股票代码
            current_date: 当前决策日期（字符串YYYYMMDD）

        Returns:
            prob, price, date or None, None, None
        """
        try:
            # 获取历史数据（决策日之前的数据）
            end_date = pd.to_datetime(current_date).strftime('%Y%m%d')
            start_date = (pd.to_datetime(current_date) - timedelta(days=120)).strftime('%Y%m%d')

            df = self.storage.get_daily_prices(ts_code, start_date, end_date)

            if df is None or len(df) < 60:
                return None, None, None

            # 提取特征
            features = self.feature_extractor.extract(df)

            if len(features) < 1:
                return None, None, None

            # 获取最后一行（决策日当天）
            latest = features.iloc[-1]
            feature_cols = [c for c in features.columns if c.startswith('f_')]

            if len(feature_cols) == 0:
                return None, None, None

            X = latest[feature_cols].values.reshape(1, -1)
            prob = self.model.predict(xgb.DMatrix(X))[0]

            current_price = latest['close']
            current_date_obj = pd.to_datetime(latest['trade_date']).date()

            return prob, current_price, current_date_obj

        except Exception as e:
            logger.warning(f"检查 {ts_code} 信号失败: {e}")
            return None, None, None

    def execute_trade(self, ts_code, action, price, date, quantity=None):
        """执行交易"""
        if action == 'buy':
            if ts_code in self.positions:
                return  # 已持有

            if len(self.positions) >= self.MAX_POSITIONS:
                return

            # 计算买入数量
            buy_value = self.capital * self.POSITION_SIZE
            quantity = int(buy_value / price / 100) * 100  # 整手

            if quantity < 100:
                return

            cost = quantity * price * (1 + self.BUY_COST_RATE)

            if cost > self.capital:
                return

            # 执行买入
            self.capital -= cost
            self.positions[ts_code] = {
                'entry_date': str(date),
                'entry_price': price,
                'quantity': quantity
            }

            trade = {
                'date': str(date),
                'ts_code': ts_code,
                'action': 'buy',
                'price': price,
                'quantity': quantity,
                'amount': cost,
                'capital_after': self.capital
            }
            self.trade_history.append(trade)
            logger.info(f"  买入 {ts_code}: {quantity}股 @ {price:.2f}, 成本={cost:.2f}")

        elif action == 'sell':
            if ts_code not in self.positions:
                return

            pos = self.positions[ts_code]
            quantity = pos['quantity']

            # 卖出
            sell_value = quantity * price * (1 - self.SELL_COST_RATE)
            pnl = sell_value - quantity * pos['entry_price']
            pnl_ratio = pnl / (quantity * pos['entry_price'])

            self.capital += sell_value
            del self.positions[ts_code]

            trade = {
                'date': str(date),
                'ts_code': ts_code,
                'action': 'sell',
                'price': price,
                'quantity': quantity,
                'amount': sell_value,
                'pnl': pnl,
                'pnl_ratio': pnl_ratio,
                'capital_after': self.capital
            }
            self.trade_history.append(trade)
            logger.info(f"  卖出 {ts_code}: {quantity}股 @ {price:.2f}, 盈亏={pnl:.2f} ({pnl_ratio*100:.2f}%)")

    def run_backtest(self, start_date='20250101', end_date='20250129'):
        """运行回测"""
        logger.info("=" * 60)
        logger.info(f"历史回测: {start_date} ~ {end_date}")
        logger.info("=" * 60)

        # 获取所有交易日
        trading_dates = self.get_trading_dates(start_date, end_date)
        logger.info(f"交易日: {len(trading_dates)} 天")

        if not trading_dates:
            logger.error("没有找到交易日数据")
            return

        total_return = 0
        for i, trade_date in enumerate(trading_dates):
            logger.info(f"\n[{i+1}/{len(trading_dates)}] 交易日期: {trade_date}")

            # 检查所有股票的信号
            buy_signals = []
            sell_signals = []

            for ts_code in self.universe:
                prob, price, date = self.check_signals(ts_code, trade_date)

                if prob is None:
                    continue

                # 买入信号
                if prob > self.BUY_THRESHOLD and ts_code not in self.positions:
                    buy_signals.append((ts_code, prob, price, date))

                # 卖出信号
                elif prob < self.SELL_THRESHOLD and ts_code in self.positions:
                    sell_signals.append((ts_code, prob, price, date))

            # 排序买入信号（按概率）
            buy_signals.sort(key=lambda x: x[1], reverse=True)

            # 执行卖出
            logger.info(f"  卖出信号: {len(sell_signals)} 个")
            for ts_code, prob, price, date in sell_signals:
                self.execute_trade(ts_code, 'sell', price, date)

            # 执行买入
            logger.info(f"  买入信号: {len(buy_signals)} 个")
            for ts_code, prob, price, date in buy_signals:
                if len(self.positions) < self.MAX_POSITIONS:
                    logger.info(f"    {ts_code}: 概率={prob:.4f}, 价格={price:.2f}")
                    self.execute_trade(ts_code, 'buy', price, date)

            # 计算当前净值
            total_value = self.capital
            latest_prices = {}

            for ts_code, pos in self.positions.items():
                try:
                    df = self.storage.get_daily_prices(ts_code, trade_date, trade_date)
                    if df is not None and not df.empty:
                        current_price = df['close'].iloc[-1]
                        market_value = pos['quantity'] * current_price
                        total_value += market_value
                        latest_prices[ts_code] = current_price
                except:
                    pass

            self.daily_values.append({
                'date': trade_date,
                'total_value': total_value,
                'capital': self.capital,
                'positions': len(self.positions)
            })

            total_return = (total_value - self.initial_capital) / self.initial_capital

            # 显示持仓汇总
            logger.info(f"  持仓: {len(self.positions)} 只")
            for ts_code, pos in self.positions.items():
                current_price = latest_prices.get(ts_code, pos['entry_price'])
                market_value = pos['quantity'] * current_price
                pnl = market_value - pos['quantity'] * pos['entry_price']
                pnl_ratio = pnl / (pos['quantity'] * pos['entry_price'])
                logger.info(f"    {ts_code}: {pos['quantity']}股, 成本={pos['entry_price']:.2f}, 盈亏={pnl:.2f} ({pnl_ratio*100:.2f}%)")

            logger.info(f"  总资产: {total_value:,.2f} 元, 收益率: {total_return*100:.2f}%")

        self.print_summary()

    def print_summary(self):
        """打印回测总结"""
        logger.info("\n" + "=" * 60)
        logger.info("回测总结")
        logger.info("=" * 60)

        if not self.daily_values:
            logger.info("无交易记录")
            return

        final_value = self.daily_values[-1]['total_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital

        logger.info(f"初始资金: {self.initial_capital:,.2f} 元")
        logger.info(f"最终资产: {final_value:,.2f} 元")
        logger.info(f"总收益: {final_value - self.initial_capital:,.2f} 元")
        logger.info(f"总收益率: {total_return*100:.2f}%")
        logger.info(f"交易次数: {len(self.trade_history)}")

        # 保存结果
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'trades': self.trade_history,
            'daily_values': self.daily_values
        }

        with open('backtest_results/historical_2025_01.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("\n结果已保存到: backtest_results/historical_2025_01.json")


def main():
    """主函数"""
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")

    backtester = HistoricalBacktester(initial_capital=1000000)
    backtester.run_backtest(start_date='20250101', end_date='20250129')


if __name__ == "__main__":
    main()
