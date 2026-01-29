#!/usr/bin/env python3
"""
模拟盘实盘测试

从2026年1月30日开始运行
每天收盘后自动更新策略信号并记录交易
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from loguru import logger
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.sqlite_storage import SQLiteStorage
from utils.features.enhanced_features import EnhancedFeatureExtractor
import xgboost as xgb


class LivePaperTrader:
    """实时模拟盘交易器"""

    # A股交易成本
    BUY_COST_RATE = 0.0003    # 0.03% 佣金
    SELL_COST_RATE = 0.0013  # 0.13% 佣金+印花税

    # 策略参数
    BUY_THRESHOLD = 0.52
    SELL_THRESHOLD = 0.48
    MAX_POSITIONS = 3  # 最多同时持有3只股票
    POSITION_SIZE = 0.3  # 每只股票30%仓位

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # {ts_code: {entry_date, entry_price, quantity, shares}}
        self.trades = []
        self.daily_values = []

        # 加载模型
        self.model = xgb.Booster()
        self.model.load_model('models/xgboost_robust.json')
        self.feature_extractor = EnhancedFeatureExtractor()

        # 数据存储
        self.storage = SQLiteStorage()

        # 状态文件
        self.state_file = project_root / "data" / "live_trading_state.json"
        self.trade_log_file = project_root / "logs" / "live_trades.csv"

        # 创建日志目录
        self.trade_log_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载或初始化状态
        self.load_state()

    def load_state(self):
        """加载交易状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.capital = state.get('capital', self.initial_capital)
                self.positions = state.get('positions', {})
                self.trades = state.get('trades', [])
                self.daily_values = state.get('daily_values', [])
                logger.info(f"加载状态: 资金={self.capital:.2f}, 持仓={len(self.positions)}")
        else:
            logger.info("初始化新状态")

    def save_state(self):
        """保存交易状态"""
        state = {
            'capital': self.capital,
            'positions': self.positions,
            'trades': self.trades,
            'daily_values': self.daily_values,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_latest_date(self):
        """获取数据库中最新的交易日期"""
        stocks = self.storage.get_all_stocks()[:10]
        latest_dates = []

        for ts_code in stocks:
            df = self.storage.get_daily_prices(ts_code, '20260101', '20261231')
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                latest_dates.append(df['trade_date'].max())

        if latest_dates:
            return max(latest_dates)
        return None

    def get_universe(self, max_stocks=100):
        """获取股票池"""
        stocks = self.storage.get_all_stocks()
        # 使用前100只股票作为股票池
        return sorted(stocks)[:max_stocks]

    def check_signals(self, ts_code):
        """检查交易信号"""
        try:
            # 获取数据（需要前面90天用于特征计算）
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')

            df = self.storage.get_daily_prices(ts_code, start_date, end_date)

            if df is None or len(df) < 60:
                return None, None, None

            # 提取特征
            features = self.feature_extractor.extract(df)

            # 获取最新一天的信号（最后一行）
            if len(features) < 1:
                return None, None, None

            latest = features.iloc[-1]
            feature_cols = [c for c in features.columns if c.startswith('f_')]

            if len(feature_cols) == 0:
                return None, None, None

            X = latest[feature_cols].values.reshape(1, -1)
            prob = self.model.predict(xgb.DMatrix(X))[0]

            current_price = latest['close']
            current_date = pd.to_datetime(latest['trade_date']).date()

            return prob, current_price, current_date

        except Exception as e:
            logger.warning(f"检查 {ts_code} 信号失败: {e}")
            return None, None, None

    def execute_trade(self, ts_code, action, price, date, quantity=None):
        """执行交易"""
        if action == 'buy':
            if ts_code in self.positions:
                return  # 已持有

            if len(self.positions) >= self.MAX_POSITIONS:
                logger.info(f"达到最大持仓数 {self.MAX_POSITIONS}，跳过买入 {ts_code}")
                return

            # 计算买入数量
            buy_value = self.capital * self.POSITION_SIZE
            quantity = int(buy_value / price / 100) * 100  # 整手

            if quantity < 100:
                logger.info(f"资金不足，无法买入 {ts_code}")
                return

            cost = quantity * price * (1 + self.BUY_COST_RATE)

            if cost > self.capital:
                logger.info(f"资金不足，无法买入 {ts_code}")
                return

            # 执行买入
            self.capital -= cost
            self.positions[ts_code] = {
                'entry_date': str(date),
                'entry_price': price,
                'quantity': quantity,
                'shares': quantity
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
            self.trades.append(trade)

            logger.info(f"买入 {ts_code}: {quantity}股 @ {price:.2f}, 成本={cost:.2f}")

        elif action == 'sell':
            if ts_code not in self.positions:
                return  # 未持有

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
            self.trades.append(trade)

            logger.info(f"卖出 {ts_code}: {quantity}股 @ {price:.2f}, 盈亏={pnl:.2f} ({pnl_ratio*100:.2f}%)")

    def run_daily(self):
        """运行每日策略"""
        logger.info("=" * 60)
        logger.info(f"模拟盘交易: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # 检查最新数据日期
        latest_date = self.get_latest_date()
        if latest_date:
            logger.info(f"数据最新日期: {latest_date}")
        else:
            logger.warning("无2026年数据")
            return

        # 获取股票池
        universe = self.get_universe()
        logger.info(f"股票池: {len(universe)} 只")

        # 检查所有股票的信号
        buy_signals = []
        sell_signals = []

        for ts_code in universe:
            prob, price, date = self.check_signals(ts_code)

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
        logger.info(f"\n卖出信号: {len(sell_signals)} 个")
        for ts_code, prob, price, date in sell_signals:
            self.execute_trade(ts_code, 'sell', price, date)

        # 执行买入
        logger.info(f"\n买入信号: {len(buy_signals)} 个")
        for ts_code, prob, price, date in buy_signals:
            if len(self.positions) < self.MAX_POSITIONS:
                logger.info(f"  {ts_code}: 概率={prob:.4f}, 价格={price:.2f}")
                self.execute_trade(ts_code, 'buy', price, date)

        # 计算当前净值
        total_value = self.capital
        latest_prices = {}

        for ts_code, pos in self.positions.items():
            try:
                df = self.storage.get_daily_prices(ts_code, latest_date.strftime('%Y%m%d'), latest_date.strftime('%Y%m%d'))
                if df is not None and not df.empty:
                    current_price = df['close'].iloc[-1]
                    market_value = pos['quantity'] * current_price
                    total_value += market_value
                    latest_prices[ts_code] = current_price
            except:
                pass

        self.daily_values.append({
            'date': str(latest_date),
            'total_value': total_value,
            'capital': self.capital,
            'positions': len(self.positions)
        })

        # 保存状态
        self.save_state()

        # 输出汇总
        total_return = (total_value - self.initial_capital) / self.initial_capital

        logger.info("\n" + "=" * 60)
        logger.info("持仓汇总:")
        for ts_code, pos in self.positions.items():
            current_price = latest_prices.get(ts_code, pos['entry_price'])
            market_value = pos['quantity'] * current_price
            pnl = market_value - pos['quantity'] * pos['entry_price']
            pnl_ratio = pnl / (pos['quantity'] * pos['entry_price'])
            logger.info(f"  {ts_code}: {pos['quantity']}股, 成本={pos['entry_price']:.2f}, "
                       f"现价={current_price:.2f}, 盈亏={pnl:.2f} ({pnl_ratio*100:.2f}%)")

        logger.info("\n账户汇总:")
        logger.info(f"  初始资金: {self.initial_capital:.2f}")
        logger.info(f"  当前现金: {self.capital:.2f}")
        logger.info(f"  持仓市值: {total_value - self.capital:.2f}")
        logger.info(f"  总资产: {total_value:.2f}")
        logger.info(f"  总收益率: {total_return*100:.2f}%")
        logger.info(f"  交易次数: {len(self.trades)}")
        logger.info("=" * 60)

    def run_loop(self, interval_hours=24):
        """持续运行"""
        logger.info("模拟盘启动...")
        logger.info(f"检查间隔: {interval_hours} 小时")
        logger.info(f"买入阈值: {self.BUY_THRESHOLD}, 卖出阈值: {self.SELL_THRESHOLD}")
        logger.info(f"最大持仓: {self.MAX_POSITIONS}, 单只仓位: {self.POSITION_SIZE*100:.0f}%")

        while True:
            try:
                self.run_daily()
            except Exception as e:
                logger.error(f"运行出错: {e}")

            # 等待下次运行
            logger.info(f"\n等待 {interval_hours} 小时后下次运行...")
            time.sleep(interval_hours * 3600)


def main():
    """主函数"""
    from loguru import logger

    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/live_trading_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG"
    )

    trader = LivePaperTrader(initial_capital=100000)

    # 先运行一次
    trader.run_daily()

    # 进入循环
    trader.run_loop(interval_hours=24)


if __name__ == "__main__":
    main()
