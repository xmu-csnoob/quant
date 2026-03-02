#!/usr/bin/env python3
"""
严格回测执行引擎

复用 backtest_lstm_oos.py 的执行组件：
- TransactionCostCalculator: 交易成本
- TimeAwareSlippage: 时间感知滑点
- PriceLimitChecker: 涨跌停检查
- SuspensionManager: 停牌检查

输入：预测结果 DataFrame（ts_code, trade_date, pred_prob, fwd_ret）
输出：严格回测结果
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from src.backtesting.costs import CostConfig, TransactionCostCalculator
from src.backtesting.slippage import TimeAwareSlippage, TradingSession
from src.trading.price_limit import get_price_limit_checker
from src.trading.suspension import get_suspension_manager
from src.data.storage.sqlite_storage import SQLiteStorage


@dataclass
class StrictTradeResult:
    """单笔交易结果"""
    ts_code: str
    signal_date: str
    buy_date: Optional[str]
    sell_date: Optional[str]
    pred_prob: float
    cash_slot: float
    shares: int
    buy_price: float
    sell_price: float
    buy_cost: float
    sell_cost: float
    slippage_buy_rate: float
    slippage_sell_rate: float
    net_proceeds: float
    trade_return: float
    holding_days: int
    status: str  # executed, skipped, forced_sell
    reason: str


class StrictBacktestEngine:
    """严格回测执行引擎"""

    def __init__(
        self,
        holding_period: int = 20,
        max_positions: int = 30,
        initial_capital: float = 1_000_000.0,
        max_volume_participation: float = 0.10,
    ):
        self.holding_period = holding_period
        self.max_positions = max_positions
        self.initial_capital = initial_capital
        self.max_volume_participation = max_volume_participation

        self.storage = SQLiteStorage()

        # 执行组件
        self.cost_calculator = TransactionCostCalculator(CostConfig.default())
        self.slippage_model = TimeAwareSlippage(base_rate=Decimal("0.001"))
        self.price_limit_checker = get_price_limit_checker()
        self.suspension_manager = get_suspension_manager()

        # 加载股票元数据
        self._load_stock_meta()

    def _load_stock_meta(self):
        """加载股票元数据（名称、上市日期）"""
        self.stock_meta: Dict[str, Dict] = {}
        try:
            df = pd.read_sql_query(
                "SELECT ts_code, name, list_date FROM stock_list",
                self.storage.conn
            )
            for _, row in df.iterrows():
                self.stock_meta[row['ts_code']] = {
                    'name': row.get('name'),
                    'list_date': row.get('list_date'),
                }
        except Exception as e:
            logger.warning(f"加载股票元数据失败: {e}")

    def _get_stock_prices(self, ts_code: str) -> pd.DataFrame:
        """获取股票日线数据"""
        return self.storage.get_daily_prices(ts_code)

    def run_backtest(
        self,
        predictions_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        运行严格回测

        Args:
            predictions_df: 包含 ts_code, trade_date, pred_prob, fwd_ret 的 DataFrame

        Returns:
            (trade_results_df, portfolio_metrics)
        """
        # 按调仓周期分组
        all_dates = sorted(predictions_df['trade_date'].unique())
        rebalance_dates = all_dates[::self.holding_period]

        cash_per_slot = self.initial_capital / self.max_positions
        trade_results = []
        portfolio_values = [self.initial_capital]
        current_positions: Dict[str, StrictTradeResult] = {}

        for signal_date in rebalance_dates:
            # 获取当日预测
            day_preds = predictions_df[
                predictions_df['trade_date'] == signal_date
            ].sort_values('pred_prob', ascending=False)

            if len(day_preds) == 0:
                continue

            # 选择 Top N
            top_stocks = day_preds.head(self.max_positions)

            # 执行卖出（先卖后买）
            sell_proceeds = 0.0
            positions_to_sell = [
                code for code in current_positions
                if code not in top_stocks['ts_code'].values
            ]

            for ts_code in positions_to_sell:
                result = current_positions[ts_code]
                if result.status == 'executed' and result.sell_date:
                    sell_proceeds += result.net_proceeds
                del current_positions[ts_code]

            # 执行买入
            buy_cost = 0.0
            for _, row in top_stocks.iterrows():
                ts_code = row['ts_code']
                if ts_code in current_positions:
                    continue  # 已持有

                result = self._execute_trade(
                    ts_code=ts_code,
                    signal_date=signal_date,
                    pred_prob=row['pred_prob'],
                    cash_slot=cash_per_slot,
                )
                trade_results.append(result)

                if result.status == 'executed':
                    current_positions[ts_code] = result
                    buy_cost += result.buy_cost * result.shares

            # 计算当日组合价值
            position_value = sum(
                r.net_proceeds for r in current_positions.values()
                if r.status == 'executed'
            )
            total_value = position_value + sell_proceeds
            portfolio_values.append(total_value)

        # 汇总结果
        results_df = pd.DataFrame([{
            'ts_code': r.ts_code,
            'signal_date': r.signal_date,
            'buy_date': r.buy_date,
            'sell_date': r.sell_date,
            'pred_prob': r.pred_prob,
            'shares': r.shares,
            'buy_price': r.buy_price,
            'sell_price': r.sell_price,
            'trade_return': r.trade_return,
            'status': r.status,
            'reason': r.reason,
        } for r in trade_results])

        # 计算组合指标
        portfolio_metrics = self._calculate_metrics(portfolio_values)

        return results_df, portfolio_metrics

    def _execute_trade(
        self,
        ts_code: str,
        signal_date: str,
        pred_prob: float,
        cash_slot: float,
    ) -> StrictTradeResult:
        """执行单笔交易（严格模式）"""
        df_stock = self._get_stock_prices(ts_code)
        if df_stock.empty:
            return StrictTradeResult(
                ts_code=ts_code, signal_date=signal_date, buy_date=None,
                sell_date=None, pred_prob=pred_prob, cash_slot=cash_slot,
                shares=0, buy_price=0, sell_price=0, buy_cost=0, sell_cost=0,
                slippage_buy_rate=0, slippage_sell_rate=0, net_proceeds=cash_slot,
                trade_return=0, holding_days=0, status='skipped', reason='no_price_data'
            )

        # 找到信号日索引
        df_stock = df_stock.sort_values('trade_date').reset_index(drop=True)
        signal_idx = df_stock.index[df_stock['trade_date'] == signal_date].to_list()

        if not signal_idx:
            return StrictTradeResult(
                ts_code=ts_code, signal_date=signal_date, buy_date=None,
                sell_date=None, pred_prob=pred_prob, cash_slot=cash_slot,
                shares=0, buy_price=0, sell_price=0, buy_cost=0, sell_cost=0,
                slippage_buy_rate=0, slippage_sell_rate=0, net_proceeds=cash_slot,
                trade_return=0, holding_days=0, status='skipped', reason='signal_date_not_found'
            )

        signal_idx = signal_idx[0]

        # T+1 买入
        buy_idx = signal_idx + 1
        if buy_idx >= len(df_stock):
            return StrictTradeResult(
                ts_code=ts_code, signal_date=signal_date, buy_date=None,
                sell_date=None, pred_prob=pred_prob, cash_slot=cash_slot,
                shares=0, buy_price=0, sell_price=0, buy_cost=0, sell_cost=0,
                slippage_buy_rate=0, slippage_sell_rate=0, net_proceeds=cash_slot,
                trade_return=0, holding_days=0, status='skipped', reason='no_t1_data'
            )

        buy_row = df_stock.iloc[buy_idx]
        buy_date = buy_row['trade_date']

        # 停牌检查（volume 列来自 SQLiteStorage.get_daily_prices）
        if buy_row.get('volume', 0) == 0:
            return StrictTradeResult(
                ts_code=ts_code, signal_date=signal_date, buy_date=buy_date,
                sell_date=None, pred_prob=pred_prob, cash_slot=cash_slot,
                shares=0, buy_price=0, sell_price=0, buy_cost=0, sell_cost=0,
                slippage_buy_rate=0, slippage_sell_rate=0, net_proceeds=cash_slot,
                trade_return=0, holding_days=0, status='skipped', reason='suspended'
            )

        # 涨停检查
        prev_close = df_stock.iloc[signal_idx]['close']
        stock_name = self.stock_meta.get(ts_code, {}).get('name')
        list_date = self.stock_meta.get(ts_code, {}).get('list_date')

        raw_buy_price = buy_row.get('open', buy_row['close'])
        can_buy, buy_reason = self.price_limit_checker.can_buy(
            code=ts_code,
            price=raw_buy_price,
            prev_close=prev_close,
            name=stock_name,
            check_date=buy_date,
            list_date=list_date,
        )
        if not can_buy:
            return StrictTradeResult(
                ts_code=ts_code, signal_date=signal_date, buy_date=buy_date,
                sell_date=None, pred_prob=pred_prob, cash_slot=cash_slot,
                shares=0, buy_price=0, sell_price=0, buy_cost=0, sell_cost=0,
                slippage_buy_rate=0, slippage_sell_rate=0, net_proceeds=cash_slot,
                trade_return=0, holding_days=0, status='skipped', reason=f'limit_up:{buy_reason}'
            )

        # 滑点调整
        buy_slippage = self.slippage_model.apply_slippage(
            price=Decimal(str(raw_buy_price)),
            side='buy',
            volume=Decimal('1000'),
            avg_volume=Decimal(str(buy_row.get('volume', 1000000))),
            session=TradingSession.OPENING,
        )
        buy_price = float(buy_slippage.adjusted_price)

        # 计算可买股数
        buy_rate = float(self.cost_calculator.get_effective_buy_rate_with_code(ts_code))
        max_shares_by_vol = int(buy_row.get('volume', 0) * self.max_volume_participation / 100) * 100
        shares = int(cash_slot / (buy_price * (1 + buy_rate)) / 100) * 100
        shares = min(shares, max_shares_by_vol) if max_shares_by_vol > 0 else shares

        if shares < 100:
            return StrictTradeResult(
                ts_code=ts_code, signal_date=signal_date, buy_date=buy_date,
                sell_date=None, pred_prob=pred_prob, cash_slot=cash_slot,
                shares=0, buy_price=0, sell_price=0, buy_cost=0, sell_cost=0,
                slippage_buy_rate=0, slippage_sell_rate=0, net_proceeds=cash_slot,
                trade_return=0, holding_days=0, status='skipped', reason='insufficient_cash'
            )

        # 持有 N 天后卖出
        sell_idx = buy_idx + self.holding_period
        if sell_idx >= len(df_stock):
            # 强制在最后一天平仓
            sell_idx = len(df_stock) - 1

        sell_row = df_stock.iloc[sell_idx]
        sell_date = sell_row['trade_date']

        # 跌停检查
        raw_sell_price = sell_row.get('open', sell_row['close'])
        can_sell, sell_reason = self.price_limit_checker.can_sell(
            code=ts_code,
            price=raw_sell_price,
            prev_close=df_stock.iloc[sell_idx - 1]['close'],
            name=stock_name,
            check_date=sell_date,
            list_date=list_date,
        )

        # 如果跌停，尝试往后找可卖日
        while not can_sell and sell_idx < len(df_stock) - 1:
            sell_idx += 1
            sell_row = df_stock.iloc[sell_idx]
            sell_date = sell_row['trade_date']
            raw_sell_price = sell_row.get('open', sell_row['close'])
            can_sell, sell_reason = self.price_limit_checker.can_sell(
                code=ts_code,
                price=raw_sell_price,
                prev_close=df_stock.iloc[sell_idx - 1]['close'],
                name=stock_name,
                check_date=sell_date,
                list_date=list_date,
            )

        # 卖出滑点
        sell_slippage = self.slippage_model.apply_slippage(
            price=Decimal(str(raw_sell_price)),
            side='sell',
            volume=Decimal(str(shares)),
            avg_volume=Decimal(str(sell_row.get('volume', 1000000))),
            session=TradingSession.OPENING,
        )
        sell_price = float(sell_slippage.adjusted_price)

        # 计算成本
        sell_rate = float(self.cost_calculator.get_effective_sell_rate_with_code(ts_code))
        buy_cost = buy_price * shares * (1 + buy_rate)
        sell_revenue = sell_price * shares * (1 - sell_rate)
        net_proceeds = sell_revenue
        trade_return = (net_proceeds - cash_slot) / cash_slot if cash_slot > 0 else 0

        return StrictTradeResult(
            ts_code=ts_code,
            signal_date=signal_date,
            buy_date=buy_date,
            sell_date=sell_date,
            pred_prob=pred_prob,
            cash_slot=cash_slot,
            shares=shares,
            buy_price=buy_price,
            sell_price=sell_price,
            buy_cost=buy_cost,
            sell_cost=sell_price * shares * sell_rate,
            slippage_buy_rate=float(buy_slippage.slippage_rate),
            slippage_sell_rate=float(sell_slippage.slippage_rate),
            net_proceeds=net_proceeds,
            trade_return=trade_return,
            holding_days=sell_idx - buy_idx,
            status='executed',
            reason='success',
        )

    def _calculate_metrics(self, portfolio_values: List[float]) -> Dict:
        """计算组合指标"""
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] - values[0]) / values[0]
        n_periods = len(returns)

        if n_periods > 0:
            annual_return = (1 + total_return) ** (252 / (n_periods * self.holding_period)) - 1
            volatility = np.std(returns) * np.sqrt(252 / self.holding_period)
            sharpe = annual_return / volatility if volatility > 0 else 0
        else:
            annual_return = 0
            volatility = 0
            sharpe = 0

        # 最大回撤
        cummax = np.maximum.accumulate(values)
        drawdowns = (cummax - values) / cummax
        max_drawdown = np.max(drawdowns)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_trades': n_periods,
        }
