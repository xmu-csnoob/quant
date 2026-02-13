"""
Strategy service - 策略服务层
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional
import random

from src.api.schemas.strategy import (
    Strategy, Signal, StrategyStatus, SignalDirection,
    BacktestConfig, BacktestResult, BacktestStatus,
    EquityPoint, Trade
)


class StrategyService:
    """策略服务"""

    def __init__(self):
        # 系统内置的6个策略
        self._strategies = [
            Strategy(
                id="ma_macd_rsi",
                name="MA+MACD+RSI趋势策略",
                description="基于均线、MACD和RSI的趋势跟踪策略",
                status=StrategyStatus.RUNNING,
                return_rate=15.8,
                win_rate=62.5,
                trade_count=48
            ),
            Strategy(
                id="mean_reversion",
                name="均值回归策略",
                description="基于布林带和RSI的均值回归策略",
                status=StrategyStatus.RUNNING,
                return_rate=12.3,
                win_rate=58.2,
                trade_count=36
            ),
            Strategy(
                id="ml_strategy",
                name="机器学习策略",
                description="基于XGBoost的预测策略",
                status=StrategyStatus.STOPPED,
                return_rate=18.5,
                win_rate=65.8,
                trade_count=42
            ),
            Strategy(
                id="ensemble",
                name="集成策略",
                description="多策略投票/加权组合策略",
                status=StrategyStatus.RUNNING,
                return_rate=14.2,
                win_rate=61.0,
                trade_count=55
            ),
            Strategy(
                id="adaptive_dynamic",
                name="自适应动态策略",
                description="基于市场状态识别的自适应策略",
                status=StrategyStatus.RUNNING,
                return_rate=16.8,
                win_rate=59.5,
                trade_count=38
            ),
            Strategy(
                id="momentum",
                name="动量策略",
                description="基于价格动量的趋势策略",
                status=StrategyStatus.STOPPED,
                return_rate=11.2,
                win_rate=55.0,
                trade_count=32
            ),
        ]
        self._signals: List[Signal] = []
        self._init_mock_signals()

    def _init_mock_signals(self):
        """初始化模拟信号"""
        codes = [
            ("600000.SH", "浦发银行"),
            ("600519.SH", "贵州茅台"),
            ("000858.SZ", "五粮液"),
            ("601318.SH", "中国平安"),
            ("000333.SZ", "美的集团"),
        ]

        for i, strategy in enumerate(self._strategies[:3]):
            for code, name in codes[:2]:
                self._signals.append(Signal(
                    id=str(uuid.uuid4()),
                    strategy_id=strategy.id,
                    strategy_name=strategy.name,
                    code=code,
                    name=name,
                    direction=random.choice([SignalDirection.BUY, SignalDirection.SELL]),
                    price=round(random.uniform(10, 100), 2),
                    confidence=round(random.uniform(60, 90), 1),
                    created_at=datetime.now() - timedelta(minutes=random.randint(1, 60))
                ))

    def get_strategies(self) -> List[Strategy]:
        """获取策略列表"""
        return self._strategies

    def get_signals(self) -> List[Signal]:
        """获取实时信号"""
        return self._signals

    def toggle_strategy(self, strategy_id: str, action: str) -> bool:
        """启动/停止策略"""
        for strategy in self._strategies:
            if strategy.id == strategy_id:
                if action == "start":
                    strategy.status = StrategyStatus.RUNNING
                elif action == "stop":
                    strategy.status = StrategyStatus.STOPPED
                return True
        return False


class BacktestService:
    """回测服务"""

    def __init__(self):
        self._results: dict[str, BacktestResult] = {}

    def run_backtest(self, config: BacktestConfig) -> str:
        """运行回测"""
        backtest_id = str(uuid.uuid4())

        # 生成模拟净值曲线
        equity_curve = self._generate_equity_curve(config)
        trades = self._generate_trades(config)

        # 计算统计指标
        total_return = (equity_curve[-1].equity - config.initial_capital) / config.initial_capital * 100

        result = BacktestResult(
            id=backtest_id,
            status=BacktestStatus.COMPLETED,
            config=config,
            total_return=round(total_return, 2),
            annual_return=round(total_return * 250 / 365, 2),
            max_drawdown=round(random.uniform(5, 15), 2),
            sharpe_ratio=round(random.uniform(1.0, 2.5), 2),
            win_rate=round(random.uniform(55, 70), 1),
            profit_factor=round(random.uniform(1.2, 2.0), 2),
            trade_count=len(trades),
            equity_curve=equity_curve,
            trades=trades,
            # T+1统计（模拟数据）
            t1_violations=random.randint(0, 3),
            t1_skipped_sells=random.randint(0, 2)
        )

        self._results[backtest_id] = result
        return backtest_id

    def get_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """获取回测结果"""
        return self._results.get(backtest_id)

    def _generate_equity_curve(self, config: BacktestConfig) -> List[EquityPoint]:
        """生成模拟净值曲线"""
        curve = []
        current_date = config.start_date
        equity = config.initial_capital

        while current_date <= config.end_date:
            # 模拟每日收益
            daily_return = random.gauss(0.0005, 0.015)
            equity = equity * (1 + daily_return)

            curve.append(EquityPoint(
                date=current_date.isoformat(),
                equity=round(equity, 2),
                return_rate=round((equity - config.initial_capital) / config.initial_capital * 100, 2)
            ))

            current_date += timedelta(days=1)

        return curve

    def _generate_trades(self, config: BacktestConfig) -> List[Trade]:
        """生成模拟交易记录"""
        trades = []
        codes = [
            ("600000.SH", "浦发银行"),
            ("600519.SH", "贵州茅台"),
            ("000858.SZ", "五粮液"),
        ]

        current_date = config.start_date
        for _ in range(random.randint(10, 30)):
            current_date += timedelta(days=random.randint(1, 10))
            if current_date > config.end_date:
                break

            code, name = random.choice(codes)
            direction = random.choice([SignalDirection.BUY, SignalDirection.SELL])

            trades.append(Trade(
                date=current_date.isoformat(),
                code=code,
                name=name,
                direction=direction,
                price=round(random.uniform(10, 100), 2),
                shares=random.choice([100, 200, 500, 1000]),
                profit=round(random.uniform(-500, 1000), 2)
            ))

        return trades


# 单例
strategy_service = StrategyService()
backtest_service = BacktestService()
