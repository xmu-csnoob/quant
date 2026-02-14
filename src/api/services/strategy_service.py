"""
Strategy service - 策略服务层
"""

import uuid
from datetime import datetime, timedelta, date
from typing import List, Optional
from decimal import Decimal
import random
from loguru import logger

from src.api.schemas.strategy import (
    Strategy, Signal, StrategyStatus, SignalDirection,
    BacktestConfig, BacktestResult, BacktestStatus,
    EquityPoint, Trade
)

# 导入真实的回测引擎和策略
from src.backtesting.simple_backtester import SimpleBacktester
from src.backtesting.costs import CostConfig
from src.data.storage.sqlite_storage import SQLiteStorage
from src.strategies.ml_strategy import MLStrategy
from src.utils.features.enhanced_features import EnhancedFeatureExtractor


class StrategyService:
    """策略服务"""

    def __init__(self):
        # 系统内置的策略
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
                status=StrategyStatus.STOPPED,
                return_rate=14.2,
                win_rate=61.0,
                trade_count=55
            ),
            Strategy(
                id="adaptive_dynamic",
                name="自适应动态策略",
                description="基于市场状态识别的自适应策略",
                status=StrategyStatus.STOPPED,
                return_rate=16.8,
                win_rate=59.5,
                trade_count=38
            ),
        ]
        self._signals: List[Signal] = []

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

    def get_strategy_instance(self, strategy_id: str):
        """获取策略实例"""
        from src.strategies import MaMacdRsiStrategy
        from src.strategies.mean_reversion import MeanReversionStrategy
        from src.strategies.ml_strategy import MLStrategy
        from src.utils.features.enhanced_features import EnhancedFeatureExtractor

        # ML策略需要加载模型，单独处理
        if strategy_id == "ml_strategy":
            return self._create_ml_strategy()

        strategy_map = {
            "ma_macd_rsi": MaMacdRsiStrategy,
            "mean_reversion": MeanReversionStrategy,
        }

        strategy_class = strategy_map.get(strategy_id)
        if strategy_class:
            return strategy_class()
        return None

    def _create_ml_strategy(self):
        """创建ML策略实例"""
        import json
        import xgboost as xgb
        from pathlib import Path

        model_path = Path("models/xgboost_model.json")
        feature_path = Path("models/feature_cols.json")

        if not model_path.exists():
            logger.warning("ML模型文件不存在，无法创建ML策略")
            return None

        try:
            # 加载模型
            model = xgb.Booster()
            model.load_model(str(model_path))

            # 加载特征列
            feature_cols = None
            if feature_path.exists():
                with open(feature_path, 'r') as f:
                    feature_cols = json.load(f)
                logger.info(f"加载特征列: {len(feature_cols)} 个")

            # 创建特征提取器
            feature_extractor = EnhancedFeatureExtractor(prediction_period=5)

            # 创建ML策略
            strategy = MLStrategy(
                model=model,
                feature_extractor=feature_extractor,
                threshold=0.02,  # 预测收益超过2%时买入
                prediction_period=5,
                feature_cols=feature_cols,
            )

            logger.info("ML策略创建成功")
            return strategy

        except Exception as e:
            logger.error(f"创建ML策略失败: {e}")
            return None


class BacktestService:
    """回测服务 - 使用真实数据"""

    @staticmethod
    def _safe_float(value, default=0.0) -> float:
        """安全转换浮点数，处理inf和nan"""
        import math
        if value is None:
            return default
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        except:
            return default

    def __init__(self):
        self._results: dict[str, BacktestResult] = {}
        self._storage = SQLiteStorage()

    def run_backtest(self, config: BacktestConfig) -> str:
        """
        运行回测 - 使用真实数据

        Args:
            config: 回测配置

        Returns:
            backtest_id: 回测ID
        """
        backtest_id = str(uuid.uuid4())

        try:
            # 1. 获取策略实例
            strategy_service = StrategyService()
            strategy = strategy_service.get_strategy_instance(config.strategy_id)

            if strategy is None:
                logger.error(f"策略不存在: {config.strategy_id}")
                return self._create_failed_result(backtest_id, config, "策略不存在")

            # 2. 从数据库获取K线数据
            # 使用一个默认的测试股票（可以后续扩展为多股票回测）
            test_codes = ["600000.SH", "600519.SH", "000858.SH", "601318.SH"]
            df = None
            used_code = None

            for code in test_codes:
                try:
                    df = self._storage.get_daily_prices(
                        code,
                        config.start_date.strftime("%Y%m%d"),
                        config.end_date.strftime("%Y%m%d")
                    )
                    if df is not None and len(df) >= 60:  # 至少需要60条数据计算指标
                        used_code = code
                        logger.info(f"使用股票 {code} 进行回测，数据量: {len(df)}")
                        break
                except Exception as e:
                    logger.warning(f"获取 {code} 数据失败: {e}")
                    continue

            if df is None or len(df) < 60:
                error_msg = f"数据库中没有足够的数据（需要至少60条），请先导入数据"
                logger.error(error_msg)
                return self._create_failed_result(backtest_id, config, error_msg)

            # 3. 创建回测引擎
            cost_config = CostConfig(
                commission_rate=Decimal(str(config.commission_rate)),
                stamp_duty_rate=Decimal("0.001"),  # 印花税0.1%
                transfer_fee_rate=Decimal("0.00001"),  # 过户费
            )

            backtester = SimpleBacktester(
                initial_capital=config.initial_capital,
                cost_config=cost_config,
                enable_t1_rule=True,
            )

            # 4. 执行回测
            logger.info(f"开始回测: 策略={config.strategy_id}, 股票={used_code}, 日期={config.start_date}~{config.end_date}")
            result = backtester.run(strategy, df)

            # 5. 转换结果格式
            api_result = self._convert_result(backtest_id, config, result, used_code)
            self._results[backtest_id] = api_result

            logger.info(f"回测完成: 收益率={result.total_return*100:.2f}%, 交易次数={result.trade_count}")
            return backtest_id

        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_failed_result(backtest_id, config, str(e))

    def _create_failed_result(self, backtest_id: str, config: BacktestConfig, error_msg: str) -> str:
        """创建失败的回测结果"""
        result = BacktestResult(
            id=backtest_id,
            status=BacktestStatus.FAILED,
            config=config,
            total_return=0,
            annual_return=0,
            max_drawdown=0,
            sharpe_ratio=0,
            win_rate=0,
            profit_factor=0,
            trade_count=0,
            t1_violations=0,
            t1_skipped_sells=0,
            equity_curve=[],
            trades=[],
        )
        # 在trades中添加错误信息（临时方案）
        result.trades = [Trade(
            date="error",
            code="ERROR",
            name=error_msg[:50],  # 截断过长的错误信息
            direction=SignalDirection.BUY,
            price=0,
            shares=0,
            profit=0,
        )]
        self._results[backtest_id] = result
        return backtest_id

    def _convert_result(
        self,
        backtest_id: str,
        config: BacktestConfig,
        backtester_result,
        stock_code: str
    ) -> BacktestResult:
        """将回测引擎结果转换为API格式"""

        # 转换净值曲线
        equity_curve = []
        if hasattr(backtester_result, 'equity_curve') and backtester_result.equity_curve:
            for point in backtester_result.equity_curve:
                equity_curve.append(EquityPoint(
                    date=str(point.date) if hasattr(point, 'date') else str(point.get('date', '')),
                    equity=float(point.equity) if hasattr(point, 'equity') else float(point.get('equity', 0)),
                    return_rate=float(point.return_rate) if hasattr(point, 'return_rate') else float(point.get('return_rate', 0)) * 100,
                ))
        else:
            # 如果没有净值曲线，手动生成
            equity_curve = [
                EquityPoint(
                    date=str(config.start_date),
                    equity=config.initial_capital,
                    return_rate=0,
                ),
                EquityPoint(
                    date=str(config.end_date),
                    equity=backtester_result.final_capital,
                    return_rate=backtester_result.total_return * 100,
                ),
            ]

        # 转换交易记录
        trades = []
        stock_names = {
            "600000.SH": "浦发银行",
            "600519.SH": "贵州茅台",
            "000858.SH": "五粮液",
            "601318.SH": "中国平安",
        }

        for trade in backtester_result.trades:
            trades.append(Trade(
                date=str(trade.entry_date),
                code=stock_code,
                name=stock_names.get(stock_code, "未知"),
                direction=SignalDirection.BUY if trade.entry_price else SignalDirection.SELL,
                price=float(trade.entry_price),
                shares=trade.quantity,
                profit=float(trade.pnl),
            ))

        # 处理卖出交易（如果有单独的卖出记录）
        # SimpleBacktester的trades包含完整的买卖对，需要拆分
        detailed_trades = []
        for trade in backtester_result.trades:
            # 买入记录
            detailed_trades.append(Trade(
                date=str(trade.entry_date),
                code=stock_code,
                name=stock_names.get(stock_code, "未知"),
                direction=SignalDirection.BUY,
                price=float(trade.entry_price),
                shares=trade.quantity,
                profit=0,  # 买入时无盈亏
            ))
            # 卖出记录
            if trade.exit_date:
                detailed_trades.append(Trade(
                    date=str(trade.exit_date),
                    code=stock_code,
                    name=stock_names.get(stock_code, "未知"),
                    direction=SignalDirection.SELL,
                    price=float(trade.exit_price) if hasattr(trade, 'exit_price') else float(trade.entry_price),
                    shares=trade.quantity,
                    profit=float(trade.pnl),
                ))

        return BacktestResult(
            id=backtest_id,
            status=BacktestStatus.COMPLETED,
            config=config,
            total_return=round(self._safe_float(backtester_result.total_return * 100), 2),
            annual_return=round(self._safe_float(getattr(backtester_result, 'annual_return', 0) * 100), 2),
            max_drawdown=round(self._safe_float(backtester_result.max_drawdown * 100), 2),
            sharpe_ratio=round(self._safe_float(backtester_result.sharpe_ratio), 2),
            win_rate=round(self._safe_float(backtester_result.win_rate * 100), 1),
            profit_factor=round(self._safe_float(getattr(backtester_result, 'profit_factor', 0)), 2),
            trade_count=backtester_result.trade_count,
            t1_violations=getattr(backtester_result, 't1_violations', 0),
            t1_skipped_sells=getattr(backtester_result, 't1_skipped_sells', 0),
            equity_curve=equity_curve,
            trades=detailed_trades if detailed_trades else trades,
        )

    def get_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """获取回测结果"""
        return self._results.get(backtest_id)

    def get_available_stocks(self) -> List[str]:
        """获取数据库中有数据的股票列表"""
        try:
            stocks = self._storage.get_all_stocks()
            return stocks[:20] if stocks else []  # 返回前20只
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []


# 单例
strategy_service = StrategyService()
backtest_service = BacktestService()
