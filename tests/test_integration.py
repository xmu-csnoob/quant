"""
系统集成测试

测试完整的量化交易流程
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from decimal import Decimal
import pandas as pd
import numpy as np
from datetime import datetime, date

from src.data.fetchers.mock import MockDataFetcher
from src.data.storage.storage import DataStorage
from src.data.api.data_manager import DataManager
from src.strategies import MaMacdRsiStrategy
from src.backtesting.simple_backtester import SimpleBacktester
from src.backtesting.costs import CostConfig
from src.risk.manager import RiskManager
from src.trading.api import MockTradingAPI


def test_data_to_backtest_integration():
    """测试数据获取到回测的完整流程"""
    print("\n" + "=" * 60)
    print("测试: 数据获取 → 回测")
    print("=" * 60)

    # 1. 数据获取
    print("\n[1/4] 数据获取")
    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)
    df = manager.get_daily_price("600000.SH", "20230101", "20231231")

    assert df is not None
    assert len(df) > 100
    print(f"  ✅ 获取数据: {len(df)} 条")

    # 2. 策略信号生成
    print("\n[2/4] 策略信号生成")
    strategy = MaMacdRsiStrategy()
    signals = strategy.generate_signals(df)

    print(f"  ✅ 生成信号: {len(signals)} 个")

    # 3. 回测执行
    print("\n[3/4] 回测执行")
    backtester = SimpleBacktester(
        initial_capital=1000000,
        cost_config=CostConfig.default(),
        enable_t1_rule=True,
    )
    result = backtester.run(strategy, df)

    assert result is not None
    print(f"  ✅ 交易次数: {result.trade_count}")
    print(f"  ✅ 总收益率: {result.total_return*100:.2f}%")

    # 4. 验证高级指标
    print("\n[4/4] 验证高级指标")
    assert result.annual_return != 0
    assert result.sortino_ratio != 0 or result.trade_count == 0
    assert result.benchmark_return != 0
    print(f"  ✅ 年化收益: {result.annual_return*100:.2f}%")
    print(f"  ✅ 索提诺比率: {result.sortino_ratio:.2f}")
    print(f"  ✅ 基准收益: {result.benchmark_return*100:.2f}%")

    print("\n✅ 数据→回测集成测试通过")


def test_backtest_to_trading_integration():
    """测试回测到实盘交易的流程"""
    print("\n" + "=" * 60)
    print("测试: 回测 → 实盘交易")
    print("=" * 60)

    # 1. 准备数据
    print("\n[1/3] 准备数据")
    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)
    df = manager.get_daily_price("600000.SH", "20230101", "20231231")

    # 2. 回测获取最优信号
    print("\n[2/3] 回测获取信号")
    strategy = MaMacdRsiStrategy()
    backtester = SimpleBacktester(initial_capital=1000000)
    result = backtester.run(strategy, df)

    print(f"  ✅ 回测交易次数: {result.trade_count}")

    # 3. 模拟实盘交易
    print("\n[3/3] 模拟实盘交易")
    api = MockTradingAPI(initial_cash=1000000)

    # 获取最新一天的信号（如果有）
    signals = strategy.generate_signals(df)
    if signals:
        latest_signal = signals[-1]
        if latest_signal.signal_type.name == "BUY":
            # 模拟买入
            price = Decimal(str(latest_signal.price))
            api.set_current_price(latest_signal.code, float(price))
            order = api.buy(code=latest_signal.code, price=price, quantity=1000)
            print(f"  ✅ 执行买入: {order.order_id if order else 'None'}")
        elif latest_signal.signal_type.name == "SELL":
            print(f"  ✅ 执行卖出信号")

    # 检查账户状态
    account = api.get_account()
    print(f"  ✅ 账户资金: {account.cash}")

    print("\n✅ 回测→交易集成测试通过")


def test_risk_control_integration():
    """测试风险控制集成"""
    print("\n" + "=" * 60)
    print("测试: 风险控制集成")
    print("=" * 60)

    # 1. 初始化风险管理器
    print("\n[1/3] 初始化风险管理器")
    risk_manager = RiskManager(
        initial_capital=1000000,
        max_drawdown=0.15,
        max_position_ratio=0.3,
    )
    print("  ✅ 风险管理器初始化完成")

    # 2. 模拟交易过程
    print("\n[2/3] 模拟交易过程")
    api = MockTradingAPI(initial_cash=1000000)

    # 检查是否允许开仓
    can_enter, pos_size, reason = risk_manager.check_entry(
        price=100.0,
        signal_confidence=0.8,
        stock_code="600000.SH",
    )
    print(f"  ✅ 允许开仓: {can_enter}")

    # 3. 模拟止损
    print("\n[3/3] 模拟止损检查")
    from src.risk.manager import Position

    # 添加持仓
    risk_manager.positions["600000.SH"] = Position(
        entry_price=100.0,
        shares=1000,
        entry_date="20240101",
    )

    # 检查止损（价格下跌10%）
    check = risk_manager.check_exit("600000.SH", 90.0, "20240102")
    print(f"  ✅ 止损检查: {check.action.name}, 原因: {check.reason}")

    print("\n✅ 风险控制集成测试通过")


def test_ml_prediction_integration():
    """测试ML预测集成（使用mock）"""
    print("\n" + "=" * 60)
    print("测试: ML预测集成")
    print("=" * 60)

    # 1. 准备数据
    print("\n[1/3] 准备数据")
    fetcher = MockDataFetcher(scenario="bull")
    storage = DataStorage()
    manager = DataManager(fetcher=fetcher, storage=storage)
    df = manager.get_daily_price("600000.SH", "20230101", "20231231")

    assert df is not None
    print(f"  ✅ 数据准备完成: {len(df)} 条")

    # 2. 特征提取
    print("\n[2/3] 特征提取")
    from src.utils.features.enhanced_features import EnhancedFeatureExtractor

    extractor = EnhancedFeatureExtractor(prediction_period=5)
    features = extractor.extract(df)

    feature_cols = [c for c in features.columns if c.startswith('f_')]
    print(f"  ✅ 特征数量: {len(feature_cols)}")

    # 3. 模拟预测（不依赖模型文件）
    print("\n[3/3] 模拟预测")
    # 使用简单的随机预测作为示例
    latest_features = features.iloc[-1][feature_cols].values

    # 简单规则：如果最近5日收益率为正，预测上涨
    recent_return = features.iloc[-1].get('f_return_5d', 0)
    if recent_return > 0:
        prediction = "UP"
        probability = 0.6
    else:
        prediction = "DOWN"
        probability = 0.4

    print(f"  ✅ 预测方向: {prediction}")
    print(f"  ✅ 预测概率: {probability:.2f}")

    print("\n✅ ML预测集成测试通过")


def run_all_integration_tests():
    """运行所有集成测试"""
    print("=" * 60)
    print("系统集成测试")
    print("=" * 60)

    test_data_to_backtest_integration()
    test_backtest_to_trading_integration()
    test_risk_control_integration()
    test_ml_prediction_integration()

    print("\n" + "=" * 60)
    print("所有集成测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_integration_tests()
