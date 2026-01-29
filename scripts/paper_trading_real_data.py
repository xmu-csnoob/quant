"""
模拟盘交易 - 使用真实历史数据

完整流程：获取数据 -> 策略生成信号 -> 模拟盘执行 -> 显示结果
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.engine import LiveTradingEngine
from trading.api import MockTradingAPI
from trading.orders import OrderSide
from risk.manager import RiskManager, PositionSizer
from strategies.ma_macd_rsi import MaMacdRsiStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from strategies.base import SignalType
from data.storage.storage import DataStorage
from data.api.data_manager import DataManager
from data.fetchers.tushare import TushareDataFetcher


def run_paper_trading_real_data():
    """使用真实数据进行模拟盘交易"""
    print("=" * 80)
    print("模拟盘交易 - 真实历史数据")
    print("=" * 80)

    # 1. 初始化数据管理器
    print("\n[1] 初始化数据管理器...")

    storage = DataStorage()

    # 尝试使用Tushare，如果没有token则使用Mock
    import os
    token = os.getenv("TUSHARE_TOKEN")

    if token:
        print("  使用 Tushare 真实数据")
        fetcher = TushareDataFetcher()
    else:
        print("  未检测到 TUSHARE_TOKEN，使用 Mock 数据")
        from data.fetchers.mock import MockDataFetcher
        fetcher = MockDataFetcher(scenario="sideways")  # 震荡市

    manager = DataManager(fetcher=fetcher, storage=storage)

    # 2. 获取股票数据
    print("\n[2] 获取股票数据...")

    # 测试股票
    symbols = ["600000.SH"]  # 浦发银行

    # 获取数据
    start_date = "20200101"
    end_date = "20241231"

    print(f"  获取 {symbols[0]} 数据 ({start_date} - {end_date})")

    df = manager.get_daily_price(symbols[0], start_date, end_date)

    if df is None or df.empty:
        print("  ✗ 数据获取失败")
        return

    print(f"  ✓ 获取到 {len(df)} 条数据")
    print(f"  日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"  价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

    # 3. 选择策略
    print("\n[3] 选择策略...")
    print("  可用策略:")
    print("    1. 趋势跟踪 (MA+MACD+RSI)")
    print("    2. 均值回归 (布林带+RSI)")
    print("    3. ML预测 (XGBoost)")

    # 默认使用均值回归（更适合震荡市）
    strategy = MeanReversionStrategy()
    print(f"  使用: {strategy.name}")

    # 4. 生成交易信号
    print("\n[4] 生成交易信号...")

    signals = strategy.generate_signals(df)

    print(f"  ✓ 生成 {len(signals)} 个信号")

    if not signals:
        print("  无交易信号，退出")
        return

    # 显示前几个信号
    print("\n  前5个信号:")
    for i, sig in enumerate(signals[:5]):
        print(f"    {sig.date} {sig.signal_type.value} "
              f"@ {sig.price:.2f} (置信度: {sig.confidence:.2f}) - {sig.reason}")

    # 5. 创建模拟盘引擎
    print("\n[5] 创建模拟盘引擎...")

    initial_cash = 100000
    api = MockTradingAPI(initial_cash=initial_cash)

    # 创建风控
    position_sizer = PositionSizer(initial_capital=initial_cash, method="fixed_ratio")
    risk_manager = RiskManager(
        initial_capital=initial_cash,
        position_sizer=position_sizer,
        stop_loss=0.05,      # 5% 止损
        take_profit=0.15,    # 15% 止盈
        max_drawdown=0.15,   # 最大回撤15%
    )

    engine = LiveTradingEngine(
        strategy=strategy,
        trading_api=api,
        risk_manager=risk_manager,
        symbols=symbols,
    )

    print(f"  ✓ 初始资金: {initial_cash:,.0f} 元")

    # 6. 启动引擎
    print("\n[6] 启动引擎...")
    if not engine.start():
        print("  ✗ 启动失败")
        return
    print("  ✓ 引擎已启动")

    # 7. 模拟历史交易（按时间顺序执行信号）
    print("\n[7] 执行交易信号...")

    # 按日期排序
    signals_sorted = sorted(signals, key=lambda x: x.date)

    executed_count = 0
    for i, signal in enumerate(signals_sorted):
        # 检查当前持仓
        positions = api.get_positions()
        has_position = any(p.symbol == symbols[0] for p in positions)

        # 风控检查：如果有持仓且是买入信号，跳过
        if has_position and signal.signal_type == SignalType.BUY:
            # 检查是否需要止损/止盈
            for pos in positions:
                if pos.symbol == symbols[0]:
                    # 模拟获取当前价格
                    current_price = signal.price
                    risk_check = risk_manager.check_exit(pos.symbol, current_price, signal.date)

                    if not risk_check.passed:
                        print(f"  {signal.date} 触发{risk_check.action.value}: {risk_check.reason}")

                        # 创建平仓订单
                        exit_signal = signal.__class__(
                            date=signal.date,
                            signal_type=SignalType.SELL,
                            price=current_price,
                            reason=risk_check.reason,
                            confidence=1.0,
                        )

                        order = engine.process_signal(exit_signal, current_price=current_price)
                        if order:
                            api.simulate_fill(order, fill_price=current_price, fill_quantity=order.quantity)
                            executed_count += 1
            continue

        # 检查账户状态
        account = api.get_account()

        # 没有持仓且是卖出信号，跳过
        if not has_position and signal.signal_type == SignalType.SELL:
            continue

        # 处理信号
        print(f"  {signal.date} {signal.signal_type.value} "
              f"@ {signal.price:.2f} (置信度: {signal.confidence:.2f})")

        order = engine.process_signal(signal, current_price=signal.price)

        if order:
            # 模拟成交
            fill_price = signal.price
            fill_quantity = order.quantity
            api.simulate_fill(order, fill_price=fill_price, fill_quantity=fill_quantity)
            executed_count += 1
            print(f"    ✓ 成交: {order.order_id}")
        else:
            print(f"    ✗ 订单被拒绝或风控拦截")

    print(f"\n  ✓ 执行了 {executed_count} 笔交易")

    # 8. 显示最终结果
    print("\n[8] 最终结果")

    engine.print_status()

    # 显示成交记录
    trades = api.get_trades()
    if trades:
        print(f"\n  成交明细 ({len(trades)} 笔):")
        for trade in trades:
            print(f"    {trade.trade_time} {trade.symbol} {trade.side.value} "
                  f"{trade.quantity}股 @ {trade.price:.2f}")

    # 计算收益
    account = api.get_account()
    total_return = (account.total_assets - initial_cash) / initial_cash

    print(f"\n  收益统计:")
    print(f"    初始资金: {initial_cash:,.2f} 元")
    print(f"    最终资产: {account.total_assets:,.2f} 元")
    print(f"    总收益: {account.total_assets - initial_cash:,.2f} 元")
    print(f"    收益率: {total_return:.2%}")

    # 9. 停止引擎
    print("\n[9] 停止引擎...")
    engine.stop()

    print("\n" + "=" * 80)
    print("模拟盘交易完成")
    print("=" * 80)


if __name__ == "__main__":
    run_paper_trading_real_data()
