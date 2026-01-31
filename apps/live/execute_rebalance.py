#!/usr/bin/env python3
"""
执行调仓操作
"""

import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# A股交易成本
SELL_COST_RATE = 0.0013  # 0.13% 佣金+印花税
BUY_COST_RATE = 0.0003   # 0.03% 佣金


def main():
    """执行调仓"""
    print("=" * 60)
    print("调仓操作")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 加载状态
    state_file = project_root / "data" / "live_trading_state.json"

    with open(state_file, 'r') as f:
        state = json.load(f)

    capital = state['capital']
    positions = state['positions']
    trades = state.get('trades', [])

    print(f"\n调仓前:")
    print(f"  现金: {capital:,.2f} 元")
    print(f"  持仓: {len(positions)} 只")

    # 调仓计划
    # 卖出: 000586.SZ (概率0.5595, 最弱持仓)
    # 买入: 000590.SZ (概率0.7388, 最强信号)

    sell_ts_code = "000586.SZ"
    buy_ts_code = "000590.SZ"
    sell_price = 9.21  # 1月29日收盘价
    buy_price = 6.90   # 1月29日收盘价
    trade_date = datetime.now().strftime('%Y-%m-%d')

    # 执行卖出
    if sell_ts_code in positions:
        pos = positions[sell_ts_code]
        quantity = pos['quantity']
        entry_price = pos['entry_price']

        # 计算卖出收入
        sell_value = quantity * sell_price * (1 - SELL_COST_RATE)
        pnl = sell_value - quantity * entry_price
        pnl_ratio = pnl / (quantity * entry_price)

        print(f"\n🔴 卖出 {sell_ts_code}:")
        print(f"  数量: {quantity:,} 股")
        print(f"  成本: {entry_price:.2f} 元")
        print(f"  卖价: {sell_price:.2f} 元")
        print(f"  收入: {sell_value:,.2f} 元")
        print(f"  盈亏: {pnl:+,.2f} 元 ({pnl_ratio*100:+.2f}%)")

        # 更新现金
        capital += sell_value
        del positions[sell_ts_code]

        # 记录交易
        trades.append({
            "date": trade_date,
            "ts_code": sell_ts_code,
            "action": "sell",
            "price": sell_price,
            "quantity": quantity,
            "amount": sell_value,
            "pnl": pnl,
            "pnl_ratio": pnl_ratio,
            "capital_after": capital
        })

    # 执行买入
    buy_value = capital * 0.3  # 30%仓位
    quantity = int(buy_value / buy_price / 100) * 100  # 整手
    cost = quantity * buy_price * (1 + BUY_COST_RATE)

    if quantity >= 100:
        print(f"\n🟢 买入 {buy_ts_code}:")
        print(f"  可用资金: {buy_value:,.2f} 元")
        print(f"  买入数量: {quantity:,} 股")
        print(f"  买入价格: {buy_price:.2f} 元")
        print(f"  买入成本: {cost:,.2f} 元")

        # 更新现金和持仓
        capital -= cost
        positions[buy_ts_code] = {
            "entry_date": trade_date,
            "entry_price": buy_price,
            "quantity": quantity
        }

        # 记录交易
        trades.append({
            "date": trade_date,
            "ts_code": buy_ts_code,
            "action": "buy",
            "price": buy_price,
            "quantity": quantity,
            "amount": cost,
            "capital_after": capital
        })

    # 更新状态
    state['capital'] = capital
    state['positions'] = positions
    state['trades'] = trades
    state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 添加调仓记录
    if 'rebalance_history' not in state:
        state['rebalance_history'] = []

    state['rebalance_history'].append({
        "date": trade_date,
        "sell": sell_ts_code,
        "buy": buy_ts_code,
        "reason": f"信号优化: {sell_ts_code}概率0.56 -> {buy_ts_code}概率0.74"
    })

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    print(f"\n" + "=" * 60)
    print(f"调仓完成:")
    print(f"  现金: {capital:,.2f} 元")
    print(f"  持仓: {len(positions)} 只")
    print(f"\n当前持仓:")
    for ts_code, pos in positions.items():
        print(f"  {ts_code}: {pos['quantity']:,}股 @ {pos['entry_price']:.2f}")
    print("=" * 60)

    print(f"\n状态已保存到: {state_file}")


if __name__ == "__main__":
    main()
