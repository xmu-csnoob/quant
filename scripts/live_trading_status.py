#!/usr/bin/env python3
"""查询模拟盘状态"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

state_file = project_root / "data" / "live_trading_state.json"

print("=" * 60)
print("模拟盘状态查询")
print("=" * 60)

if not state_file.exists():
    print("⚠️  状态文件不存在，模拟盘尚未运行")
    sys.exit(1)

with open(state_file) as f:
    state = json.load(f)

print(f"\n账户信息:")
print(f"  初始资金: {100000:.2f}")
print(f"  当前现金: {state['capital']:.2f}")
print(f"  持仓数量: {len(state['positions'])}")
print(f"  交易次数: {len(state['trades'])}")
print(f"  最后更新: {state.get('last_update', '未知')}")

if state['daily_values']:
    latest = state['daily_values'][-1]
    total_value = latest['total_value']
    total_return = (total_value - 100000) / 100000

    print(f"\n净值信息:")
    print(f"  当前净值: {total_value:.2f}")
    print(f"  总收益率: {total_return*100:.2f}%")

    # 净值曲线
    print(f"\n净值记录: {len(state['daily_values'])} 天")
    if len(state['daily_values']) >= 2:
        values_df = pd.DataFrame(state['daily_values'])
        print(f"  最早: {values_df['date'].iloc[0]}")
        print(f"  最新: {values_df['date'].iloc[-1]}")

if state['positions']:
    print(f"\n当前持仓:")
    for ts_code, pos in state['positions'].items():
        entry_price = pos['entry_price']
        quantity = pos['quantity']
        entry_date = pos['entry_date']
        cost = entry_price * quantity
        print(f"  {ts_code}:")
        print(f"    买入日: {entry_date}")
        print(f"    数量: {quantity}股")
        print(f"    成本: {entry_price:.2f} / {cost:.2f}")

if state['trades']:
    trades_df = pd.DataFrame(state['trades'])

    print(f"\n交易统计:")

    # 最近交易
    print(f"\n最近5笔交易:")
    for i, trade in enumerate(state['trades'][-5:], 1):
        action = trade['action']
        ts_code = trade['ts_code']
        price = trade['price']
        quantity = trade['quantity']
        date = trade['date']
        action_cn = '买入' if action == 'buy' else '卖出'

        if action == 'sell':
            pnl = trade.get('pnl', 0)
            pnl_ratio = trade.get('pnl_ratio', 0) * 100
            print(f"  {i}. [{date}] {action_cn} {ts_code} {quantity}股 @{price:.2f} | "
                  f"盈亏: {pnl:.2f} ({pnl_ratio:+.2f}%)")
        else:
            print(f"  {i}. [{date}] {action_cn} {ts_code} {quantity}股 @{price:.2f}")

    # 盈亏统计
    sell_trades = [t for t in state['trades'] if t['action'] == 'sell']
    if sell_trades:
        total_pnl = sum(t.get('pnl', 0) for t in sell_trades)
        win_count = sum(1 for t in sell_trades if t.get('pnl', 0) > 0)
        win_rate = win_count / len(sell_trades) * 100

        print(f"\n已完成交易统计:")
        print(f"  交易数: {len(sell_trades)}")
        print(f"  总盈亏: {total_pnl:.2f}")
        print(f"  胜率: {win_rate:.2f}%")

print("\n" + "=" * 60)
