#!/usr/bin/env python3
"""
计算今日收盘后的持仓收益

从1月30日收盘后开始，计算持仓盈亏
"""

import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage


def main():
    """计算今日收益"""
    print("=" * 60)
    print("今日持仓收益计算")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 加载状态
    state_file = project_root / "data" / "live_trading_state.json"

    if not state_file.exists():
        print("错误: 未找到状态文件")
        return

    with open(state_file, 'r') as f:
        state = json.load(f)

    capital = state['capital']
    positions = state['positions']
    initial_capital = state.get('initial_capital', 1000000)

    print(f"\n现金: {capital:,.2f} 元")
    print(f"持仓: {len(positions)} 只\n")

    # 获取今天的收盘价
    storage = SQLiteStorage()
    today = datetime.now().strftime('%Y%m%d')

    print(f"获取 {today} 收盘价...")

    total_market_value = 0
    position_details = []

    for ts_code, pos in positions.items():
        try:
            df = storage.get_daily_prices(ts_code, today, today)
            if df is None or df.empty:
                print(f"  {ts_code}: 无数据，使用成本价")
                current_price = pos['entry_price']
            else:
                current_price = df['close'].iloc[-1]

            market_value = pos['quantity'] * current_price
            cost_basis = pos['quantity'] * pos['entry_price']
            pnl = market_value - cost_basis
            pnl_ratio = pnl / cost_basis

            total_market_value += market_value

            position_details.append({
                'ts_code': ts_code,
                'quantity': pos['quantity'],
                'entry_price': pos['entry_price'],
                'current_price': current_price,
                'cost_basis': cost_basis,
                'market_value': market_value,
                'pnl': pnl,
                'pnl_ratio': pnl_ratio
            })

        except Exception as e:
            print(f"  {ts_code}: 错误 - {e}")

    # 计算总资产
    total_value = capital + total_market_value
    total_return = (total_value - initial_capital) / initial_capital

    print("\n" + "=" * 60)
    print("持仓明细")
    print("=" * 60)

    for detail in position_details:
        print(f"\n股票: {detail['ts_code']}")
        print(f"  持仓: {detail['quantity']:,} 股")
        print(f"  成本: {detail['entry_price']:.2f} 元")
        print(f"  现价: {detail['current_price']:.2f} 元")
        print(f"  成本基: {detail['cost_basis']:,.2f} 元")
        print(f"  市值: {detail['market_value']:,.2f} 元")
        print(f"  盈亏: {detail['pnl']:>15,.2f} 元 ({detail['pnl_ratio']*100:>6.2f}%)")

    print("\n" + "=" * 60)
    print("账户汇总")
    print("=" * 60)
    print(f"  现金: {capital:,.2f} 元")
    print(f"  持仓市值: {total_market_value:,.2f} 元")
    print(f"  总资产: {total_value:,.2f} 元")
    print(f"  初始资金: {initial_capital:,.2f} 元")
    print(f"  总收益率: {total_return*100:>6.2f}%")
    print(f"  盈亏金额: {total_value - initial_capital:>+15,.2f} 元")
    print("=" * 60)

    # 保存今日净值
    daily_values = state.get('daily_values', [])
    daily_values.append({
        'date': today,
        'total_value': total_value,
        'capital': capital,
        'positions': len(positions),
        'market_value': total_market_value
    })

    # 更新状态
    state['daily_values'] = daily_values
    state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    print(f"\n结果已保存到: {state_file}")
    print(f"下次运行: 明天收盘后")


if __name__ == "__main__":
    main()
