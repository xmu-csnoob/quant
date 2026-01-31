#!/bin/bash
# 停止模拟盘实盘测试

echo "========================================"
echo "停止模拟盘实盘测试"
echo "========================================"

if [ ! -f /tmp/live_trading.pid ]; then
    echo "⚠️  模拟盘未在运行"
    exit 0
fi

PID=$(cat /tmp/live_trading.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "正在停止进程 (PID: $PID)..."
    kill $PID
    sleep 2

    if ps -p $PID > /dev/null 2>&1; then
        echo "强制停止..."
        kill -9 $PID
    fi

    rm /tmp/live_trading.pid
    echo "✓ 已停止"
else
    echo "⚠️  进程不存在 (PID: $PID)"
    rm /tmp/live_trading.pid
fi

echo ""
echo "最后状态:"
python3 -c "
import json
from pathlib import Path

state_file = Path('data/live_trading_state.json')
if state_file.exists():
    with open(state_file) as f:
        state = json.load(f)
    print(f\"  资金: {state['capital']:.2f}\")
    print(f\"  持仓: {len(state['positions'])}\")
    print(f\"  交易次数: {len(state['trades'])}\")
    if state['daily_values']:
        latest = state['daily_values'][-1]
        print(f\"  最新净值: {latest['total_value']:.2f}\")
        print(f\"  更新时间: {latest['date']}\")
else:
    print('  无状态文件')
"
echo ""
