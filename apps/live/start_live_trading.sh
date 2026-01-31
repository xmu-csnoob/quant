#!/bin/bash
# 启动模拟盘实盘测试

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "启动模拟盘实盘测试"
echo "========================================"
echo "项目路径: $PROJECT_ROOT"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 激活虚拟环境
source .venv/bin/activate

# 检查是否已有运行实例
if [ -f /tmp/live_trading.pid ]; then
    PID=$(cat /tmp/live_trading.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  模拟盘已在运行 (PID: $PID)"
        echo "如需重启，请先运行: kill $PID"
        exit 1
    fi
fi

# 后台运行
nohup python3 scripts/live_paper_trading.py > logs/live_trading_stdout.log 2>&1 &

# 保存PID
echo $! > /tmp/live_trading.pid

echo "✓ 模拟盘已启动"
echo "  PID: $(cat /tmp/live_trading.pid)"
echo "  日志: logs/live_trading_$(date +%Y-%m-%d).log"
echo "  标准输出: logs/live_trading_stdout.log"
echo ""
echo "查看实时日志:"
echo "  tail -f logs/live_trading_$(date +%Y-%m-%d).log"
echo ""
echo "停止运行:"
echo "  kill $(cat /tmp/live_trading.pid)"
echo ""
