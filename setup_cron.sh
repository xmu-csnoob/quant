#!/bin/bash
#
# 定时任务配置脚本
#
# 用法: bash setup_cron.sh
#

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "量化交易系统 - 定时任务配置"
echo "=========================================="
echo ""
echo "项目路径: $PROJECT_ROOT"
echo ""

# 方案1: 使用 cron (推荐用于简单定时任务)
echo "方案1: Cron 定时任务"
echo "-------------------"
echo "每天 15:30 (A股收盘后) 自动执行交易例程"
echo ""
echo "执行以下命令添加定时任务:"
echo ""
echo "  crontab -e"
echo ""
echo "然后添加以下行:"
echo ""
echo "  # 量化交易系统 - 每日收盘后执行"
echo "  30 15 * * 1-5 cd $PROJECT_ROOT && /usr/bin/python3 scripts/daily_routine.py >> logs/daily_routine.log 2>&1"
echo ""
echo "说明:"
echo "  - 30 15 * * 1-5: 周一到周五 15:30"
echo "  - cd $PROJECT_ROOT: 切换到项目目录"
echo "  - python3 scripts/daily_routine.py: 执行交易例程"
echo "  - >> logs/daily_routine.log: 日志保存到文件"
echo ""

# 方案2: 使用 systemd timer (推荐用于更高级的控制)
echo "=========================================="
echo "方案2: Systemd Timer (更强大)"
echo "=========================================="
echo ""
echo "创建服务文件:"
cat << 'EOF'
sudo tee /etc/systemd/system/quant-trading.service > /dev/null << 'SERVICE'
[Unit]
Description=Quant Trading Daily Routine
After=network.target

[Service]
Type=oneshot
User=YOUR_USERNAME
WorkingDirectory=/home/wenfei/workspace/quant
ExecStart=/usr/bin/python3 scripts/daily_routine.py
StandardOutput=append:/home/wenfei/workspace/quant/logs/daily_routine.log
StandardError=append:/home/wenfei/workspace/quant/logs/daily_routine.log

[Install]
WantedBy=multi-user.target
SERVICE

echo ""
echo "创建定时器文件:"
sudo tee /etc/systemd/system/quant-trading.timer > /dev/null << 'TIMER'
[Unit]
Description=Quant Trading Timer (Weekdays at 15:30)
Requires=quant-trading.service

[Timer]
OnCalendar=Mon-Fri *-*-* 15:30:00
Persistent=true

[Install]
WantedBy=timers.target
TIMER

echo ""
echo "启用并启动定时器:"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable quant-trading.timer"
echo "  sudo systemctl start quant-trading.timer"
echo ""
echo "检查定时器状态:"
echo "  sudo systemctl status quant-trading.timer"
echo "  sudo systemctl list-timers"
echo ""

# 手动测试
echo "=========================================="
echo "手动测试"
echo "=========================================="
echo ""
echo "立即运行一次测试:"
echo "  cd $PROJECT_ROOT"
echo "  python3 scripts/daily_routine.py"
echo ""

# 查看日志
echo "查看日志:"
echo "  tail -f logs/daily_trading.log"
echo ""

echo "配置完成！"
echo "请选择方案1 (cron) 或方案2 (systemd) 进行配置"
