# A股量化交易系统

[![CI](https://github.com/xmu-csnoob/quant/actions/workflows/ci.yml/badge.svg)](https://github.com/xmu-csnoob/quant/actions/workflows/ci.yml)
[![Release](https://github.com/xmu-csnoob/quant/actions/workflows/release.yml/badge.svg)](https://github.com/xmu-csnoob/quant/actions/workflows/release.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 子仓库

此项目已拆分为多个独立仓库，便于多 Agent 并行开发：

| 仓库 | 职责 | Agent | 状态 |
|------|------|-------|------|
| [quant-data](https://github.com/xmu-csnoob/quant-data) | 数据采集、存储、API | Agent A | ![stars](https://img.shields.io/github/stars/xmu-csnoob/quant-data?style=social) |
| [quant-strategies](https://github.com/xmu-csnoob/quant-strategies) | 策略研究、回测、模型 | Agent B | ![stars](https://img.shields.io/github/stars/xmu-csnoob/quant-strategies?style=social) |
| [quant-trading](https://github.com/xmu-csnoob/quant-trading) | 实盘交易、风控 | Agent C | ![stars](https://img.shields.io/github/stars/xmu-csnoob/quant-trading?style=social) |
| [quant-infra](https://github.com/xmu-csnoob/quant-infra) | 基础设施、文档 | PM | ![stars](https://img.shields.io/github/stars/xmu-csnoob/quant-infra?style=social) |

### 克隆包含子模块的完整仓库

```bash
git clone --recurse-submodules git@github.com:xmu-csnoob/quant.git
cd quant
```

### 更新子模块

```bash
git submodule update --remote --merge
```

---

## 核心目标
通过量化策略在中国A股市场实现稳定盈利

## A股市场特点
- **交易所**：上交所(SSE)、深交所(SZSE)、北交所(BSE)
- **交易时间**：周一至周五 9:30-11:30, 13:00-15:00
- **T+1制度**：当日买入次日才能卖出
- **涨跌幅限制**：主板±10%，创业板/科创板±20%，北交所±30%
- **交易成本**：印花税0.1%（卖出）、佣金（万2.5-万5）、过户费0.001%

## 目录结构

```
quant/
├── src/                         # 核心源代码
│   ├── data/                    # 数据层
│   │   ├── fetchers/            # 数据获取器
│   │   ├── storage/             # 存储层
│   │   ├── cache/               # 缓存层
│   │   └── api/                 # 数据API
│   ├── strategies/              # 策略层
│   │   ├── stock_selection/     # 选股策略
│   │   ├── timing/              # 择时策略
│   │   ├── hedging/             # 对冲策略
│   │   └── arbitrage/           # 套利策略
│   ├── backtesting/             # 回测层
│   │   ├── engines/             # 回测引擎
│   │   └── metrics/             # 绩效指标
│   ├── trading/                 # 交易层
│   │   ├── order_management/    # 订单管理
│   │   ├── slippage/            # 滑点模拟
│   │   └── execution/           # 执行算法
│   ├── risk/                    # 风险管理
│   │   ├── position_limit/      # 仓位限制
│   │   └── drawdown/            # 回撤控制
│   └── utils/                   # 工具函数
│       ├── indicators/          # 技术指标
│       └── features/            # 特征工程
├── apps/                        # 应用脚本
│   ├── data/                    # 数据脚本
│   │   ├── download/            # 数据下载
│   │   └── update/              # 数据更新
│   ├── backtest/                # 回测脚本
│   ├── train/                   # 模型训练
│   ├── live/                    # 实盘交易
│   └── monitor/                 # 监控脚本
├── docs/                        # 文档
│   ├── guides/                  # 指南
│   ├── reports/                 # 报告
│   └── designs/                 # 设计文档
├── data/                        # 数据文件
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后数据
│   └── cache/                   # 缓存数据
├── config/                      # 配置文件
├── tests/                       # 测试
├── tutorial/                    # 学习教程
├── models/                      # 训练好的模型
├── logs/                        # 日志
└── backtest_results/            # 回测结果
```

## 策略方向

### 选股策略
- 多因子选股（价值、成长、质量、动量）
- 行业轮动
- 基本面量化

### 择时策略
- 市场情绪指标
- 资金流向
- 技术形态

### 对冲策略
- 股指期货对冲
- ETF对冲
- 期权保护

## 数据源
- Tushare（免费/付费）
- AkShare（免费）
- 东方财富API
- 同花顺API

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行回测

```bash
# 运行历史回测
python apps/backtest/backtest_historical.py

# 运行ML策略回测
python apps/backtest/backtest_ml_model.py
```

### 3. 连接模拟盘

```bash
# 安装掘金SDK
pip install gm-python

# 访问 https://www.myquant.cn/ 获取Token

# 运行模拟盘交易
python apps/live/live_paper_trading.py
```

## 📊 已实现功能

### 数据层 ✅
- Mock数据生成器（9种市场场景）
- Tushare数据获取器
- 文件存储和LRU缓存

### 策略层 ✅
- 趋势跟踪策略（MA+MACD+RSI）
- 均值回归策略（布林带+RSI）
- ML预测策略（XGBoost）
- 组合策略（投票/加权）
- 动态策略（市场环境识别）

### 回测引擎 ✅
- SimpleBacktester（快速回测）
- 支持多策略对比

### 风控系统 ✅
- 止损止盈（固定比例/移动止损）
- 仓位管理（固定比例/凯利公式/ATR）
- 回撤控制
- 连续亏损保护

### 实时交易 ✅
- 订单管理系统
- 模拟盘API（MockTradingAPI）
- 掘金适配器（GMTradingAdapter）
- 实时交易引擎

## 🎯 模拟盘接入指南

### 方案1：掘金（推荐）

**优点**：
- 真实A股行情
- 完整的模拟盘交易
- 免费使用

**步骤**：
1. 访问 https://www.myquant.cn/
2. 注册账号并实名认证
3. 创建策略获取Token
4. 运行：`pip install gm-python`
5. 运行：`python scripts/run_gm_simulation.py`

### 方案2：米筐RiceQuant

- 美股/港股/A股
- 访问 https://www.ricequant.com/

### 方案3：聚宽JoinQuant

- A股模拟盘
- 访问 https://www.joinquant.com/

## 📈 策略测试结果

| 策略 | 适用场景 | 收益 | 评价 |
|------|---------|------|------|
| 趋势跟踪 | 牛市 | -0.05% | 风险控制好 |
| 均值回归 | 震荡市 | -7.15% | 熊市失效 |
| ML预测 | 多市场 | +5.51% | 方向准确率~50% |
| 有风控 | 熊市 | +7.47% | 显著改善 |

## 免责声明
本系统仅供学习和研究使用，实盘交易有风险，投资需谨慎。

---

## 🐳 Docker部署

### 使用Docker Compose（推荐）

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 单独构建镜像

```bash
# 构建镜像
docker build -t quant:latest .

# 运行容器
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e TUSHARE_TOKEN=your_token \
  quant:latest
```

访问 http://localhost 即可使用。

---

## 🔄 CI/CD流程

项目使用GitHub Actions实现自动化：

| Workflow | 触发条件 | 功能 |
|----------|----------|------|
| `ci.yml` | Push/PR | Python测试、代码检查、前端构建 |
| `release.yml` | Release发布 | 构建发布包、Docker镜像 |
| `scheduled.yml` | 定时/手动 | 数据更新、模型重训练 |

### 本地运行测试

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行所有测试
python -m pytest tests/ -v

# 运行单独模块测试
python tests/test_price_limit.py
python tests/test_t1_manager.py
python tests/test_trade_calendar.py
python tests/test_ml_api.py
python src/backtesting/test_costs.py
```
