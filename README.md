# A股量化交易系统

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
├── data/                        # 数据层
│   ├── raw/daily/              # 日线行情
│   │   ├── sse/                # 上交所
│   │   ├── szse/               # 深交所
│   │   └── bse/                # 北交所
│   ├── processed/factors/      # 因子数据
│   │   ├── value/              # 价值因子
│   │   ├── growth/             # 成长因子
│   │   └── momentum/           # 动量因子
│   └── indices/                # 指数数据
├── strategies/                  # 策略层
│   ├── stock_selection/        # 选股策略
│   ├── timing/                 # 择时策略
│   ├── hedging/                # 对冲策略
│   └── arbitrage/              # 套利策略
│       ├── etf/                # ETF套利
│       └── stock_futures/      # 股指期货套利
├── backtesting/                # 回测层
│   ├── engines/                # 回测引擎
│   └── metrics/                # 绩效指标
├── trading/                     # 交易层
│   ├── order_management/       # 订单管理
│   ├── slippage/               # 滑点模拟
│   └── execution/              # 执行算法
├── live_trading/               # 实盘交易
│   ├── broker/                 # 券商接口
│   │   ├── simulated/          # 模拟交易
│   │   └── real/               # 实盘交易
│   └── gateways/               # 数据网关
│       ├── simulated/          # 模拟行情
│       ├── ydx/                # 东方财富
│       └── sina/               # 新浪
├── risk_management/            # 风险管理
│   ├── position_limit/         # 仓位限制
│   ├── drawdown/               # 回撤控制
│   └── sector_limit/           # 行业限制
├── analysis/                    # 分析层
│   ├── performance/            # 绩效分析
│   ├── attribution/            # 归因分析
│   └── regime/                 # 市场状态识别
├── config/                      # 配置文件
├── scripts/                     # 脚本工具
├── utils/                       # 工具函数
│   ├── data_fetchers/          # 数据获取
│   └── indicators/             # 技术指标
├── tests/                       # 测试
├── logs/                        # 日志
└── docs/                        # 文档
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
# 测试风控系统
python scripts/test_risk_management.py

# 测试多策略组合
python scripts/test_ensemble.py

# 测试动态策略
python scripts/test_dynamic_strategy.py
```

### 3. 连接模拟盘

```bash
# 安装掘金SDK
pip install gm-python

# 访问 https://www.myquant.cn/ 获取Token

# 运行模拟盘交易
python scripts/run_gm_simulation.py
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
