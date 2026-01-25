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

1. 安装依赖：`pip install tushare akshare pandas numpy`
2. 获取Tushare Token
3. 配置 `config/a_stock.yaml`
4. 运行数据下载
5. 回测策略
6. 实盘验证

## 免责声明
本系统仅供学习和研究使用，实盘交易有风险，投资需谨慎。
