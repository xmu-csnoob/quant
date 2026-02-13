# Quant - A股量化交易系统

<p align="center">
  <img src="docs/images/logo.png" alt="Quant Logo" width="200">
</p>

<p align="center">
  <strong>专为A股市场设计的AI量化交易平台</strong>
</p>

<p align="center">
  从数据获取、策略研究、回测分析到实盘交易的完整解决方案
</p>

<p align="center">
  <a href="https://github.com/xmu-csnoob/quant/actions/workflows/ci.yml">
    <img src="https://github.com/xmu-csnoob/quant/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/xmu-csnoob/quant/actions/workflows/release.yml">
    <img src="https://github.com/xmu-csnoob/quant/actions/workflows/release.yml/badge.svg" alt="Release">
  </a>
  <a href="https://www.python.org/downloads/release/python-3110/">
    <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python Version">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </a>
</p>

---

## 📰 特性亮点

| 特性 | 描述 |
|------|------|
| 🎯 **A股专项优化** | T+1规则、涨跌停限制、交易成本精确模拟 |
| 🤖 **ML预测引擎** | XGBoost + 58个技术特征，预测5日涨跌方向 |
| 📊 **专业回测** | 支持滑点、成本、T+1的完整回测系统 |
| 🌐 **Web界面** | React前端，实时监控、策略管理、ML预测可视化 |
| 🔒 **风控系统** | 止损止盈、仓位管理、回撤控制 |
| 🔄 **CI/CD** | GitHub Actions自动化测试与部署 |

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/xmu-csnoob/quant.git
cd quant

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 运行回测

```python
from src.backtesting.simple_backtester import SimpleBacktester
from src.strategies.trend_following import MaMacdRsiStrategy

# 加载策略
strategy = MaMacdRsiStrategy()

# 运行回测
backtester = SimpleBacktester(initial_capital=1000000)
result = backtester.run(strategy, data)

print(f"总收益率: {result.total_return:.2%}")
print(f"最大回撤: {result.max_drawdown:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
```

### 启动Web服务

```bash
# 启动后端API
python -m uvicorn src.api.main:app --reload --port 8000

# 启动前端（另一个终端）
cd web && npm install && npm run dev
```

访问 http://localhost:5173 查看界面。

## 📦 核心功能

### 1. 数据层

- **多数据源支持**: Tushare, AkShare, Mock数据
- **本地存储**: SQLite数据库 + 文件缓存
- **自动更新**: 支持定时增量更新

### 2. 策略层

| 策略 | 类型 | 适用场景 |
|------|------|----------|
| `MaMacdRsiStrategy` | 趋势跟踪 | 牛市 |
| `MeanReversionStrategy` | 均值回归 | 震荡市 |
| `MLStrategy` | 机器学习 | 全市场 |
| `EnsembleStrategy` | 组合策略 | 多市场 |
| `AdaptiveDynamicStrategy` | 自适应 | 动态市场 |

### 3. 回测引擎

```python
from src.backtesting.costs import CostConfig
from src.backtesting.slippage import VolumeBasedSlippage

# 精确模拟真实交易环境
backtester = SimpleBacktester(
    initial_capital=1000000,
    cost_config=CostConfig.default(),      # 佣金、印花税、过户费
    slippage_model=VolumeBasedSlippage(),  # 成交量滑点
    enable_t1_rule=True,                   # T+1规则
)
```

### 4. ML预测

```python
from src.api.services.ml_service import MLPredictionService

service = MLPredictionService()

# 预测单只股票
result = service.predict("600519.SH")
print(f"上涨概率: {result.probability:.2%}")
print(f"信号: {result.signal}")  # buy/sell/hold

# 获取TOP信号
top_buys = service.get_top_signals(limit=10, signal_type="buy")
```

### 5. 风险管理

- **止损止盈**: 固定比例、移动止损
- **仓位管理**: 固定比例、凯利公式、ATR-based
- **回撤控制**: 最大回撤限制
- **连续亏损保护**: 自动暂停交易

## 🏗️ 项目结构

```
quant/
├── src/                      # 核心源码
│   ├── api/                  # FastAPI后端服务
│   │   ├── routers/          # API路由
│   │   ├── services/         # 业务逻辑
│   │   └── schemas/          # 数据模型
│   ├── backtesting/          # 回测引擎
│   │   ├── costs.py          # 交易成本
│   │   ├── slippage.py       # 滑点模型
│   │   └── simple_backtester.py
│   ├── data/                 # 数据层
│   │   ├── fetchers/         # 数据获取
│   │   └── storage/          # 数据存储
│   ├── strategies/           # 交易策略
│   ├── trading/              # 交易执行
│   │   ├── price_limit.py    # 涨跌停检查
│   │   └── t1_manager.py     # T+1管理
│   ├── risk/                 # 风险管理
│   └── utils/                # 工具函数
├── web/                      # React前端
│   ├── src/
│   │   ├── pages/            # 页面组件
│   │   │   ├── Dashboard/    # 仪表盘
│   │   │   ├── Trading/      # 交易管理
│   │   │   ├── Strategy/     # 策略中心
│   │   │   ├── ML/           # ML预测
│   │   │   ├── Backtest/     # 回测分析
│   │   │   ├── Risk/         # 风险管理
│   │   │   └── Data/         # 数据中心
│   │   └── api/              # API调用
│   └── package.json
├── apps/                     # 应用脚本
│   ├── train_model.py        # ML模型训练
│   ├── backtest/             # 回测脚本
│   └── live/                 # 实盘脚本
├── tests/                    # 测试用例
├── models/                   # 训练好的模型
├── config/                   # 配置文件
└── data/                     # 数据文件
```

## 📊 性能基准

| 策略 | 年化收益 | 最大回撤 | 夏普比率 | 胜率 |
|------|----------|----------|----------|------|
| 趋势跟踪 | -0.05% | 8.2% | -0.01 | 45% |
| 均值回归 | -7.15% | 15.3% | -0.52 | 42% |
| ML预测 | +5.51% | 12.1% | 0.38 | 52% |
| ML+风控 | +7.47% | 8.5% | 0.65 | 55% |

> 注：以上为模拟回测结果，不代表实际收益。

## 🐳 Docker部署

```bash
# 使用Docker Compose
docker-compose up -d

# 访问
# 前端: http://localhost
# API: http://localhost:8000/docs
```

## 📖 文档

- [系统架构](docs/SYSTEM_SUMMARY.md)
- [API文档](http://localhost:8000/docs)
- [策略开发指南](docs/guides/strategy_development.md)
- [贡献指南](CONTRIBUTING.md)

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

```bash
# 1. Fork仓库
# 2. 创建特性分支
git checkout -b feature/your-feature

# 3. 提交更改
git commit -m "feat: your feature"

# 4. 推送分支
git push origin feature/your-feature

# 5. 创建Pull Request
```

详见 [CLAUDE.md](CLAUDE.md) 中的Git工作流程。

## 📄 许可证

[MIT License](LICENSE)

## 🙏 致谢

- [Tushare](https://tushare.pro/) - 金融数据接口
- [AkShare](https://akshare.akfamily.xyz/) - 开源金融数据
- [XGBoost](https://xgboost.readthedocs.io/) - 梯度提升框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Web框架
- [Ant Design](https://ant.design/) - React UI组件库

---

<p align="center">
  <strong>⚠️ 免责声明</strong>
</p>

<p align="center">
  本系统仅供学习和研究使用，实盘交易有风险，投资需谨慎。
</p>
