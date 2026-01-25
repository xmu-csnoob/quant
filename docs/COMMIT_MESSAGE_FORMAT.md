# Git Commit Message 格式规范

## 格式定义

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Type 类型

| Type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug修复 |
| `docs` | 文档更新 |
| `style` | 代码格式（不影响功能） |
| `refactor` | 重构（不是新功能也不是修复） |
| `test` | 测试相关 |
| `chore` | 构建/工具/依赖更新 |

## Scope 范围

针对本项目的主要模块：

| Scope | 说明 |
|-------|------|
| `data` | 数据层（数据获取/处理/存储） |
| `strategies` | 策略层（选股/择时/对冲/套利） |
| `backtesting` | 回测层（回测引擎/绩效指标） |
| `trading` | 交易层（订单管理/执行算法） |
| `live_trading` | 实盘交易（券商接口/数据网关） |
| `risk_management` | 风险管理（仓位/回撤控制） |
| `analysis` | 分析层（绩效/归因/市场状态） |
| `config` | 配置文件 |
| `utils` | 工具函数（技术指标等） |
| `tests` | 测试代码 |
| `docs` | 文档 |

## Subject 主题

- 使用中文描述
- 不超过50个字
- 首字母小写
- 不以句号结尾
- 使用祈使语气

## Body 正文（可选）

- 详细说明本次提交的内容
- 可以分多条说明，每行以 `-` 开头

## Footer 脚注（可选）

- 关联Issue: `Close #123`, `Related #456`
- BREAKING CHANGE: 不兼容变更

## 示例

### 新功能
```
feat(strategies): 添加多因子选股策略

- 实现价值因子（PE/PB/PS）
- 实现成长因子（营收增长/利润增长）
- 实现动量因子（20日/60日收益率）
```

### Bug修复
```
fix(data): 修复Tushare数据获取日期格式问题

Tushare API要求日期格式为YYYYMMDD，原代码使用YYYY-MM-DD导致查询失败
```

### 重构
```
refactor(backtesting): 重构回测引擎以支持多策略并行

- 将单策略回测改为抽象基类
- 实现多策略组合回测
```

### 配置更新
```
chore(config): 更新A股交易手续费配置

- 调整佣金率从万3降至万2.5
- 添加北交所涨跌幅±30%配置
```

### 文档更新
```
docs: 添加技术指标使用文档

详细说明MA/MACD/RSI/布林带等常用指标的计算方法和使用场景
```
