# 上下文与决策记录

## 核心原则
> 不要试图"信任模型"；要设计一个流程，让模型必须提交可验证证据，并让不可验证的部分自然被挡在门外。

## 关键概念

### 收敛性模型
```
B_{t+1} = (1 - r(1-a)) B_t + b
```
- `r` = reviewer发现能力
- `a` = 修复引入新错的概率
- `b` = 每轮改动规模（**关键控制点**）

**结论**: 小PR + 强CI = 更快收敛

### LLM-as-a-Judge的局限性
- GitHub Copilot review 不计入 required approvals
- temperature=0 不等于可靠
- 存在偏差和脆弱性
- **只能做分诊，不能做最终裁判**

## 业界实践参考

| 实践 | 来源 | 我们的应用 |
|------|------|-----------|
| PR模板必填项 | ChatGPT 5.3建议 | Phase 1.1 |
| PR大小限制 | 收敛模型 | Phase 1.2 |
| Agent写测试 | Veracode报告 | Phase 2.1 |
| 分诊式review | GitHub设计 | Phase 2.2 |
| Semgrep diff-aware | 业界标准 | Phase 3.1 |
| Seeded bugs | 学术研究 | Phase 3.2 |

## 项目特定上下文

### Quant系统架构
```
数据层 (fetchers/storage)
    ↓
策略层 (strategies)
    ↓
回测引擎 (backtesting)
    ↓
风控层 (risk)
    ↓
交易执行 (trading)
    ↓
API层 (api)
```

### 历史问题（需转化为回归测试）
1. **Signal无symbol字段** → 多标的下单错误
2. **移动止损逻辑错误** → 永远不触发
3. **T+1全仓锁定** → 加仓后无法卖出
4. **日期格式不匹配** → 统计恒为0

### 高风险模块（需重点关注）
- `src/trading/api.py` - T+1逻辑
- `src/risk/manager.py` - 止损止盈
- `src/api/services/risk_service.py` - 风控统计
- `src/strategies/lstm_strategy.py` - 策略信号

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-02-15 | PR限制500行 | 平衡效率与可审查性 |
| 2026-02-15 | AI不做最终裁判 | 参考GitHub设计 |
| 2026-02-15 | 使用Semgrep | 业界标准，免费 |

## 待确认问题
1. 紧急修复的豁免机制如何设计？（label？审批？）
2. 测试覆盖率阈值设为多少合适？（60%？80%？）
3. 是否需要引入AI review工具（如CodeRabbit）？
