# Code Review Checklist

> AI辅助Review指南：指出高风险点+复现路径，不做最终裁判
> 最终决定权在人类Reviewer

## 使用方式

1. PR作者：提交前自查
2. Reviewer：Review时逐项检查
3. AI Review：自动化检查（高风险项）

---

## 1. 数据/金融风险 (高风险)

### 1.1 前视偏差 (Look-ahead Bias)
- [ ] 时间序列操作是否使用 `shift()` 避免未来数据泄露
- [ ] 特征计算是否只使用历史数据
- [ ] 标签计算是否正确shift到未来

**复现路径**:
```python
# 检查特征是否使用了未来数据
df['feature'].head(10)  # 前几行应该是NaN（因为shift）
df['label'].tail(10)    # 最后几行应该是NaN（因为预测未来）
```

### 1.2 除零保护
- [ ] 所有除法操作是否有除零保护
- [ ] 使用 `np.divide(..., where=...)` 或条件判断

**复现路径**:
```python
# 测试除零场景
import numpy as np
result = np.divide([1, 2, 3], [0, 0, 0], where=[False, False, False],
                   out=np.zeros(3))
assert np.all(result == 0)
```

### 1.3 边界条件
- [ ] 价格为0或负数的处理
- [ ] 成交量为0的处理
- [ ] 空DataFrame的处理

**复现路径**:
```python
# 测试边界条件
df_empty = pd.DataFrame()
df_zero = pd.DataFrame({'close': [0], 'volume': [0]})
# 函数应该优雅处理这些情况
```

---

## 2. 交易逻辑风险 (高风险)

### 2.1 T+1规则
- [ ] 买入当天是否禁止卖出
- [ ] `T1Manager.can_sell()` 是否正确调用
- [ ] 持仓日期是否正确记录

**复现路径**:
```python
from src.trading.t1_manager import T1Manager
from datetime import date

manager = T1Manager()
manager.record_buy("600000.SH", 1000, date.today(), 10.0)
assert manager.can_sell("600000.SH", 1000, date.today()) == False
assert manager.can_sell("600000.SH", 1000, date.today() + timedelta(days=1)) == True
```

### 2.2 仓位限制
- [ ] 单只股票不超过总资产30%
- [ ] 持仓数量不超过3只
- [ ] 止损/止盈逻辑是否正确

**复现路径**:
```python
from src.risk.manager import RiskManager

rm = RiskManager(max_positions=3, max_position_pct=0.3)
# 模拟大额买入
can_buy = rm.check_position_limit(current_value=100000, total_assets=300000)
assert can_buy == False  # 超过30%
```

### 2.3 涨跌停处理
- [ ] 涨停股票是否禁止买入
- [ ] 跌停股票是否禁止卖出
- [ ] 不同板块涨跌停幅度（10%/20%/30%/5%）

**复现路径**:
```python
# 测试涨停场景
close = 10.0
limit_up = close * 1.1  # 主板涨停价
# 买入价格 >= 涨停价 应该被拒绝
```

---

## 3. 代码质量风险

### 3.1 错误处理
- [ ] 外部API调用是否有try/except
- [ ] 异常是否正确记录日志
- [ ] 是否有合理的默认值/fallback

**复现路径**:
```python
# 模拟API失败
import pytest
with pytest.raises(Exception):
    fetcher.get_data("invalid_code")
```

### 3.2 资源管理
- [ ] 数据库连接是否正确关闭
- [ ] 文件句柄是否正确关闭
- [ ] 使用context manager (`with` 语句)

**复现路径**:
```python
# 检查资源是否释放
import gc
gc.collect()
# 检查是否有未关闭的资源
```

### 3.3 类型安全
- [ ] 是否有类型注解
- [ ] 输入参数是否有验证
- [ ] 返回值类型是否一致

---

## 4. 安全风险

### 4.1 敏感信息
- [ ] API Token是否通过环境变量读取
- [ ] 密码/密钥是否硬编码
- [ ] `.env` 文件是否在 `.gitignore` 中

**复现路径**:
```bash
# 检查是否有泄露的密钥
grep -r "token\s*=" --include="*.py" | grep -v "os.environ"
grep -r "password\s*=" --include="*.py" | grep -v "os.environ"
```

### 4.2 注入风险
- [ ] SQL查询是否使用参数化
- [ ] 命令执行是否有输入验证
- [ ] 文件路径是否有路径遍历保护

**复现路径**:
```python
# 测试SQL注入
malicious_input = "600000.SH'; DROP TABLE prices; --"
# 应该被安全处理，不执行恶意SQL
```

---

## 5. 性能风险

### 5.1 数据处理
- [ ] 大数据集是否分批处理
- [ ] 是否避免在循环中创建DataFrame
- [ ] 是否使用向量化操作

### 5.2 内存管理
- [ ] 是否有不必要的内存复制
- [ ] 大对象是否及时释放
- [ ] 是否使用生成器替代列表

**复现路径**:
```python
# 内存使用测试
import tracemalloc
tracemalloc.start()
# 执行代码
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
```

---

## 6. 测试覆盖

### 6.1 单元测试
- [ ] 核心逻辑是否有单元测试
- [ ] 边界条件是否有测试
- [ ] 异常情况是否有测试

### 6.2 集成测试
- [ ] 模块间交互是否有测试
- [ ] 端到端流程是否有测试

**复现路径**:
```bash
# 运行测试并检查覆盖率
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Review决策指南

### 必须修复 (Blocker)
- 前视偏差
- 除零错误
- T+1规则违反
- 仓位限制违反
- 敏感信息泄露

### 建议修复 (Major)
- 错误处理缺失
- 性能问题
- 测试覆盖不足

### 可选改进 (Minor)
- 代码风格
- 文档完善
- 重构建议

---

## AI Review自动化

高风险项目可使用以下脚本自动检查：

```bash
# 1. 检查前视偏差（特征计算是否使用shift）
grep -r "\.shift(" src/utils/features/ --include="*.py" -c

# 2. 检查除零保护
grep -r "np.divide\|where=" src/ --include="*.py" -c

# 3. 检查敏感信息
grep -rE "(token|password|secret)\s*=\s*['\"]" src/ --include="*.py"

# 4. 运行安全扫描
pip install bandit && bandit -r src/

# 5. 运行测试
pytest tests/ -v --tb=short
```

---

*Last updated: 2026-02-22*
