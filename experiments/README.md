# 策略实验目录

本目录用于存放策略实验和跑批代码，与正式代码分离。

## 目录结构

```
experiments/
├── strategy_results/    # 回测结果
├── README.md            # 本文件
└── *.py                # 实验脚本
```

## 使用方式

```python
# 实验脚本示例
from strategies import MaMacdRsiStrategy
from backtesting import BacktestEngine

# 创建策略
strategy = MaMacdRsiStrategy()

# 运行回测
engine = BacktestEngine()
result = engine.run(strategy, data)

# 保存结果
result.to_csv("experiments/strategy_results/test_20260126.csv")
```

## 注意事项

- 实验代码可以随意修改，不需要太规范
- 验证成功的策略再移到正式代码目录
- 实验结果定期清理
