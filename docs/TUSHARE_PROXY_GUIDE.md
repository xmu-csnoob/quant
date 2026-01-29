# Tushare代理API使用指南

## 快速开始

### 1. 测试代理API

```bash
# 测试是否工作
python scripts/test_proxy_download.py
```

### 2. 批量下载所有A股

```bash
# 下载全部A股（约5000只，需要10-20分钟）
python scripts/download_a_shares.py
```

### 3. 在代码中使用

```python
from data.fetchers.tushare import TushareDataFetcher

# 使用代理API（无频率限制）
fetcher = TushareDataFetcher(
    token="464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976",
    proxy_url="http://lianghua.nanyangqiankun.top"
)

# 获取数据
df = fetcher.get_daily_price("600519.SH", "20200101", "20241231")
```

### 4. 环境变量配置

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export TUSHARE_TOKEN="464ceb6993e02bc20232c10a279ba9bc8fc96b05374d9419b19c0d104976"
export TUSHARE_PROXY_URL="http://lianghua.nanyangqiankun.top"
```

## 对比

| 特性 | 官方API | 代理API |
|------|---------|---------|
| Token申请 | 需要注册 | 无需申请 |
| 频率限制 | 有（免费版） | 无 |
| 数据完整性 | 完整 | 完整 |
| 稳定性 | 高 | 中等 |
| 速度 | 一般 | 快 |
| 适用场景 | 生产环境 | 开发/测试 |

## 当前数据库状态

```bash
# 查看数据库统计
python -c "
from data.storage.sqlite_storage import SQLiteStorage
storage = SQLiteStorage()
stats = storage.get_stats()
print(f'股票数: {stats[\"stock_count\"]}')
print(f'数据行: {stats[\"total_rows\"]:,}')
print(f'大小: {stats[\"db_size_mb\"]:.2f} MB')
"
```

## 注意事项

1. **仅供学习使用** - 代理API可能不稳定
2. **数据备份** - 定期备份 `data/quant.db`
3. **数据验证** - 下载后验证数据完整性

## SQL查询示例

```python
from data.storage.sqlite_storage import SQLiteStorage
import pandas as pd

storage = SQLiteStorage()

# 查询所有股票
stocks = storage.get_all_stocks()

# 查询特定股票数据
df = storage.get_daily_prices("600519.SH", "20200101", "20241231")

# SQL查询
conn = storage.conn
df = pd.read_sql_query("""
    SELECT ts_code, trade_date, close
    FROM daily_prices
    WHERE ts_code = '600519.SH'
    ORDER BY trade_date DESC
    LIMIT 10
""", conn)
```
