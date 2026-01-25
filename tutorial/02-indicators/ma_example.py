"""
计算移动平均线 MA5 的简单示例
"""
import pandas as pd

# 示例数据
data = {
    '日期': ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
             '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12'],
    '收盘': [1875, 1888, 1905, 1915, 1935, 1945, 1958, 1962, 1975]
}

df = pd.DataFrame(data)

# 计算 MA5（5日移动平均线）
df['MA5'] = df['收盘'].rolling(window=5).mean()

# 计算 MA20（需要更多数据才能计算）
# df['MA20'] = df['收盘'].rolling(window=20).mean()

print("日期        收盘价    MA5")
print("-" * 28)
for _, row in df.iterrows():
    ma5 = f"{row['MA5']:.1f}" if pd.notna(row['MA5']) else "N/A"
    print(f"{row['日期']}  {row['收盘']}    {ma5}")
