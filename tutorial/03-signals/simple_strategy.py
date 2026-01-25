"""
简单交易策略示例：MA金叉策略
"""
import pandas as pd

def generate_signals(df):
    """
    生成交易信号：MA金叉买入，死叉卖出

    Args:
        df: 包含收盘价的DataFrame

    Returns:
        添加了信号列的DataFrame
    """
    # 计算 MA5 和 MA20
    df['MA5'] = df['收盘'].rolling(window=5).mean()
    df['MA20'] = df['收盘'].rolling(window=20).mean()

    # 初始化信号列
    df['信号'] = None

    # 遍历每一天（从第21天开始）
    for i in range(20, len(df)):
        today_ma5 = df.loc[i, 'MA5']
        today_ma20 = df.loc[i, 'MA20']
        yesterday_ma5 = df.loc[i-1, 'MA5']
        yesterday_ma20 = df.loc[i-1, 'MA20']

        # 检查数据是否有效
        if pd.isna(today_ma5) or pd.isna(today_ma20):
            continue

        # 金叉：MA5 上穿 MA20（买入信号）
        if yesterday_ma5 <= yesterday_ma20 and today_ma5 > today_ma20:
            df.loc[i, '信号'] = '买入'

        # 死叉：MA5 下穿 MA20（卖出信号）
        elif yesterday_ma5 >= yesterday_ma20 and today_ma5 < today_ma20:
            df.loc[i, '信号'] = '卖出'

    return df


# 示例使用
if __name__ == '__main__':
    # 创建示例数据（需要至少20天数据才能计算MA20）
    dates = [f'2024-01-{i:02d}' for i in range(2, 32)]
    # 生成模拟数据（简化）
    closes = [
        1850, 1860, 1870, 1865, 1875, 1888, 1905, 1915, 1935, 1945,
        1958, 1962, 1975, 1980, 1990, 1985, 1995, 2005, 2010, 2000,
        1995, 2005, 2015, 2020, 2030, 2025, 2035, 2040, 2050, 2055
    ]

    df = pd.DataFrame({
        '日期': dates,
        '收盘': closes
    })

    # 生成信号
    df = generate_signals(df)

    # 打印有信号的日期
    print("日期        收盘价    MA5     MA20    信号")
    print("-" * 50)
    for _, row in df.iterrows():
        if row['信号'] is not None:
            ma5 = f"{row['MA5']:.1f}" if pd.notna(row['MA5']) else "N/A"
            ma20 = f"{row['MA20']:.1f}" if pd.notna(row['MA20']) else "N/A"
            print(f"{row['日期']}  {row['收盘']}    {ma5}   {ma20}   {row['信号']}")
