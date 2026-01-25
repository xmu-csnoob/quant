"""
简单回测示例：MA金叉策略回测
"""
import pandas as pd


def generate_signals(df):
    """生成交易信号"""
    df['MA5'] = df['收盘'].rolling(window=5).mean()
    df['MA20'] = df['收盘'].rolling(window=20).mean()
    df['信号'] = None

    for i in range(20, len(df)):
        today_ma5 = df.loc[i, 'MA5']
        today_ma20 = df.loc[i, 'MA20']
        yesterday_ma5 = df.loc[i-1, 'MA5']
        yesterday_ma20 = df.loc[i-1, 'MA20']

        if pd.isna(today_ma5) or pd.isna(today_ma20):
            continue

        if yesterday_ma5 <= yesterday_ma20 and today_ma5 > today_ma20:
            df.loc[i, '信号'] = '买入'
        elif yesterday_ma5 >= yesterday_ma20 and today_ma5 < today_ma20:
            df.loc[i, '信号'] = '卖出'

    return df


def simple_backtest(df, initial_capital=100000):
    """
    简单回测

    Args:
        df: 包含价格和信号的DataFrame
        initial_capital: 初始资金

    Returns:
        df: 添加了回测结果的DataFrame
        final_return: 最终收益率
    """
    cash = initial_capital  # 现金
    shares = 0              # 持股数量

    for i in range(len(df)):
        price = df.loc[i, '收盘']
        signal = df.loc[i, '信号']

        if signal == '买入' and shares == 0:
            # 全部买入
            shares = int(cash / price)
            cash = cash - shares * price

        elif signal == '卖出' and shares > 0:
            # 全部卖出
            cash = cash + shares * price
            shares = 0

        # 计算当日总资产
        df.loc[i, '现金'] = cash
        df.loc[i, '持股'] = shares
        df.loc[i, '总资产'] = cash + shares * price

    # 计算收益率
    final_value = df['总资产'].iloc[-1]
    final_return = (final_value - initial_capital) / initial_capital

    return df, final_return


# 示例使用
if __name__ == '__main__':
    # 创建模拟数据（50天）
    dates = [f'2024-01-{i:02d}' for i in range(2, 32)]
    dates += [f'2024-02-{i:02d}' for i in range(1, 22)]

    # 生成模拟价格数据（有趋势的模拟）
    import random
    random.seed(42)
    price = 1850
    closes = []
    for _ in range(50):
        change = random.uniform(-20, 25)
        price = max(1700, min(2100, price + change))
        closes.append(int(price))

    df = pd.DataFrame({
        '日期': dates,
        '收盘': closes
    })

    # 生成信号
    df = generate_signals(df)

    # 回测
    df, final_return = simple_backtest(df, initial_capital=100000)

    # 打印结果
    print("=" * 60)
    print(f"初始资金: 100,000 元")
    print(f"最终资金: {df['总资产'].iloc[-1]:,.0f} 元")
    print(f"收益率: {final_return*100:.2f}%")
    print("=" * 60)

    # 打印交易记录
    print("\n交易记录:")
    print("日期        收盘价    信号      持股    现金        总资产")
    print("-" * 60)
    for _, row in df.iterrows():
        if row['信号'] is not None:
            print(f"{row['日期']}  {row['收盘']:>4}      {row['信号']}  {row['持股']:>3}   {row['现金']:>8.0f}    {row['总资产']:>8.0f}")

    # 计算最大回撤
    df['cummax'] = df['总资产'].cummax()
    df['drawdown'] = (df['总资产'] - df['cummax']) / df['cummax']
    max_drawdown = df['drawdown'].min()
    print(f"\n最大回撤: {max_drawdown*100:.2f}%")
