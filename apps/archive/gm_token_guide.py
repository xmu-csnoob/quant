"""
掘金Token获取详细指南

帮助用户找到掘金的Token
"""

def print_detailed_guide():
    """打印详细的Token获取指南"""
    print("=" * 70)
    print("掘金Token获取详细指南")
    print("=" * 70)

    print("\n【方法1】通过策略页面获取（推荐）")
    print("-" * 70)
    print("1. 登录掘金: https://www.myquant.cn/")
    print("2. 点击顶部菜单: 策略中心 -> 策略列表")
    print("3. 点击: 创建新策略")
    print("4. 填写策略信息:")
    print("   - 策略名称: 例如 '我的量化策略'")
    print("   - 策略类型: 选择 'Python策略'")
    print("   - 交易市场: 选择 'A股'")
    print("5. 创建后，进入策略详情页")
    print("6. 在策略详情页可以看到:")
    print("   - Token: 一串很长的字符串")
    print("   - 复制这个Token")

    print("\n【方法2】通过账号设置获取")
    print("-" * 70)
    print("1. 登录掘金")
    print("2. 点击右上角头像 -> 账号设置")
    print("3. 查找 'Token' 或 'API密钥' 相关选项")

    print("\n【方法3】在模拟交易页面")
    print("-" * 70)
    print("1. 登录掘金")
    print("2. 点击: 模拟交易")
    print("3. 创建模拟交易账户")
    print("4. 在账户页面可能会显示Token")

    print("\n" + "=" * 70)
    print("Token的位置提示:")
    print("=" * 70)
    print("- 通常在策略设置页面")
    print("- 可能显示为 'Token'、'API密钥'、'访问令牌'")
    print("- 是一串很长的字母数字组合")
    print("- 通常在页面右上角或设置区域")

    print("\n【截图示例】")
    print("-" * 70)
    print("在掘金页面中，你可能会看到:")
    print()
    print("  策略名称: [我的策略]")
    print("  Token:  ************************************")
    print("         [复制按钮]")
    print()
    print("复制 '************************************' 这部分即可")

    print("\n" + "=" * 70)
    print("如果找不到Token，可能的原因:")
    print("=" * 70)
    print("1. 需要先创建一个策略")
    print("2. Token在策略创建后才会显示")
    print("3. 需要完成实名认证")

    print("\n" + "=" * 70)


def print_alternative_solution():
    """打印替代方案"""
    print("=" * 70)
    print("找不到Token？替代方案")
    print("=" * 70)

    print("\n【方案1】使用本地模拟模式")
    print("-" * 70)
    print("如果你暂时找不到Token，可以:")
    print("1. 使用我们内置的MockTradingAPI")
    print("2. 它可以模拟完整的交易流程")
    print("3. 运行: python scripts/test_paper_trading.py")

    print("\n【方案2】先回测，再模拟盘")
    print("-" * 70)
    print("1. 先用历史数据回测: python scripts/test_ensemble.py")
    print("2. 验证策略有效性")
    print("3. 再连接模拟盘")

    print("\n【方案3】联系掘金客服")
    print("-" * 70)
    print("1. 在掘金页面找客服")
    print("2. 咨询如何获取Token")
    print("3. 查看帮助文档")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    print_detailed_guide()

    print("\n\n" + "=" * 70)
    print("现在请在掘金网站查找Token")
    print("找到后告诉我，我会帮你配置")
    print("=" * 70)

    print("\n如果实在找不到，可以先运行:")
    print("  python scripts/test_paper_trading.py")
    print("\n体验本地模拟交易系统")

    print("\n" + "=" * 70)
