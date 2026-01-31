"""
掘金模拟盘交易运行脚本

使用掘金平台进行模拟盘交易
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.gm_adapter import create_gm_simulation_engine, print_gm_setup_guide
from src.strategies.ma_macd_rsi import MaMacdRsiStrategy


def run_gm_simulation():
    """运行掘金模拟盘"""
    print("=" * 70)
    print("掘金模拟盘交易系统")
    print("=" * 70)

    # 显示设置指南
    print_gm_setup_guide()

    # 获取Token
    print("\n请按上述步骤获取掘金Token")
    token = input("\n请输入掘金Token (按Enter跳过): ").strip()

    if not token:
        print("\nToken为空，进入演示模式...")
        print("\n演示模式下的功能:")
        print("  1. 展示策略如何接入掘金")
        print("  2. 展示交易流程")
        print("  3. 不进行实际交易")
        return

    # 创建策略
    print("\n创建策略...")
    strategy = MaMacdRsiStrategy()

    # 创建引擎
    print("创建掘金模拟盘引擎...")
    symbols = ["600000.SH"]  # 浦发银行

    try:
        engine = create_gm_simulation_engine(
            strategy=strategy,
            symbols=symbols,
            token=token,
            initial_cash=100000,
        )

        print("引擎创建成功")

        # 启动
        print("\n启动引擎...")
        if engine.start():
            print("引擎已启动")

            # 显示状态
            engine.print_status()

            # 在实际应用中，这里会进入主循环
            # 订阅行情 -> 处理信号 -> 下单
            print("\n进入交易循环...")
            print("(实际应用中这里会持续运行)")

            # 等待用户输入
            input("\n按Enter停止引擎...")

            # 停止
            engine.stop()

        else:
            print("引擎启动失败")

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n可能的原因:")
        print("  1. Token错误")
        print("  2. 网络问题")
        print("  3. 掘金服务暂时不可用")
        print("\n请检查后重试")

    print("\n" + "=" * 70)
    print("程序结束")
    print("=" * 70)


if __name__ == "__main__":
    run_gm_simulation()
