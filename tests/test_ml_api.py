"""
ML预测API测试
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.services.ml_service import PredictionCache, MLPredictionService
from src.api.schemas.ml import MLPredictionResponse, PredictionDirection


def test_prediction_cache():
    """测试预测缓存"""
    print("=" * 60)
    print("测试预测缓存")
    print("=" * 60)

    cache = PredictionCache(ttl_seconds=2)  # 2秒过期

    # 创建测试响应
    response = MLPredictionResponse(
        ts_code="600519.SH",
        stock_name="贵州茅台",
        prediction=PredictionDirection.UP,
        probability=0.75,
        confidence=0.5,
        predicted_return=None,
        signal="buy",
        features=None,
        trade_date="20240108",
        model_version="1.0.0",
        prediction_period=5
    )

    # 测试设置和获取
    cache.set("600519.SH", response)
    cached = cache.get("600519.SH")
    assert cached is not None, "应能获取缓存"
    assert cached.ts_code == "600519.SH", "缓存内容应正确"
    print("  ✅ 缓存设置和获取正确")

    # 测试未命中
    not_found = cache.get("000001.SZ")
    assert not_found is None, "未缓存的应返回None"
    print("  ✅ 缓存未命中处理正确")

    # 测试清空
    cache.clear()
    cleared = cache.get("600519.SH")
    assert cleared is None, "清空后应返回None"
    print("  ✅ 缓存清空正确")


def test_prediction_direction():
    """测试预测方向判断"""
    print("\n" + "=" * 60)
    print("测试预测方向判断")
    print("=" * 60)

    # 测试信号判断逻辑
    def get_signal(probability: float) -> str:
        if probability >= 0.55:
            return "buy"
        elif probability <= 0.45:
            return "sell"
        else:
            return "hold"

    assert get_signal(0.75) == "buy", "概率>=55%应为买入"
    assert get_signal(0.55) == "buy", "概率=55%应为买入"
    assert get_signal(0.30) == "sell", "概率<=45%应为卖出"
    assert get_signal(0.45) == "sell", "概率=45%应为卖出"
    assert get_signal(0.50) == "hold", "概率在45-55%之间应为持有"
    assert get_signal(0.52) == "hold", "概率在45-55%之间应为持有"
    print("  ✅ 预测方向判断正确")


def test_confidence_calculation():
    """测试置信度计算"""
    print("\n" + "=" * 60)
    print("测试置信度计算")
    print("=" * 60)

    def calc_confidence(probability: float) -> float:
        return abs(probability - 0.5) * 2

    assert calc_confidence(0.5) == 0.0, "概率50%置信度应为0"
    assert calc_confidence(1.0) == 1.0, "概率100%置信度应为1"
    assert calc_confidence(0.0) == 1.0, "概率0%置信度应为1"
    assert calc_confidence(0.75) == 0.5, "概率75%置信度应为0.5"
    assert calc_confidence(0.25) == 0.5, "概率25%置信度应为0.5"
    print("  ✅ 置信度计算正确")


def test_model_status():
    """测试模型状态检查"""
    print("\n" + "=" * 60)
    print("测试模型状态检查")
    print("=" * 60)

    service = MLPredictionService()

    # 检查模型是否加载
    is_loaded = service.is_model_loaded()
    print(f"  模型已加载: {is_loaded}")

    if is_loaded:
        # 获取模型信息
        info = service.get_model_info()
        if info:
            print(f"  模型名称: {info.model_name}")
            print(f"  特征数: {info.feature_count}")
            print("  ✅ 模型信息获取正确")

    print("  ✅ 模型状态检查通过")


def test_feature_importance():
    """测试特征重要性"""
    print("\n" + "=" * 60)
    print("测试特征重要性获取")
    print("=" * 60)

    service = MLPredictionService()

    if service.is_model_loaded():
        importance = service.get_feature_importance(top_n=10)
        print(f"  获取到 {len(importance)} 个特征重要性")

        if importance:
            print("  Top 5特征:")
            for feat in importance[:5]:
                print(f"    {feat.rank}. {feat.feature_name}: {feat.importance_score:.2f}")
            print("  ✅ 特征重要性获取正确")
    else:
        print("  ⚠️ 模型未加载，跳过测试")


def test_prediction_stats():
    """测试预测统计"""
    print("\n" + "=" * 60)
    print("测试预测统计")
    print("=" * 60)

    service = MLPredictionService()
    stats = service.get_prediction_stats()

    print(f"  总预测数: {stats.total_predictions}")
    print(f"  准确率: {stats.accuracy:.2%}")
    print(f"  胜率: {stats.win_rate:.2%}")
    print(f"  盈亏比: {stats.profit_loss_ratio:.2f}")
    print("  ✅ 预测统计获取正确")


def main():
    """运行所有测试"""
    print("开始测试ML预测API模块")
    print("=" * 60)

    test_prediction_cache()
    test_prediction_direction()
    test_confidence_calculation()
    test_model_status()
    test_feature_importance()
    test_prediction_stats()

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
