"""
LSTM交易策略

使用训练好的LSTM模型进行交易信号预测
"""

import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.strategies.base import BaseStrategy, Signal, SignalType
from src.utils.features.enhanced_features import EnhancedFeatureExtractor


class LSTMStrategy(BaseStrategy):
    """
    LSTM深度学习交易策略

    使用PyTorch LSTM模型预测股票涨跌概率：
    - 概率 > 0.60: 买入信号
    - 概率 < 0.40: 卖出信号
    - 其他: 持有
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_extractor: EnhancedFeatureExtractor,
        scaler,
        feature_cols: List[str],
        seq_len: int = 20,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
        prediction_period: int = 5,
        device: str = "cpu",
    ):
        """
        初始化LSTM策略

        Args:
            model: 训练好的LSTM模型
            feature_extractor: 特征提取器
            scaler: StandardScaler
            feature_cols: 特征列名列表
            seq_len: 序列长度
            buy_threshold: 买入阈值（概率>此值买入）
            sell_threshold: 卖出阈值（概率<此值卖出）
            prediction_period: 预测周期
            device: 设备 (cpu/cuda)
        """
        super().__init__(name="LSTM_Strategy")

        self.model = model.to(device)
        self.model.eval()  # 设置为评估模式
        self.feature_extractor = feature_extractor
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.prediction_period = prediction_period
        self.device = device

        logger.info(
            f"LSTM策略初始化: 特征数={len(feature_cols)}, "
            f"序列长度={seq_len}, 买入阈值={buy_threshold}, 卖出阈值={sell_threshold}"
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        生成交易信号

        Args:
            df: 原始OHLCV数据

        Returns:
            信号列表
        """
        signals = []

        # 提取特征
        df = self.feature_extractor.extract(df)

        # 获取特征列
        feature_cols = self.feature_cols
        if not feature_cols:
            feature_cols = [c for c in df.columns if c.startswith("f_")]

        # 过滤有效数据
        df_valid = df.dropna(subset=feature_cols).copy()

        if len(df_valid) < self.seq_len:
            logger.warning("数据不足，无法生成信号")
            return signals

        # 提取特征并标准化
        features = df_valid[feature_cols].values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = self.scaler.transform(features)

        # 构建序列并预测
        self.model.eval()
        position = None

        with torch.no_grad():
            for i in range(self.seq_len, len(features)):
                # 构建输入序列
                seq = features[i - self.seq_len:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

                # 预测
                prob, _ = self.model(seq_tensor)
                prob = prob.item()

                # 获取当前行数据
                row = df_valid.iloc[i]
                date_str = row["trade_date"]
                if hasattr(date_str, "strftime"):
                    date_str = date_str.strftime("%Y%m%d")
                elif not isinstance(date_str, str):
                    date_str = str(date_str)[:8]

                # 生成信号
                if position is None:
                    # 无持仓，检查买入
                    if prob > self.buy_threshold:
                        signals.append(Signal(
                            date=date_str,
                            signal_type=SignalType.BUY,
                            price=row["close"],
                            reason=f"LSTM预测上涨概率={prob:.2%} > 阈值{self.buy_threshold:.0%}",
                            confidence=prob,
                        ))
                        position = "long"
                else:
                    # 有持仓，检查卖出
                    if prob < self.sell_threshold:
                        signals.append(Signal(
                            date=date_str,
                            signal_type=SignalType.SELL,
                            price=row["close"],
                            reason=f"LSTM预测上涨概率={prob:.2%} < 阈值{self.sell_threshold:.0%}",
                            confidence=prob,
                        ))
                        position = None

        logger.info(f"LSTM策略生成 {len(signals)} 个信号")
        return signals

    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """
        预测最新数据

        Args:
            df: 原始OHLCV数据

        Returns:
            预测概率 (0-1)，如果数据不足返回None
        """
        # 提取特征
        df = self.feature_extractor.extract(df)

        # 获取特征列
        feature_cols = self.feature_cols
        if not feature_cols:
            feature_cols = [c for c in df.columns if c.startswith("f_")]

        # 过滤有效数据
        df_valid = df.dropna(subset=feature_cols).copy()

        if len(df_valid) < self.seq_len:
            logger.warning("数据不足，无法预测")
            return None

        # 提取最近seq_len天的特征
        features = df_valid[feature_cols].values[-self.seq_len:]
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = self.scaler.transform(features)

        # 预测
        self.model.eval()
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            prob, _ = self.model(seq_tensor)
            return prob.item()

    @classmethod
    def load(
        cls,
        model_path: str = "models/lstm_best.pt",
        scaler_path: str = "models/lstm_scaler.pkl",
        feature_path: str = "models/feature_cols.json",
        info_path: str = "models/model_info.json",
        device: str = "auto",
        **kwargs,
    ) -> Optional["LSTMStrategy"]:
        """
        从文件加载LSTM策略

        Args:
            model_path: 模型文件路径
            scaler_path: Scaler文件路径
            feature_path: 特征列文件路径
            info_path: 模型信息文件路径
            device: 设备
            **kwargs: 其他参数覆盖

        Returns:
            LSTMStrategy实例，加载失败返回None
        """
        try:
            # 设置设备
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # 检查文件存在
            if not Path(model_path).exists():
                logger.error(f"模型文件不存在: {model_path}")
                return None

            # 加载模型信息
            info = {}
            if Path(info_path).exists():
                with open(info_path, "r") as f:
                    info = json.load(f)
                logger.info(f"加载模型信息: {info.get('model_version', 'unknown')}")

            # 加载特征列
            if not Path(feature_path).exists():
                logger.error(f"特征文件不存在: {feature_path}")
                return None

            with open(feature_path, "r") as f:
                feature_cols = json.load(f)

            input_size = len(feature_cols)
            seq_len = info.get("seq_len", 20)
            prediction_period = info.get("prediction_period", 5)

            # 创建模型
            from src.models.lstm_model import StockLSTMClassifier
            model = StockLSTMClassifier(input_size=input_size)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            logger.info(f"模型加载成功: {model_path}")

            # 加载scaler
            if not Path(scaler_path).exists():
                logger.error(f"Scaler文件不存在: {scaler_path}")
                return None

            import joblib
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler加载成功: {scaler_path}")

            # 创建特征提取器
            feature_extractor = EnhancedFeatureExtractor(prediction_period=prediction_period)

            # 创建策略
            strategy = cls(
                model=model,
                feature_extractor=feature_extractor,
                scaler=scaler,
                feature_cols=feature_cols,
                seq_len=seq_len,
                prediction_period=prediction_period,
                device=device,
                **kwargs,
            )

            return strategy

        except Exception as e:
            logger.error(f"加载LSTM策略失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def create_lstm_strategy(
    device: str = "auto",
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.40,
) -> Optional[LSTMStrategy]:
    """
    创建LSTM策略（便捷函数）

    Args:
        device: 设备
        buy_threshold: 买入阈值
        sell_threshold: 卖出阈值

    Returns:
        LSTMStrategy实例
    """
    return LSTMStrategy.load(
        device=device,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
