"""
特征提取器基础类

所有特征提取器都应继承 BaseFeatureExtractor
"""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseFeatureExtractor(ABC):
    """
    特征提取器基类

    所有特征提取器都应继承此类并实现 extract 方法
    """

    def __init__(self, name: str):
        """
        初始化特征提取器

        Args:
            name: 特征提取器名称
        """
        self.name = name
        self._feature_names: List[str] = []

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取特征

        Args:
            df: 原始数据，必须包含 OHLCV 列

        Returns:
            包含特征列的 DataFrame
        """
        pass

    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表

        Returns:
            特征名称列表
        """
        return self._feature_names.copy()

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        验证输入数据

        Args:
            df: 输入数据

        Raises:
            ValueError: 如果数据不符合要求
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")

        if len(df) == 0:
            raise ValueError("数据不能为空")

    def _register_feature(self, feature_name: str) -> None:
        """
        注册特征名称

        Args:
            feature_name: 特征名称
        """
        if feature_name not in self._feature_names:
            self._feature_names.append(feature_name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
