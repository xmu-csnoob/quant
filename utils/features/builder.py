"""
特征构建器

用于组合多个特征提取器，构建完整的特征集
"""

from typing import List, Optional
import pandas as pd

from .base import BaseFeatureExtractor
from .technical import TechnicalFeatureExtractor


class FeatureBuilder:
    """
    特征构建器

    用于组合多个特征提取器，按顺序提取特征
    """

    def __init__(self, extractors: Optional[List[BaseFeatureExtractor]] = None):
        """
        初始化特征构建器

        Args:
            extractors: 特征提取器列表
        """
        self.extractors = extractors or []
        self._all_feature_names: List[str] = []

    def add_extractor(self, extractor: BaseFeatureExtractor) -> "FeatureBuilder":
        """
        添加特征提取器

        Args:
            extractor: 特征提取器

        Returns:
            self，支持链式调用
        """
        self.extractors.append(extractor)
        return self

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建特征

        Args:
            df: 原始数据

        Returns:
            包含所有特征的DataFrame
        """
        result = df.copy()

        for extractor in self.extractors:
            result = extractor.extract(result)

            # 收集所有特征名
            self._all_feature_names.extend(extractor.get_feature_names())

        return result

    def get_feature_names(self) -> List[str]:
        """
        获取所有特征名称

        Returns:
            特征名称列表
        """
        return list(set(self._all_feature_names))

    def get_feature_matrix(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取特征矩阵（只包含特征列，不包含原始数据）

        Args:
            df: 原始数据
            feature_columns: 指定要返回的特征列，如果为None则返回所有特征

        Returns:
            只包含特征列的DataFrame
        """
        # 构建完整特征
        result = self.build(df)

        # 获取特征列
        if feature_columns is None:
            feature_columns = self.get_feature_names()

        # 过滤出特征列
        existing_features = [col for col in feature_columns if col in result.columns]

        return result[existing_features]

    @classmethod
    def create_default(cls) -> "FeatureBuilder":
        """
        创建默认的特征构建器（包含技术指标特征提取器）

        Returns:
            配置好的FeatureBuilder实例
        """
        builder = cls()
        builder.add_extractor(TechnicalFeatureExtractor())
        return builder

    def __repr__(self) -> str:
        extractor_names = [e.name for e in self.extractors]
        return f"FeatureBuilder(extractors={extractor_names})"
