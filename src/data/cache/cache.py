"""
数据缓存模块

提供 LRU 缓存功能，用于加速数据访问
"""

from collections import OrderedDict
from typing import Optional
import pandas as pd
from loguru import logger


class DataCache:
    """
    数据缓存 - LRU 策略

    特点：
    1. LRU（Least Recently Used）缓存策略
    2. 自动淘汰最久未使用的数据
    3. 缓存统计（命中率、未命中次数）
    4. 线程安全（单线程环境）

    用途：
    - 缓存最近访问的股票数据
    - 避免重复读取文件
    - 减少 API 调用
    """

    def __init__(self, size: int = 100):
        """
        初始化缓存

        Args:
            size: 最大缓存条目数（默认 100）
                  每条约 100KB，100 条约 10MB
        """
        self.cache: OrderedDict = OrderedDict()
        self.size = size
        self.hits = 0
        self.misses = 0

        logger.debug(f"DataCache initialized with size={size}")

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        获取缓存

        如果命中，移动到末尾（标记为最近使用）

        Args:
            key: 缓存键

        Returns:
            缓存的 DataFrame，如果不存在返回 None
        """
        if key in self.cache:
            self.hits += 1
            # 移到末尾（标记为最近使用）
            self.cache.move_to_end(key)
            logger.debug(f"Cache hit for key: {key}")
            return self.cache[key].copy()

        self.misses += 1
        logger.debug(f"Cache miss for key: {key}")
        return None

    def put(self, key: str, value: pd.DataFrame) -> None:
        """
        存入缓存

        如果达到上限，删除最旧的条目（LRU）

        Args:
            key: 缓存键
            value: 要缓存的 DataFrame
        """
        # 如果键已存在，先删除（后续会重新添加到末尾）
        if key in self.cache:
            del self.cache[key]

        # 达到上限，删除最旧的（第一个）
        elif len(self.cache) >= self.size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Cache full, evicted: {oldest_key}")

        # 添加到末尾
        self.cache[key] = value.copy()
        logger.debug(f"Cache put: {key} ({len(value)} rows)")

    def exists(self, key: str) -> bool:
        """
        检查键是否存在

        Args:
            key: 缓存键

        Returns:
            True 如果存在，否则 False
        """
        return key in self.cache

    def remove(self, key: str) -> bool:
        """
        删除指定键

        Args:
            key: 缓存键

        Returns:
            True 如果删除成功，否则 False
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Cache removed: {key}")
            return True

        return False

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.debug("Cache cleared")

    def get_stats(self) -> dict:
        """
        获取缓存统计

        Returns:
            统计信息字典，包含：
            - hits: 命中次数
            - misses: 未命中次数
            - hit_rate: 命中率（0-1）
            - size: 当前缓存条目数
            - capacity: 缓存容量
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        stats = {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "capacity": self.size
        }

        return stats

    def __len__(self) -> int:
        """返回当前缓存大小"""
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        """支持 'in' 操作符"""
        return key in self.cache

    def __repr__(self) -> str:
        """字符串表示"""
        stats = self.get_stats()
        return (
            f"DataCache(size={stats['size']}/{stats['capacity']}, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )
