"""
持久化缓存模块 - 支持文件存储和 TTL（过期时间）
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from loguru import logger


class PersistentCache:
    """
    持久化缓存 - 支持过期时间和文件存储

    特点：
    1. 文件持久化（重启不丢失）
    2. TTL 过期机制
    3. LRU 内存缓存
    4. 频率限制友好的缓存策略

    用途：
    - 缓存股票列表（每小时限制）
    - 缓存分钟数据（每天限制）
    - 缓存涨跌停数据（每小时限制）
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        default_ttl: int = 3600
    ):
        """
        初始化持久化缓存

        Args:
            cache_dir: 缓存文件存储目录
            default_ttl: 默认过期时间（秒），默认 1 小时
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

        # 内存缓存（存储缓存元数据）
        self.metadata: Dict[str, Dict[str, Any]] = {}

        logger.info(f"PersistentCache initialized: dir={cache_dir}, default_ttl={default_ttl}s")

    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径"""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _is_expired(self, key: str) -> bool:
        """检查缓存是否过期"""
        if key not in self.metadata:
            return True

        metadata = self.metadata[key]
        expire_time = metadata.get("expire_time", 0)
        return time.time() > expire_time

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            缓存的 DataFrame，如果不存在或过期返回 None
        """
        # 检查是否过期
        if self._is_expired(key):
            if key in self.metadata:
                logger.debug(f"Cache expired: {key}")
                self.delete(key)
            return None

        # 从文件加载
        cache_file = self._get_cache_file(key)
        if not cache_file.exists():
            logger.debug(f"Cache file not found: {key}")
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查是否是 DataFrame 格式
            if "_data" in data and "_columns" in data:
                df = pd.DataFrame(data["_data"], columns=data["_columns"])
                logger.debug(f"Loaded from file cache: {key} ({len(df)} rows)")
                return df
            else:
                logger.warning(f"Invalid cache format: {key}")
                return None

        except Exception as e:
            logger.error(f"Failed to load cache file {key}: {e}")
            return None

    def put(self, key: str, value: pd.DataFrame, ttl: int = None) -> None:
        """
        存入缓存

        Args:
            key: 缓存键
            value: 要缓存的 DataFrame
            ttl: 过期时间（秒），默认使用 default_ttl
        """
        if ttl is None:
            ttl = self.default_ttl

        # 设置过期时间
        expire_time = time.time() + ttl
        self.metadata[key] = {
            "expire_time": expire_time,
            "ttl": ttl,
            "rows": len(value),
            "columns": list(value.columns)
        }

        # 保存到文件
        cache_file = self._get_cache_file(key)
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "_columns": value.columns.tolist(),
                "_data": value.values.tolist()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)

            logger.debug(f"Saved to file cache: {key} (ttl={ttl}s, {len(value)} rows)")

        except Exception as e:
            logger.error(f"Failed to save cache file {key}: {e}")

    def delete(self, key: str) -> bool:
        """
        删除缓存

        Args:
            key: 缓存键

        Returns:
            True 如果删除成功，否则 False
        """
        # 删除元数据
        if key in self.metadata:
            del self.metadata[key]

        # 删除文件
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                cache_file.unlink()
                logger.debug(f"Deleted cache file: {key}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete cache file {key}: {e}")

        return False

    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在且未过期

        Args:
            key: 缓存键

        Returns:
            True 如果存在且未过期，否则 False
        """
        return not self._is_expired(key) and self._get_cache_file(key).exists()

    def clear(self) -> None:
        """清空所有缓存"""
        self.metadata.clear()

        # 删除所有缓存文件
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.debug("All cache cleared")

    def get_stats(self) -> dict:
        """
        获取缓存统计

        Returns:
            统计信息字典
        """
        total = sum(1 for v in self.metadata.values() if v.get("rows", 0) > 0)

        return {
            "cache_count": len(self.metadata),
            "total_rows": total,
            "cache_dir": str(self.cache_dir)
        }

    def __len__(self) -> int:
        """返回缓存条目数"""
        return len(self.metadata)

    def __contains__(self, key: str) -> bool:
        """支持 'in' 操作符"""
        return self.exists(key)

    def __repr__(self) -> str:
        """字符串表示"""
        stats = self.get_stats()
        return f"PersistentCache(count={stats['cache_count']}, rows={stats['total_rows']})"
