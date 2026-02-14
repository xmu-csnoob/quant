"""
Pydantic schemas for API responses.
API响应数据模型
"""

from typing import Generic, TypeVar, Optional, List
from pydantic import BaseModel, Field

T = TypeVar('T')


class ApiResponse(BaseModel, Generic[T]):
    """通用API响应"""
    code: int = Field(default=200, description="响应状态码")
    message: str = Field(default="success", description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")


# 别名（兼容不同命名风格）
APIResponse = ApiResponse


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应"""
    items: List[T] = Field(default_factory=list, description="数据列表")
    total: int = Field(default=0, description="总数")
    page: int = Field(default=1, description="当前页")
    page_size: int = Field(default=20, description="每页数量")
