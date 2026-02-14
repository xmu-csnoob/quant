"""
认证模块
"""

from src.api.auth.auth import (
    # 模型
    Token,
    TokenData,
    User,
    UserInDB,
    UserCreate,
    # 用户工具
    get_user,
    authenticate_user,
    create_user,
    # JWT工具
    create_access_token,
    decode_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    # 依赖
    get_current_user,
    get_current_user_required,
    get_current_active_user,
    require_scopes,
    # 限流
    rate_limiter,
    check_rate_limit,
    # 数据库
    fake_users_db,
)

__all__ = [
    # 模型
    "Token",
    "TokenData",
    "User",
    "UserInDB",
    "UserCreate",
    # 用户工具
    "get_user",
    "authenticate_user",
    "create_user",
    # JWT工具
    "create_access_token",
    "decode_token",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    # 依赖
    "get_current_user",
    "get_current_user_required",
    "get_current_active_user",
    "require_scopes",
    # 限流
    "rate_limiter",
    "check_rate_limit",
    # 数据库
    "fake_users_db",
]
