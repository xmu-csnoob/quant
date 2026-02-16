"""
API身份认证和授权模块

功能：
- JWT token认证
- 用户登录/注册
- API权限控制
- 限流机制
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel, EmailStr
import os
from loguru import logger


# ==================== 配置 ====================

# JWT配置
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24小时

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# ==================== 模型 ====================

class Token(BaseModel):
    """Token响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token数据"""
    username: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    """用户模型"""
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: list[str] = ["read"]


class UserInDB(User):
    """数据库中的用户"""
    hashed_password: str


class UserCreate(BaseModel):
    """用户注册"""
    username: str
    password: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None


# ==================== 密码工具 ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


# ==================== 模拟用户数据库 ====================

# 注意：生产环境应使用真实数据库
# 以下为测试用模拟密码，格式: testpass_<role>
_TEST_ADMIN_PASSWORD = "testpass_admin"
_TEST_TRADER_PASSWORD = "testpass_trader"
_TEST_VIEWER_PASSWORD = "testpass_viewer"

fake_users_db: dict[str, UserInDB] = {}

def _init_fake_users_db():
    """延迟初始化模拟用户数据库"""
    if fake_users_db:
        return  # 已初始化

    # 在初始化时动态生成密码哈希
    fake_users_db.update({
        "admin": UserInDB(
            username="admin",
            email="admin@example.com",
            full_name="系统管理员",
            hashed_password=get_password_hash(_TEST_ADMIN_PASSWORD),
            disabled=False,
            scopes=["read", "write", "admin"],
        ),
        "trader": UserInDB(
            username="trader",
            email="trader@example.com",
            full_name="交易员",
            hashed_password=get_password_hash(_TEST_TRADER_PASSWORD),
            disabled=False,
            scopes=["read", "write"],
        ),
        "viewer": UserInDB(
            username="viewer",
            email="viewer@example.com",
            full_name="观察者",
            hashed_password=get_password_hash(_TEST_VIEWER_PASSWORD),
            disabled=False,
            scopes=["read"],
        ),
    })

# 模块加载时初始化
_init_fake_users_db()


# ==================== 用户工具 ====================

def get_user(db: dict, username: str) -> Optional[UserInDB]:
    """获取用户"""
    if username in db:
        user = db[username]
        # 支持UserInDB对象或字典
        if isinstance(user, UserInDB):
            return user
        return UserInDB(**user)
    return None


def authenticate_user(db: dict, username: str, password: str) -> Optional[UserInDB]:
    """验证用户"""
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(username: str, password: str, email: Optional[str] = None, full_name: Optional[str] = None) -> UserInDB:
    """创建用户"""
    if username in fake_users_db:
        raise ValueError(f"用户 {username} 已存在")

    user = UserInDB(
        username=username,
        email=email,
        full_name=full_name,
        hashed_password=get_password_hash(password),
        disabled=False,
        scopes=["read"],
    )
    fake_users_db[username] = user
    return user


# ==================== JWT工具 ====================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """解码令牌"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        scopes: list = payload.get("scopes", [])
        if username is None:
            return None
        return TokenData(username=username, scopes=scopes)
    except JWTError:
        return None


# ==================== 依赖注入 ====================

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[User]:
    """获取当前用户（可选认证）"""
    # 支持两种认证方式
    token_str = token or (credentials.credentials if credentials else None)

    if not token_str:
        return None

    token_data = decode_token(token_str)
    if token_data is None:
        return None

    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        return None

    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        scopes=user.scopes,
    )


async def get_current_user_required(
    current_user: Optional[User] = Depends(get_current_user),
) -> User:
    """获取当前用户（必须认证）"""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未认证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


async def get_current_active_user(
    current_user: User = Depends(get_current_user_required),
) -> User:
    """获取当前活跃用户"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="用户已禁用")
    return current_user


def require_scopes(*required_scopes: str):
    """
    权限检查装饰器

    用法：
        @router.get("/admin")
        async def admin_endpoint(user: User = Depends(require_scopes("admin"))):
            return {"message": "admin only"}
    """
    async def scope_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        for scope in required_scopes:
            if scope not in current_user.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"权限不足: 需要 {scope} 权限",
                )
        return current_user

    return scope_checker


# ==================== 限流 ====================

from collections import defaultdict
from threading import Lock
import time


class RateLimiter:
    """简单的内存限流器"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        with self.lock:
            now = time.time()
            # 清理过期记录
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if now - t < 60
            ]

            if len(self.requests[client_id]) >= self.requests_per_minute:
                return False

            self.requests[client_id].append(now)
            return True


# 全局限流器
rate_limiter = RateLimiter(requests_per_minute=100)


async def check_rate_limit(
    current_user: Optional[User] = Depends(get_current_user),
):
    """检查限流"""
    client_id = current_user.username if current_user else "anonymous"

    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试",
        )

    return True
