"""
FastAPI Main Application
A股量化交易系统 - Web API服务
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import account, trading, strategy, backtest, data, risk, ml, auth, auto_trading


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print("🚀 A股量化交易系统 API 启动中...")
    yield
    # 关闭时执行
    print("👋 A股量化交易系统 API 关闭中...")


# 创建FastAPI应用
app = FastAPI(
    title="A股量化交易系统",
    description="专业A股量化交易系统API，支持策略回测、实盘交易、风险管理",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS - 允许前端跨域访问
# 注意：当 allow_credentials=True 时，不能使用 "*" 作为 allow_origins
import os

# 根据环境决定CORS配置
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite开发服务器
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# 生产环境可以从环境变量读取额外的允许来源
if os.getenv("PRODUCTION"):
    extra_origins = os.getenv("CORS_ORIGINS", "")
    if extra_origins:
        ALLOWED_ORIGINS.extend(extra_origins.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router)  # 认证路由（无需认证）
app.include_router(account.router)
app.include_router(trading.router)
app.include_router(strategy.router)
app.include_router(backtest.router)
app.include_router(data.router)
app.include_router(risk.router)
app.include_router(ml.router)
app.include_router(auto_trading.router)  # 自动交易


@app.get("/", tags=["根路径"])
async def root():
    """API根路径"""
    return {
        "message": "A股量化交易系统 API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
