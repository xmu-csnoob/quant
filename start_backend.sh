#!/bin/bash
# 启动A股量化交易系统后端API服务

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🚀 启动A股量化交易系统 API..."
echo "📍 API文档: http://localhost:8000/docs"
echo "📍 ReDoc文档: http://localhost:8000/redoc"
echo ""

uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
