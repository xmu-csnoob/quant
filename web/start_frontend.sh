#!/bin/bash
# 启动A股量化交易系统前端开发服务器

cd "$(dirname "$0")"

echo "🚀 启动前端开发服务器..."
echo "📍 访问地址: http://localhost:5173"
echo ""

npm run dev
