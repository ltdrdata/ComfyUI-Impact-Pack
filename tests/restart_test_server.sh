#!/bin/bash
# restart_test_server.sh
# ComfyUI 서버를 빠르게 재시작하는 유틸리티 스크립트
# Usage: bash restart_test_server.sh [PORT]

PORT=${1:-8188}  # 기본 포트 8188
COMFYUI_DIR="/mnt/teratera/git/ComfyUI"
LOG_FILE="/tmp/comfyui_test_${PORT}.log"

echo "=========================================="
echo "ComfyUI Test Server Restart Utility"
echo "=========================================="
echo "Port: $PORT"
echo "Log: $LOG_FILE"
echo ""

# 1. 기존 서버 종료
echo "🛑 Stopping existing server..."
pkill -f "python.*main.py"
sleep 2

# 프로세스 종료 확인
if pgrep -f "python.*main.py" > /dev/null; then
    echo "⚠️  Warning: Some processes still running"
    ps aux | grep main.py | grep -v grep
    echo "Forcing kill..."
    pkill -9 -f "python.*main.py"
    sleep 1
fi
echo "✅ Server stopped"

# 2. 서버 시작
echo ""
echo "🚀 Starting server on port $PORT..."
cd "$COMFYUI_DIR" || {
    echo "❌ Error: Cannot access $COMFYUI_DIR"
    exit 1
}

# 백그라운드로 서버 시작
bash run.sh --listen 127.0.0.1 --port "$PORT" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo ""

# 3. 서버 준비 대기
echo "⏳ Waiting for server startup..."
for i in {1..30}; do
    sleep 1
    if curl -s http://127.0.0.1:$PORT/ > /dev/null 2>&1; then
        echo ""
        echo "✅ Server ready on port $PORT (${i}s)"
        echo "📝 Log: $LOG_FILE"
        echo "🔗 URL: http://127.0.0.1:$PORT"
        echo ""
        echo "Test endpoints:"
        echo "  curl http://127.0.0.1:$PORT/impact/wildcards/list"
        echo "  curl http://127.0.0.1:$PORT/impact/wildcards/list/loaded"
        exit 0
    fi
    echo -n "."
done

# 타임아웃
echo ""
echo "❌ Server failed to start within 30 seconds"
echo "📝 Check log: $LOG_FILE"
echo ""
echo "Last 20 lines of log:"
tail -20 "$LOG_FILE"
exit 1
