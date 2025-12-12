#!/bin/bash
# Verify that on-demand mode is actually triggered with 0.5MB limit

# Auto-detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$IMPACT_DIR/impact-pack.ini"

echo "=========================================="
echo "Verify On-Demand Mode Activation"
echo "=========================================="
echo ""

# Set config to 0.5MB limit
cat > "$CONFIG_FILE" << EOF
[default]
dependency_version = 24
mmdet_skip = True
sam_editor_cpu = False
sam_editor_model = sam_vit_h_4b8939.pth
custom_wildcards = $IMPACT_DIR/custom_wildcards
disable_gpu_opencv = True
wildcard_cache_limit_mb = 0.5
EOF

echo "Config set to 0.5MB cache limit"
echo ""

# Kill any existing servers
pkill -9 -f "python.*main.py" 2>/dev/null || true
sleep 3

# Start server
COMFYUI_DIR="$(cd "$IMPACT_DIR/../.." && pwd)"
cd "$COMFYUI_DIR"
echo "Starting ComfyUI server on port 8190..."
bash run.sh --listen 127.0.0.1 --port 8190 > /tmp/verify_ondemand.log 2>&1 &
SERVER_PID=$!

# Wait for server
echo "Waiting 70 seconds for server startup..."
for i in {1..70}; do
    sleep 1
    if [ $((i % 10)) -eq 0 ]; then
        echo "  ... $i seconds"
    fi
done

# Check server
if ! curl -s http://127.0.0.1:8190/ > /dev/null; then
    echo "✗ Server failed to start"
    cat /tmp/verify_ondemand.log
    exit 1
fi

echo "✓ Server started"
echo ""

# Check loading mode
echo "Loading mode detected:"
grep -i "wildcard.*mode\|wildcard.*size.*cache" /tmp/verify_ondemand.log | grep -v "Maximum depth"
echo ""

# Verify mode
if grep -q "Using on-demand loading mode" /tmp/verify_ondemand.log; then
    echo "✅ SUCCESS: On-demand mode activated with 0.5MB limit!"
elif grep -q "Using full cache mode" /tmp/verify_ondemand.log; then
    echo "❌ FAIL: Full cache mode used (should be on-demand)"
    echo ""
    echo "Cache limit in log:"
    grep "cache limit" /tmp/verify_ondemand.log
else
    echo "⚠️  WARNING: Could not determine mode"
fi

# Test wildcard functionality
echo ""
echo "Testing wildcard functionality in on-demand mode..."
curl -s -X POST http://127.0.0.1:8190/impact/wildcards \
    -H "Content-Type: application/json" \
    -d '{"text": "__adnd__ creature", "seed": 222}' > /tmp/verify_result.json

RESULT=$(cat /tmp/verify_result.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))" 2>/dev/null || echo "ERROR")
echo "  Depth 3 transitive (seed=222): $RESULT"

if [ "$RESULT" = "Shrewd Hatchling creature" ]; then
    echo "  ✅ Transitive wildcard works correctly"
else
    echo "  ❌ Unexpected result: $RESULT"
fi

# Stop server
kill $SERVER_PID 2>/dev/null
pkill -9 -f "python.*main.py.*8190" 2>/dev/null

echo ""
echo "Full log saved to: /tmp/verify_ondemand.log"
