#!/bin/bash
# Test wildcard consistency between full cache and on-demand modes

set -e

# Auto-detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPACT_PACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFYUI_DIR="$(cd "$IMPACT_PACK_DIR/../.." && pwd)"
CONFIG_FILE="$IMPACT_PACK_DIR/impact-pack.ini"
BACKUP_CONFIG="$IMPACT_PACK_DIR/impact-pack.ini.backup"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Wildcard Consistency Test"
echo "=========================================="
echo ""

# Backup original config
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$BACKUP_CONFIG"
    echo "✓ Backed up original config"
fi

# Function to kill ComfyUI
cleanup() {
    pkill -f "python.*main.py" 2>/dev/null || true
    sleep 2
}

# Function to test wildcard with specific config
test_with_config() {
    local MODE=$1
    local CACHE_LIMIT=$2

    echo ""
    echo "${BLUE}Testing $MODE mode (cache limit: ${CACHE_LIMIT}MB)${NC}"
    echo "----------------------------------------"

    # Update config
    cat > "$CONFIG_FILE" << EOF
[default]
dependency_version = 24
mmdet_skip = True
sam_editor_cpu = False
sam_editor_model = sam_vit_h_4b8939.pth
custom_wildcards = $IMPACT_PACK_DIR/custom_wildcards
disable_gpu_opencv = True
wildcard_cache_limit_mb = $CACHE_LIMIT
EOF

    # Start ComfyUI
    cleanup
    cd "$COMFYUI_DIR"
    bash run.sh --listen 127.0.0.1 --port 8190 > /tmp/comfyui_${MODE}.log 2>&1 &
    COMFYUI_PID=$!

    echo "  Waiting for server startup..."
    sleep 15

    # Check if server is running
    if ! curl -s http://127.0.0.1:8190/ > /dev/null; then
        echo "${RED}✗ Server failed to start${NC}"
        cat /tmp/comfyui_${MODE}.log | grep -i "wildcard\|error" | tail -20
        cleanup
        return 1
    fi

    # Check log for mode
    MODE_LOG=$(grep -i "wildcard.*mode" /tmp/comfyui_${MODE}.log | tail -1)
    echo "  $MODE_LOG"

    # Test 1: Simple wildcard
    echo ""
    echo "  Test 1: Simple wildcard substitution"
    RESULT1=$(curl -s http://127.0.0.1:8190/impact/wildcards \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": "__samples/flower__", "seed": 42}')

    TEXT1=$(echo "$RESULT1" | python3 -c "import sys, json; print(json.load(sys.stdin)['text'])")
    echo "    Input:  __samples/flower__"
    echo "    Output: $TEXT1"
    echo "    Result: $RESULT1" > /tmp/result_${MODE}_test1.json

    # Test 2: Dynamic prompt
    echo ""
    echo "  Test 2: Dynamic prompt"
    RESULT2=$(curl -s http://127.0.0.1:8190/impact/wildcards \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": "{red|blue|green} flower", "seed": 123}')

    TEXT2=$(echo "$RESULT2" | python3 -c "import sys, json; print(json.load(sys.stdin)['text'])")
    echo "    Input:  {red|blue|green} flower"
    echo "    Output: $TEXT2"
    echo "    Result: $RESULT2" > /tmp/result_${MODE}_test2.json

    # Test 3: Combined wildcard and dynamic prompt
    echo ""
    echo "  Test 3: Combined wildcard + dynamic prompt"
    RESULT3=$(curl -s http://127.0.0.1:8190/impact/wildcards \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": "beautiful {red|blue} __samples/flower__ with __samples/jewel__", "seed": 456}')

    TEXT3=$(echo "$RESULT3" | python3 -c "import sys, json; print(json.load(sys.stdin)['text'])")
    echo "    Input:  beautiful {red|blue} __samples/flower__ with __samples/jewel__"
    echo "    Output: $TEXT3"
    echo "    Result: $RESULT3" > /tmp/result_${MODE}_test3.json

    # Test 4: Transitive YAML wildcard
    echo ""
    echo "  Test 4: Transitive YAML wildcard (test.yaml)"
    RESULT4=$(curl -s http://127.0.0.1:8190/impact/wildcards \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": "__colors__", "seed": 222}')

    TEXT4=$(echo "$RESULT4" | python3 -c "import sys, json; print(json.load(sys.stdin)['text'])")
    echo "    Input:  __colors__ (transitive: __cold__|__warm__ -> blue|red|orange|yellow)"
    echo "    Output: $TEXT4"
    echo "    Expected: blue|red|orange|yellow"
    echo "    Result: $RESULT4" > /tmp/result_${MODE}_test4.json

    # Test 5: Wildcard list
    echo ""
    echo "  Test 5: Wildcard list API"
    LIST_RESULT=$(curl -s http://127.0.0.1:8190/impact/wildcards/list)
    LIST_COUNT=$(echo "$LIST_RESULT" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))")
    echo "    Wildcards found: $LIST_COUNT"
    echo "    Sample: $(echo "$LIST_RESULT" | python3 -c "import sys, json; print(', '.join(json.load(sys.stdin)['data'][:5]))")"
    echo "    Result: $LIST_RESULT" > /tmp/result_${MODE}_list.json

    # Stop server
    cleanup

    echo ""
    echo "${GREEN}✓ $MODE mode tests completed${NC}"
}

# Run tests
echo ""
echo "Starting consistency tests..."

# Test full cache mode
test_with_config "full_cache" 50

# Test on-demand mode
test_with_config "on_demand" 1

# Compare results
echo ""
echo "=========================================="
echo "Comparing Results"
echo "=========================================="

echo ""
echo "Test 1: Simple wildcard"
DIFF1=$(diff /tmp/result_full_cache_test1.json /tmp/result_on_demand_test1.json || true)
if [ -z "$DIFF1" ]; then
    echo "${GREEN}✓ Results match${NC}"
else
    echo "${RED}✗ Results differ${NC}"
    echo "$DIFF1"
fi

echo ""
echo "Test 2: Dynamic prompt"
DIFF2=$(diff /tmp/result_full_cache_test2.json /tmp/result_on_demand_test2.json || true)
if [ -z "$DIFF2" ]; then
    echo "${GREEN}✓ Results match${NC}"
else
    echo "${RED}✗ Results differ${NC}"
    echo "$DIFF2"
fi

echo ""
echo "Test 3: Combined wildcard + dynamic prompt"
DIFF3=$(diff /tmp/result_full_cache_test3.json /tmp/result_on_demand_test3.json || true)
if [ -z "$DIFF3" ]; then
    echo "${GREEN}✓ Results match${NC}"
else
    echo "${RED}✗ Results differ${NC}"
    echo "$DIFF3"
fi

echo ""
echo "Test 4: Transitive YAML wildcard"
DIFF4=$(diff /tmp/result_full_cache_test4.json /tmp/result_on_demand_test4.json || true)
if [ -z "$DIFF4" ]; then
    echo "${GREEN}✓ Results match${NC}"
else
    echo "${RED}✗ Results differ${NC}"
    echo "$DIFF4"
fi

echo ""
echo "Test 5: Wildcard list"
DIFF_LIST=$(diff /tmp/result_full_cache_list.json /tmp/result_on_demand_list.json || true)
if [ -z "$DIFF_LIST" ]; then
    echo "${GREEN}✓ Wildcard lists match${NC}"
else
    echo "${RED}✗ Wildcard lists differ${NC}"
    echo "$DIFF_LIST"
fi

# Restore original config
if [ -f "$BACKUP_CONFIG" ]; then
    mv "$BACKUP_CONFIG" "$CONFIG_FILE"
    echo ""
    echo "✓ Restored original config"
fi

# Final cleanup
cleanup

echo ""
echo "=========================================="
echo "Consistency Test Complete"
echo "=========================================="
