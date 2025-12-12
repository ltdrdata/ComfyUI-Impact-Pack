#!/bin/bash
# Verify wildcard lazy loading through ComfyUI API

set -e

# Auto-detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPACT_PACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFYUI_DIR="$(cd "$IMPACT_PACK_DIR/../.." && pwd)"
CONFIG_FILE="$IMPACT_PACK_DIR/impact-pack.ini"
BACKUP_CONFIG="$IMPACT_PACK_DIR/impact-pack.ini.backup"

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Wildcard Lazy Load Verification Test"
echo "=========================================="
echo ""
echo "This test verifies that on-demand loading produces"
echo "identical results to full cache mode."
echo ""

# Backup original config
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$BACKUP_CONFIG"
    echo "✓ Backed up original config"
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -f "python.*main.py" 2>/dev/null || true
    sleep 2
}

# Test with specific configuration
test_mode() {
    local MODE=$1
    local CACHE_LIMIT=$2
    local PORT=$3

    echo ""
    echo "${BLUE}=========================================${NC}"
    echo "${BLUE}Testing: $MODE (limit: ${CACHE_LIMIT}MB, port: $PORT)${NC}"
    echo "${BLUE}=========================================${NC}"

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

    # Start server
    cleanup
    cd "$COMFYUI_DIR"
    bash run.sh --listen 127.0.0.1 --port $PORT > /tmp/comfyui_${MODE}.log 2>&1 &
    COMFYUI_PID=$!

    echo "Waiting for server startup..."
    sleep 15

    # Check server
    if ! curl -s http://127.0.0.1:$PORT/ > /dev/null; then
        echo "${RED}✗ Server failed to start${NC}"
        cat /tmp/comfyui_${MODE}.log | grep -i "wildcard\|error" | tail -20
        return 1
    fi

    # Get loading mode from log
    MODE_LOG=$(grep -i "wildcard.*mode" /tmp/comfyui_${MODE}.log | tail -1)
    echo "${YELLOW}$MODE_LOG${NC}"
    echo ""

    # Test 1: Get wildcard list (BEFORE any access in on-demand mode)
    echo "📋 Test 1: Get wildcard list"
    LIST_RESULT=$(curl -s http://127.0.0.1:$PORT/impact/wildcards/list)
    LIST_COUNT=$(echo "$LIST_RESULT" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))")
    echo "   Total wildcards: $LIST_COUNT"
    echo "   Sample: $(echo "$LIST_RESULT" | python3 -c "import sys, json; print(', '.join(json.load(sys.stdin)['data'][:10]))")"
    echo "$LIST_RESULT" > /tmp/result_${MODE}_list.json
    echo ""

    # Test 2: Simple wildcard
    echo "📋 Test 2: Simple wildcard"
    RESULT1=$(curl -s http://127.0.0.1:$PORT/impact/wildcards \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": "__samples/flower__", "seed": 42}')
    TEXT1=$(echo "$RESULT1" | python3 -c "import sys, json; print(json.load(sys.stdin)['text'])")
    echo "   Input:  __samples/flower__"
    echo "   Output: $TEXT1"
    echo "$RESULT1" > /tmp/result_${MODE}_simple.json
    echo ""

    # Test 3: Depth 3 transitive (adnd → dragon → dragon_spirit)
    echo "📋 Test 3: Depth 3 transitive (TXT → TXT → TXT)"
    RESULT2=$(curl -s http://127.0.0.1:$PORT/impact/wildcards \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": "__adnd__ creature", "seed": 222}')
    TEXT2=$(echo "$RESULT2" | python3 -c "import sys, json; print(json.load(sys.stdin)['text'])")
    echo "   Input:  __adnd__ creature (depth 3: adnd → dragon → dragon_spirit)"
    echo "   Output: $TEXT2"
    echo "$RESULT2" > /tmp/result_${MODE}_depth3.json
    echo ""

    # Test 4: YAML transitive (colors → cold/warm → blue/red/orange/yellow)
    echo "📋 Test 4: YAML transitive"
    RESULT3=$(curl -s http://127.0.0.1:$PORT/impact/wildcards \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": "__colors__", "seed": 333}')
    TEXT3=$(echo "$RESULT3" | python3 -c "import sys, json; print(json.load(sys.stdin)['text'])")
    echo "   Input:  __colors__ (YAML: colors → cold|warm → blue|red|orange|yellow)"
    echo "   Output: $TEXT3"
    echo "$RESULT3" > /tmp/result_${MODE}_yaml.json
    echo ""

    # Test 5: Get wildcard list AGAIN (AFTER access in on-demand mode)
    echo "📋 Test 5: Get wildcard list (after access)"
    LIST_RESULT2=$(curl -s http://127.0.0.1:$PORT/impact/wildcards/list)
    LIST_COUNT2=$(echo "$LIST_RESULT2" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))")
    echo "   Total wildcards: $LIST_COUNT2"
    echo "$LIST_RESULT2" > /tmp/result_${MODE}_list_after.json
    echo ""

    # Compare before/after list
    if [ "$MODE" = "on_demand" ]; then
        if [ "$LIST_COUNT" -eq "$LIST_COUNT2" ]; then
            echo "${GREEN}✓ Wildcard list unchanged after access (${LIST_COUNT} = ${LIST_COUNT2})${NC}"
        else
            echo "${RED}✗ Wildcard list changed after access (${LIST_COUNT} != ${LIST_COUNT2})${NC}"
        fi
        echo ""
    fi

    cleanup

    echo "${GREEN}✓ $MODE tests completed${NC}"
    echo ""
}

# Run tests
test_mode "full_cache" 100 8190
test_mode "on_demand" 1 8191

# Compare results
echo ""
echo "=========================================="
echo "COMPARISON RESULTS"
echo "=========================================="
echo ""

compare_test() {
    local TEST_NAME=$1
    local FILE_SUFFIX=$2

    echo "Test: $TEST_NAME"
    DIFF=$(diff /tmp/result_full_cache_${FILE_SUFFIX}.json /tmp/result_on_demand_${FILE_SUFFIX}.json || true)
    if [ -z "$DIFF" ]; then
        echo "${GREEN}✓ Results MATCH${NC}"
    else
        echo "${RED}✗ Results DIFFER${NC}"
        echo "Difference:"
        echo "$DIFF" | head -10
    fi
    echo ""
}

compare_test "Wildcard List (before access)" "list"
compare_test "Simple Wildcard" "simple"
compare_test "Depth 3 Transitive" "depth3"
compare_test "YAML Transitive" "yaml"
compare_test "Wildcard List (after access)" "list_after"

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""

ALL_MATCH=true
for suffix in list simple depth3 yaml list_after; do
    if ! diff /tmp/result_full_cache_${suffix}.json /tmp/result_on_demand_${suffix}.json > /dev/null 2>&1; then
        ALL_MATCH=false
        break
    fi
done

if [ "$ALL_MATCH" = true ]; then
    echo "${GREEN}🎉 ALL TESTS PASSED${NC}"
    echo "${GREEN}On-demand loading produces IDENTICAL results to full cache mode!${NC}"
    EXIT_CODE=0
else
    echo "${RED}❌ TESTS FAILED${NC}"
    echo "${RED}On-demand loading has consistency issues!${NC}"
    EXIT_CODE=1
fi
echo ""

# Restore config
if [ -f "$BACKUP_CONFIG" ]; then
    mv "$BACKUP_CONFIG" "$CONFIG_FILE"
    echo "✓ Restored original config"
fi

cleanup

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="

exit $EXIT_CODE
