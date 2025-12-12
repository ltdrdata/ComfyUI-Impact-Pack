#!/bin/bash
# Progressive On-Demand Wildcard Loading Test
# Verifies that wildcards are loaded progressively as they are accessed

set -e

# Auto-detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPACT_PACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFYUI_DIR="$(cd "$IMPACT_PACK_DIR/../.." && pwd)"
CONFIG_FILE="$IMPACT_PACK_DIR/impact-pack.ini"
BACKUP_CONFIG="$IMPACT_PACK_DIR/impact-pack.ini.backup"
PORT=8195

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "=========================================="
echo "Progressive On-Demand Loading Test"
echo "=========================================="
echo ""
echo "This test verifies that /wildcards/list/loaded"
echo "increases progressively as wildcards are accessed."
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
    pkill -f "python.*main.py.*$PORT" 2>/dev/null || true
    sleep 2
}

# Setup on-demand mode (low cache limit)
echo "${BLUE}Setting up on-demand mode configuration${NC}"
cat > "$CONFIG_FILE" << EOF
[default]
dependency_version = 24
mmdet_skip = True
sam_editor_cpu = False
sam_editor_model = sam_vit_h_4b8939.pth
custom_wildcards = $IMPACT_PACK_DIR/custom_wildcards
disable_gpu_opencv = True
wildcard_cache_limit_mb = 0.5
EOF

echo "✓ Configuration: on-demand mode (0.5MB limit)"
echo ""

# Start server
cleanup
cd "$COMFYUI_DIR"
echo "Starting ComfyUI server on port $PORT..."
bash run.sh --listen 127.0.0.1 --port $PORT > /tmp/progressive_test.log 2>&1 &
COMFYUI_PID=$!

echo "Waiting for server startup..."
sleep 15

# Check server
if ! curl -s http://127.0.0.1:$PORT/ > /dev/null; then
    echo "${RED}✗ Server failed to start${NC}"
    cat /tmp/progressive_test.log | grep -i "wildcard\|error" | tail -20
    exit 1
fi

echo "${GREEN}✓ Server started${NC}"
echo ""

# Check loading mode from log
MODE_LOG=$(grep -i "wildcard.*mode" /tmp/progressive_test.log | tail -1)
echo "${YELLOW}$MODE_LOG${NC}"
echo ""

# Test Progressive Loading
echo "=========================================="
echo "Progressive Loading Verification"
echo "=========================================="
echo ""

# Step 1: Initial state (no wildcards accessed)
echo "${CYAN}Step 1: Initial state (before any wildcard access)${NC}"
RESPONSE=$(curl -s http://127.0.0.1:$PORT/impact/wildcards/list/loaded)
LOADED_COUNT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))" 2>/dev/null || echo "0")
ON_DEMAND=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('on_demand_mode', False))" 2>/dev/null || echo "false")
TOTAL_AVAILABLE=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_available', 0))" 2>/dev/null || echo "0")

echo "  On-demand mode: $ON_DEMAND"
echo "  Total available wildcards: $TOTAL_AVAILABLE"
echo "  Loaded wildcards: ${YELLOW}$LOADED_COUNT${NC}"

if [ "$ON_DEMAND" != "True" ]; then
    echo "${RED}✗ FAIL: On-demand mode not active!${NC}"
    exit 1
fi

if [ "$LOADED_COUNT" -ne 0 ]; then
    echo "${YELLOW}⚠ WARNING: Expected 0 loaded, got $LOADED_COUNT${NC}"
fi
echo ""

# Step 2: Access first wildcard
echo "${CYAN}Step 2: Access first wildcard (__samples/flower__)${NC}"
RESULT1=$(curl -s http://127.0.0.1:$PORT/impact/wildcards \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"text": "__samples/flower__", "seed": 42}')
TEXT1=$(echo "$RESULT1" | python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))")
echo "  Result: $TEXT1"

RESPONSE=$(curl -s http://127.0.0.1:$PORT/impact/wildcards/list/loaded)
LOADED_COUNT_1=$(echo "$RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))")
echo "  Loaded wildcards: ${YELLOW}$LOADED_COUNT_1${NC}"

if [ "$LOADED_COUNT_1" -lt 1 ]; then
    echo "${RED}✗ FAIL: Expected at least 1 loaded wildcard${NC}"
    exit 1
fi
echo "${GREEN}✓ PASS: Wildcard count increased${NC}"
echo ""

# Step 3: Access second wildcard (different from first)
echo "${CYAN}Step 3: Access second wildcard (__dragon__)${NC}"
RESULT2=$(curl -s http://127.0.0.1:$PORT/impact/wildcards \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"text": "__dragon__", "seed": 200}')
TEXT2=$(echo "$RESULT2" | python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))")
echo "  Result: $TEXT2"

RESPONSE=$(curl -s http://127.0.0.1:$PORT/impact/wildcards/list/loaded)
LOADED_COUNT_2=$(echo "$RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))")
echo "  Loaded wildcards: ${YELLOW}$LOADED_COUNT_2${NC}"

if [ "$LOADED_COUNT_2" -le "$LOADED_COUNT_1" ]; then
    echo "${RED}✗ FAIL: Expected loaded count to increase (was $LOADED_COUNT_1, now $LOADED_COUNT_2)${NC}"
    exit 1
fi
echo "${GREEN}✓ PASS: Wildcard count increased progressively${NC}"
echo ""

# Step 4: Access third wildcard (YAML)
echo "${CYAN}Step 4: Access third wildcard (__colors__)${NC}"
RESULT3=$(curl -s http://127.0.0.1:$PORT/impact/wildcards \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"text": "__colors__", "seed": 333}')
TEXT3=$(echo "$RESULT3" | python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))")
echo "  Result: $TEXT3"

RESPONSE=$(curl -s http://127.0.0.1:$PORT/impact/wildcards/list/loaded)
LOADED_COUNT_3=$(echo "$RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))")
LOADED_LIST=$(echo "$RESPONSE" | python3 -c "import sys, json; print(', '.join(json.load(sys.stdin)['data'][:10]))")
echo "  Loaded wildcards: ${YELLOW}$LOADED_COUNT_3${NC}"
echo "  Sample loaded: $LOADED_LIST"

if [ "$LOADED_COUNT_3" -le "$LOADED_COUNT_2" ]; then
    echo "${RED}✗ FAIL: Expected loaded count to increase (was $LOADED_COUNT_2, now $LOADED_COUNT_3)${NC}"
    exit 1
fi
echo "${GREEN}✓ PASS: Wildcard count increased progressively${NC}"
echo ""

# Step 5: Re-access first wildcard (should not increase count)
echo "${CYAN}Step 5: Re-access first wildcard (cached)${NC}"
RESULT4=$(curl -s http://127.0.0.1:$PORT/impact/wildcards \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"text": "__samples/flower__", "seed": 42}')

RESPONSE=$(curl -s http://127.0.0.1:$PORT/impact/wildcards/list/loaded)
LOADED_COUNT_4=$(echo "$RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))")
echo "  Loaded wildcards: ${YELLOW}$LOADED_COUNT_4${NC}"

if [ "$LOADED_COUNT_4" -ne "$LOADED_COUNT_3" ]; then
    echo "${YELLOW}⚠ WARNING: Count changed on cache access (was $LOADED_COUNT_3, now $LOADED_COUNT_4)${NC}"
else
    echo "${GREEN}✓ PASS: Cached access did not change count${NC}"
fi
echo ""

# Step 6: Deep transitive wildcard (should load multiple wildcards)
echo "${CYAN}Step 6: Deep transitive wildcard (__adnd__)${NC}"
RESULT5=$(curl -s http://127.0.0.1:$PORT/impact/wildcards \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"text": "__adnd__ creature", "seed": 222}')
TEXT5=$(echo "$RESULT5" | python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))")
echo "  Result: $TEXT5"

RESPONSE=$(curl -s http://127.0.0.1:$PORT/impact/wildcards/list/loaded)
LOADED_COUNT_5=$(echo "$RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))")
echo "  Loaded wildcards: ${YELLOW}$LOADED_COUNT_5${NC}"

if [ "$LOADED_COUNT_5" -le "$LOADED_COUNT_4" ]; then
    echo "${YELLOW}⚠ Transitive wildcards may already be loaded${NC}"
else
    echo "${GREEN}✓ PASS: Transitive wildcards loaded progressively${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Progressive Loading Summary"
echo "=========================================="
echo ""
echo "Total available wildcards: $TOTAL_AVAILABLE"
echo "Loading progression:"
echo "  Initial:       $LOADED_COUNT"
echo "  After step 2:  $LOADED_COUNT_1 (+$(($LOADED_COUNT_1 - $LOADED_COUNT)))"
echo "  After step 3:  $LOADED_COUNT_2 (+$(($LOADED_COUNT_2 - $LOADED_COUNT_1)))"
echo "  After step 4:  $LOADED_COUNT_3 (+$(($LOADED_COUNT_3 - $LOADED_COUNT_2)))"
echo "  After step 5:  $LOADED_COUNT_4 (cache, no change)"
echo "  After step 6:  $LOADED_COUNT_5 (+$(($LOADED_COUNT_5 - $LOADED_COUNT_4)))"
echo ""

# Validation
ALL_PASSED=true

if [ "$LOADED_COUNT_1" -le "$LOADED_COUNT" ]; then
    echo "${RED}✗ FAIL: Step 2 did not increase count${NC}"
    ALL_PASSED=false
fi

if [ "$LOADED_COUNT_2" -le "$LOADED_COUNT_1" ]; then
    echo "${RED}✗ FAIL: Step 3 did not increase count${NC}"
    ALL_PASSED=false
fi

if [ "$LOADED_COUNT_3" -le "$LOADED_COUNT_2" ]; then
    echo "${RED}✗ FAIL: Step 4 did not increase count${NC}"
    ALL_PASSED=false
fi

if [ "$ALL_PASSED" = true ]; then
    echo "${GREEN}🎉 ALL TESTS PASSED${NC}"
    echo "${GREEN}Progressive on-demand loading verified successfully!${NC}"
    EXIT_CODE=0
else
    echo "${RED}❌ TESTS FAILED${NC}"
    echo "${RED}Progressive loading did not work as expected!${NC}"
    EXIT_CODE=1
fi
echo ""

# Restore config
cleanup
if [ -f "$BACKUP_CONFIG" ]; then
    mv "$BACKUP_CONFIG" "$CONFIG_FILE"
    echo "✓ Restored original config"
fi

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo "Log saved to: /tmp/progressive_test.log"
echo ""

exit $EXIT_CODE
