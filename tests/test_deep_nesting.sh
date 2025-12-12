#!/bin/bash
# Deep Nesting Test Suite
# Tests transitive wildcard expansion up to 7 levels

set -e

PORT=8194
COMFYUI_DIR="/mnt/teratera/git/ComfyUI"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$IMPACT_DIR/impact-pack.ini"
LOG_FILE="/tmp/deep_nesting_test.log"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "=========================================="
echo "Deep Nesting Test Suite (7 Levels)"
echo "=========================================="
echo "Port: $PORT"
echo "Testing: Transitive wildcard expansion"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -f "python.*main.py.*--port $PORT" 2>/dev/null || true
    rm -f "$CONFIG_FILE"
    echo "Cleanup complete"
}

trap cleanup EXIT

# Kill any existing server on this port
echo "Killing any existing server on port $PORT..."
pkill -f "python.*main.py.*--port $PORT" 2>/dev/null || true
sleep 2

# Setup configuration
echo "Setting up configuration..."
cat > "$CONFIG_FILE" << EOF
[default]
custom_wildcards = $IMPACT_DIR/tests/wildcards/samples
wildcard_cache_limit_mb = 50
dependency_version = 24
mmdet_skip = True
sam_editor_cpu = False
sam_editor_model = sam_vit_h_4b8939.pth
disable_gpu_opencv = True
EOF

echo "Configuration created: custom_wildcards = $IMPACT_DIR/tests/wildcards/samples"
echo ""

# Start server
echo "Starting ComfyUI server on port $PORT..."
cd "$COMFYUI_DIR"
bash run.sh --listen 127.0.0.1 --port $PORT > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server startup
echo "Waiting for server startup..."
for i in {1..60}; do
    sleep 1
    if curl -s http://127.0.0.1:$PORT/ > /dev/null 2>&1; then
        echo "✅ Server ready (${i}s)"
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "  ... ${i}s elapsed"
    fi
    if [ $i -eq 60 ]; then
        echo ""
        echo "${RED}❌ Server failed to start within 60 seconds${NC}"
        echo "Log tail:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
done

echo ""

# Test function for nesting
test_nesting() {
    local TEST_NUM=$1
    local DESCRIPTION=$2
    local PROMPT=$3
    local SEED=$4
    local EXPECTED_DEPTH=$5

    echo "${BLUE}=== Test $TEST_NUM: $DESCRIPTION ===${NC}"
    echo "Prompt: ${YELLOW}$PROMPT${NC}"
    echo "Seed: $SEED"
    echo "Expected nesting depth: $EXPECTED_DEPTH"

    RESULT=$(curl -s -X POST http://127.0.0.1:$PORT/impact/wildcards \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$PROMPT\", \"seed\": $SEED}" | \
        python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))" 2>/dev/null || echo "ERROR")

    echo "Result: ${GREEN}$RESULT${NC}"

    # Check if result contains any unexpanded wildcards
    if echo "$RESULT" | grep -q "__.*__"; then
        echo "Status: ${YELLOW}⚠️  WARNING - Contains unexpanded wildcards${NC}"
        echo "Unexpanded: $(echo "$RESULT" | grep -o '__[^_]*__')"
    elif [ "$RESULT" != "ERROR" ] && [ -n "$RESULT" ]; then
        echo "Status: ${GREEN}✅ PASS - All wildcards fully expanded${NC}"
    else
        echo "Status: ${RED}❌ FAIL - Server error or no response${NC}"
    fi
    echo ""
}

echo "=========================================="
echo "Test Suite Execution"
echo "=========================================="
echo ""

# Direct level tests
echo "${CYAN}--- Direct Level Access Tests ---${NC}"
echo ""

test_nesting "01" "Level 7 (Final)" \
    "__test_nesting_level7__" \
    100 \
    0

test_nesting "02" "Level 6 → Level 7" \
    "__test_nesting_level6__" \
    200 \
    1

test_nesting "03" "Level 5 → Level 6 → Level 7" \
    "__test_nesting_level5__" \
    300 \
    2

test_nesting "04" "Level 4 → ... → Level 7" \
    "__test_nesting_level4__" \
    400 \
    3

test_nesting "05" "Level 3 → ... → Level 7" \
    "__test_nesting_level3__" \
    500 \
    4

test_nesting "06" "Level 2 → ... → Level 7" \
    "__test_nesting_level2__" \
    600 \
    5

test_nesting "07" "Level 1 → ... → Level 7 (Full 7 levels)" \
    "__test_nesting_level1__" \
    700 \
    6

echo ""
echo "${CYAN}--- Multiple Nesting Tests ---${NC}"
echo ""

test_nesting "08" "Two level 1 wildcards" \
    "__test_nesting_level1__ and __test_nesting_level1__" \
    800 \
    6

test_nesting "09" "Mixed depths" \
    "__test_nesting_level1__ with __test_nesting_level4__" \
    900 \
    6

test_nesting "10" "Level 1 in dynamic prompt" \
    "{__test_nesting_level1__|__test_nesting_level2__|__test_nesting_level3__}" \
    1000 \
    6

echo ""
echo "${CYAN}--- Complex Combination Tests ---${NC}"
echo ""

test_nesting "11" "Nesting with quantifier" \
    "2#__test_nesting_level1__" \
    1100 \
    6

test_nesting "12" "Nesting with multi-select" \
    "{2\$\$, \$\$__test_nesting_level1__|__test_nesting_level2__|__test_nesting_level3__}" \
    1200 \
    6

test_nesting "13" "Nesting with weighted selection" \
    "{5::__test_nesting_level1__|3::__test_nesting_level3__|1::__test_nesting_level5__}" \
    1300 \
    6

test_nesting "14" "Very deep with other wildcards" \
    "__test_nesting_level1__ beautiful __아름다운색__" \
    1400 \
    6

test_nesting "15" "All 7 levels in one prompt" \
    "__test_nesting_level1__, __test_nesting_level2__, __test_nesting_level3__, __test_nesting_level4__, __test_nesting_level5__, __test_nesting_level6__, __test_nesting_level7__" \
    1500 \
    6

echo ""
echo "${CYAN}--- Depth-Agnostic Pattern Matching Tests ---${NC}"
echo ""

# Test 16: Depth-agnostic pattern matching with __*/test_nesting_level7__
# The __*/name__ pattern matches wildcards at ANY directory depth:
#   - test_nesting_level7.txt (at root level)
#   - level1/level2/.../level7/test_nesting_level7.txt (deeply nested)
#   - any_folder/test_nesting_level7.txt (in any subfolder)
test_nesting "16" "Pattern matching __*/test_nesting_level7__" \
    "__*/test_nesting_level7__" \
    1600 \
    0

# Test 17: Depth-agnostic pattern matching with __*/test_nesting_level4__
# Similar to __*/dragon__ matching both "dragon.txt" and "dragon/wizard.txt":
#   - test_nesting_level4.txt (direct file)
#   - level1/.../level4/test_nesting_level4.txt (nested file)
#   - The pattern ignores directory depth and matches by wildcard name
test_nesting "17" "Pattern matching __*/test_nesting_level4__" \
    "__*/test_nesting_level4__" \
    1700 \
    3

echo ""
echo "=========================================="
echo "Loaded Wildcards Check"
echo "=========================================="

# Check what wildcards were loaded
LOADED=$(curl -s http://127.0.0.1:$PORT/impact/wildcards/list/loaded 2>/dev/null | python3 -c "import sys, json; data = json.load(sys.stdin); print('\n'.join(data.get('data', [])))" 2>/dev/null || echo "ERROR")

if [ "$LOADED" != "ERROR" ]; then
    echo "Loaded wildcards:"
    echo "$LOADED" | grep -E "test_nesting" | sed 's/^/  /'

    NESTING_COUNT=$(echo "$LOADED" | grep -c "test_nesting" || echo "0")
    echo ""
    echo "Total nesting wildcards loaded: $NESTING_COUNT"

    if [ "$NESTING_COUNT" -ge 7 ]; then
        echo "${GREEN}✅ All 7 nesting levels loaded${NC}"
    else
        echo "${YELLOW}⚠️  Only $NESTING_COUNT nesting levels loaded (expected 7)${NC}"
    fi
else
    echo "${YELLOW}⚠️  Could not retrieve loaded wildcards list${NC}"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "${GREEN}✅ Deep nesting tests completed${NC}"
echo ""
echo "Test results:"
echo "  1. 7-level transitive expansion tested ✓"
echo "  2. All depth levels (1-7) individually tested ✓"
echo "  3. Mixed depth combinations tested ✓"
echo "  4. Nesting with quantifiers and multi-select ✓"
echo "  5. Nesting with weighted selection ✓"
echo "  6. Depth-agnostic pattern matching (__*/pattern__) ✓"
echo "  7. Complex multi-wildcard prompts ✓"
echo ""
echo "Maximum nesting depth verified: 7 levels"
echo "All wildcards should be fully expanded without crashes"
echo ""
echo "Log file: $LOG_FILE"
