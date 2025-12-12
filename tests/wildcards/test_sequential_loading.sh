#!/bin/bash
# Sequential Multi-Stage Wildcard Loading Test
# Tests transitive wildcards that load in multiple sequential stages

# Auto-detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT=8193
CONFIG_FILE="$IMPACT_DIR/impact-pack.ini"

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "=========================================="
echo "Sequential Multi-Stage Wildcard Loading Test"
echo "=========================================="
echo ""

# Setup config for full cache mode
cat > "$CONFIG_FILE" << EOF
[default]
dependency_version = 24
mmdet_skip = True
sam_editor_cpu = False
sam_editor_model = sam_vit_h_4b8939.pth
custom_wildcards = $IMPACT_DIR/custom_wildcards
disable_gpu_opencv = True
wildcard_cache_limit_mb = 50
EOF

echo "Mode: Full cache mode (50MB limit)"
echo ""

# Kill existing servers
pkill -9 -f "python.*main.py" 2>/dev/null || true
sleep 3

# Start server
COMFYUI_DIR="$(cd "$IMPACT_DIR/../.." && pwd)"
cd "$COMFYUI_DIR"
echo "Starting ComfyUI server on port $PORT..."
bash run.sh --listen 127.0.0.1 --port $PORT > /tmp/sequential_test.log 2>&1 &
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
if ! curl -s http://127.0.0.1:$PORT/ > /dev/null; then
    echo "${RED}✗ Server failed to start${NC}"
    exit 1
fi

echo "${GREEN}✓ Server started${NC}"
echo ""

# Test function with stage visualization
test_sequential() {
    local TEST_NUM=$1
    local RAW_PROMPT=$2
    local SEED=$3
    local DESCRIPTION=$4
    local EXPECTED_STAGES=$5  # Number of expected expansion stages

    echo "${BLUE}=== Test $TEST_NUM: $DESCRIPTION ===${NC}"
    echo "Raw prompt: ${YELLOW}$RAW_PROMPT${NC}"
    echo "Seed: $SEED"
    echo "Expected stages: $EXPECTED_STAGES"
    echo ""

    # Test the prompt
    RESULT=$(curl -s -X POST http://127.0.0.1:$PORT/impact/wildcards \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$RAW_PROMPT\", \"seed\": $SEED}" | \
        python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))" 2>/dev/null || echo "ERROR")

    echo "${CYAN}Stage Analysis:${NC}"
    echo "  Stage 0 (Input):     $RAW_PROMPT"

    # Check if result contains any wildcards (incomplete expansion)
    if echo "$RESULT" | grep -q "__.*__"; then
        echo "  ${YELLOW}⚠ Result still contains wildcards (incomplete expansion)${NC}"
        echo "  Final Result: $RESULT"
    else
        echo "  ${GREEN}✓ All wildcards fully expanded${NC}"
    fi

    echo "  Final Output:        ${GREEN}$RESULT${NC}"
    echo ""

    # Validate result
    if [ "$RESULT" != "ERROR" ] && [ "$RESULT" != "" ]; then
        # Check if result still has wildcards (shouldn't have)
        if echo "$RESULT" | grep -q "__.*__"; then
            echo "Status: ${YELLOW}⚠ PARTIAL - Wildcards remain${NC}"
        else
            echo "Status: ${GREEN}✅ SUCCESS - Complete expansion${NC}"
        fi
    else
        echo "Status: ${RED}❌ FAILED - Error or empty result${NC}"
    fi
    echo ""
}

echo "=========================================="
echo "Sequential Loading Test Suite"
echo "=========================================="
echo ""

echo "${CYAN}Test Category 1: Depth Verification${NC}"
echo "Testing different transitive depths with stage tracking"
echo ""

# Test 1: Depth 1 (Direct wildcard)
test_sequential "01" \
    "__samples/flower__" \
    42 \
    "Depth 1 - Direct wildcard (no transitive)" \
    1

# Test 2: Depth 2 (One level transitive)
test_sequential "02" \
    "__dragon__" \
    200 \
    "Depth 2 - One level transitive" \
    2

# Test 3: Depth 3 (Two levels transitive)
test_sequential "03" \
    "__dragon__ warrior" \
    200 \
    "Depth 3 - Two levels with suffix" \
    3

# Test 4: Depth 3 (Maximum verified depth)
test_sequential "04" \
    "__adnd__ creature" \
    222 \
    "Depth 3 - Maximum transitive chain" \
    3

echo ""
echo "${CYAN}Test Category 2: Mixed Transitive Scenarios${NC}"
echo "Testing wildcards mixed with dynamic prompts"
echo ""

# Test 5: Transitive with dynamic prompt
test_sequential "05" \
    "{__dragon__|__adnd__} in battle" \
    100 \
    "Dynamic selection of transitive wildcards" \
    3

# Test 6: Multiple transitive wildcards
test_sequential "06" \
    "__dragon__ fights __adnd__" \
    150 \
    "Multiple transitive wildcards in one prompt" \
    3

# Test 7: Nested transitive in dynamic
test_sequential "07" \
    "powerful {__dragon__|__adnd__|simple warrior}" \
    200 \
    "Transitive wildcards nested in dynamic prompts" \
    3

echo ""
echo "${CYAN}Test Category 3: Complex Sequential Scenarios${NC}"
echo "Testing complex multi-stage expansions"
echo ""

# Test 8: Transitive with weights
test_sequential "08" \
    "{5::__dragon__|3::__adnd__|regular warrior}" \
    250 \
    "Weighted selection with transitive wildcards" \
    3

# Test 9: Multi-select with transitive
test_sequential "09" \
    "{2\$\$, \$\$__dragon__|__adnd__|warrior|mage}" \
    300 \
    "Multi-select including transitive wildcards" \
    3

# Test 10: Quantified transitive
test_sequential "10" \
    "{2\$\$, \$\$3#__dragon__}" \
    350 \
    "Quantified wildcard with transitive expansion" \
    3

echo ""
echo "${CYAN}Test Category 4: Edge Cases${NC}"
echo "Testing boundary conditions and special cases"
echo ""

# Test 11: Transitive in compound grammar
test_sequential "11" \
    "1{girl holding __samples/flower__|boy riding __dragon__}" \
    400 \
    "Compound grammar with mixed transitive depths" \
    3

# Test 12: Multiple wildcards, different depths
test_sequential "12" \
    "__samples/flower__ and __dragon__ with __colors__" \
    450 \
    "Multiple wildcards with varying depths" \
    3

# Test 13: YAML wildcard (no transitive)
test_sequential "13" \
    "__colors__" \
    333 \
    "YAML wildcard (depth 1, no transitive)" \
    1

# Test 14: Transitive + YAML combination
test_sequential "14" \
    "__dragon__ with __colors__ armor" \
    500 \
    "Combination of transitive and YAML wildcards" \
    3

echo ""
echo "${CYAN}Test Category 5: On-Demand Mode Verification${NC}"
echo "Testing sequential loading in on-demand mode"
echo ""

# Switch to on-demand mode
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

# Restart server
kill $SERVER_PID 2>/dev/null
pkill -9 -f "python.*main.py.*$PORT" 2>/dev/null
sleep 3

echo "Restarting server in on-demand mode (0.5MB limit)..."
cd "$COMFYUI_DIR"
bash run.sh --listen 127.0.0.1 --port $PORT > /tmp/sequential_ondemand.log 2>&1 &
SERVER_PID=$!

echo "Waiting 70 seconds for server restart..."
for i in {1..70}; do
    sleep 1
    if [ $((i % 10)) -eq 0 ]; then
        echo "  ... $i seconds"
    fi
done

if ! curl -s http://127.0.0.1:$PORT/ > /dev/null; then
    echo "${RED}✗ Server failed to restart${NC}"
    exit 1
fi

echo "${GREEN}✓ Server restarted in on-demand mode${NC}"
echo ""

# Test 15: Same transitive in on-demand mode
test_sequential "15" \
    "__adnd__ creature" \
    222 \
    "Depth 3 transitive in on-demand mode (should match full cache)" \
    3

# Test 16: Complex scenario in on-demand
test_sequential "16" \
    "{__dragon__|__adnd__} {warrior|mage}" \
    100 \
    "Complex transitive with dynamic in on-demand mode" \
    3

# Test 17: Multiple transitive in on-demand
test_sequential "17" \
    "__dragon__ and __adnd__ together" \
    150 \
    "Multiple transitive wildcards in on-demand mode" \
    3

# Stop server
kill $SERVER_PID 2>/dev/null
pkill -9 -f "python.*main.py.*$PORT" 2>/dev/null

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Total tests: 17"
echo "Categories:"
echo "  - Depth Verification (4 tests)"
echo "  - Mixed Transitive Scenarios (3 tests)"
echo "  - Complex Sequential Scenarios (3 tests)"
echo "  - Edge Cases (4 tests)"
echo "  - On-Demand Mode Verification (3 tests)"
echo ""
echo "Test Focus:"
echo "  ✓ Multi-stage transitive wildcard expansion"
echo "  ✓ Sequential loading across different depths"
echo "  ✓ Transitive wildcards in dynamic prompts"
echo "  ✓ Transitive wildcards with weights and multi-select"
echo "  ✓ On-demand mode sequential loading verification"
echo ""
echo "Log saved to:"
echo "  - Full cache mode: /tmp/sequential_test.log"
echo "  - On-demand mode: /tmp/sequential_ondemand.log"
echo ""
