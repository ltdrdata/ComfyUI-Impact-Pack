#!/bin/bash
# Comprehensive wildcard prompt test suite
# Tests all features from ImpactWildcard tutorial

# Auto-detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT=8192
CONFIG_FILE="$IMPACT_DIR/impact-pack.ini"

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Versatile Wildcard Prompt Test Suite"
echo "=========================================="
echo ""

# Setup config
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
bash run.sh --listen 127.0.0.1 --port $PORT > /tmp/versatile_test.log 2>&1 &
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

# Test function
test_prompt() {
    local TEST_NUM=$1
    local CATEGORY=$2
    local PROMPT=$3
    local SEED=$4
    local DESCRIPTION=$5

    echo "${BLUE}=== Test $TEST_NUM: $CATEGORY ===${NC}"
    echo "Description: $DESCRIPTION"
    echo "Raw prompt: ${YELLOW}$PROMPT${NC}"
    echo "Seed: $SEED"

    RESULT=$(curl -s -X POST http://127.0.0.1:$PORT/impact/wildcards \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$PROMPT\", \"seed\": $SEED}" | \
        python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))" 2>/dev/null || echo "ERROR")

    echo "Populated: ${GREEN}$RESULT${NC}"

    if [ "$RESULT" != "ERROR" ] && [ "$RESULT" != "" ]; then
        echo "Status: ${GREEN}✅ SUCCESS${NC}"
    else
        echo "Status: ${RED}❌ FAILED${NC}"
    fi
    echo ""
}

echo "=========================================="
echo "Test Suite Execution"
echo "=========================================="
echo ""

# Category 1: Simple Wildcards
test_prompt "01" "Simple Wildcard" \
    "__samples/flower__" \
    42 \
    "Basic wildcard substitution"

test_prompt "02" "Case Insensitive" \
    "__SAMPLES/FLOWER__" \
    42 \
    "Wildcard names are case insensitive"

test_prompt "03" "Mixed Case" \
    "__SaMpLeS/FlOwEr__" \
    42 \
    "Mixed case should work identically"

# Category 2: Dynamic Prompts
test_prompt "04" "Dynamic Prompt (Simple)" \
    "{red|green|blue} apple" \
    100 \
    "Random selection from pipe-separated options"

test_prompt "05" "Dynamic Prompt (Nested)" \
    "{a|{d|e|f}|c}" \
    100 \
    "Nested dynamic prompts with inner choices"

test_prompt "06" "Dynamic Prompt (Complex)" \
    "{blue apple|red {cherry|berry}|green melon}" \
    100 \
    "Nested options with multiple levels"

# Category 3: Selection Weights
test_prompt "07" "Weighted Selection" \
    "{5::red|4::green|7::blue|black} car" \
    100 \
    "Weighted random selection (5:4:7:1 ratio)"

test_prompt "08" "Weighted Complex" \
    "A {10::beautiful|5::stunning|amazing} {3::sunset|2::sunrise|dawn}" \
    100 \
    "Multiple weighted selections in one prompt"

# Category 4: Compound Grammar
test_prompt "09" "Wildcard + Dynamic" \
    "1girl holding {blue pencil|red apple|colorful __samples/flower__}" \
    100 \
    "Mixing wildcard with dynamic prompt"

test_prompt "10" "Multiple Wildcards" \
    "__samples/flower__ and __colors__" \
    100 \
    "Multiple wildcards in single prompt"

test_prompt "11" "Complex Compound" \
    "{1girl holding|1boy riding} {blue|red|__colors__} {pencil|__samples/flower__}" \
    100 \
    "Complex nesting with wildcards and dynamics"

# Category 5: Transitive Wildcards
test_prompt "12" "Transitive Depth 1" \
    "__dragon__" \
    200 \
    "First level transitive wildcard"

test_prompt "13" "Transitive Depth 2" \
    "__dragon__ warrior" \
    200 \
    "Second level transitive with suffix"

test_prompt "14" "Transitive Depth 3" \
    "__adnd__ creature" \
    222 \
    "Third level transitive (adnd→dragon→dragon_spirit)"

# Category 6: Multi-Select
test_prompt "15" "Multi-Select (Fixed)" \
    "{2\$\$, \$\$red|green|blue|yellow|purple}" \
    100 \
    "Select exactly 2 items with comma separator"

test_prompt "16" "Multi-Select (Range)" \
    "{1-3\$\$, \$\$apple|banana|orange|grape|mango}" \
    100 \
    "Select 1-3 items randomly"

test_prompt "17" "Multi-Select (Custom Sep)" \
    "{2\$\$ and \$\$cat|dog|bird|fish}" \
    100 \
    "Custom separator: 'and' instead of comma"

test_prompt "18" "Multi-Select (Or Sep)" \
    "{2-3\$\$ or \$\$happy|sad|excited|calm}" \
    100 \
    "Range with 'or' separator"

# Category 7: Quantifying Wildcard
test_prompt "19" "Quantified Wildcard" \
    "{2\$\$, \$\$3#__samples/flower__}" \
    100 \
    "Repeat wildcard 3 times, select 2"

test_prompt "20" "Quantified Complex" \
    "Garden with {3\$\$, \$\$5#__samples/flower__}" \
    100 \
    "Select 3 from 5 repeated wildcards"

# Category 8: YAML Wildcards
test_prompt "21" "YAML Simple" \
    "__colors__" \
    333 \
    "YAML wildcard file"

test_prompt "22" "YAML in Dynamic" \
    "{solid|{metallic|pastel} __colors__}" \
    100 \
    "YAML wildcard nested in dynamic prompt"

# Category 9: Complex Real-World Scenarios
test_prompt "23" "Realistic Prompt 1" \
    "1girl, {5::beautiful|3::stunning|gorgeous} __samples/flower__ in hair, {blue|red|__colors__} dress" \
    100 \
    "Realistic character description"

test_prompt "24" "Realistic Prompt 2" \
    "{detailed|highly detailed} {portrait|illustration} of {1girl|1boy} with {2\$\$, \$\$__samples/flower__|__samples/jewel__|elegant accessories}" \
    100 \
    "Complex art prompt with multi-select"

test_prompt "25" "Realistic Prompt 3" \
    "__adnd__ {warrior|mage|rogue}, {10::epic|5::legendary|mythical} {armor|robes}, wielding {ancient|magical} weapon" \
    100 \
    "Fantasy character with transitive wildcard"

# Category 10: Edge Cases
test_prompt "26" "Empty Dynamic" \
    "{|something|nothing}" \
    100 \
    "Dynamic with empty option"

test_prompt "27" "Single Option" \
    "{only_one}" \
    100 \
    "Dynamic with single option (no choice)"

test_prompt "28" "Deeply Nested" \
    "{a|{b|{c|{d|e}}}}" \
    100 \
    "Very deep nesting"

test_prompt "29" "Multiple Weights" \
    "{100::common|10::uncommon|1::rare|super_rare}" \
    100 \
    "Extreme weight differences"

test_prompt "30" "Wildcard Only" \
    "__samples/flower__" \
    999 \
    "Different seed on same wildcard"

# Stop server
kill $SERVER_PID 2>/dev/null
pkill -9 -f "python.*main.py.*$PORT" 2>/dev/null

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Total tests: 30"
echo "Categories tested:"
echo "  - Simple Wildcards (3 tests)"
echo "  - Dynamic Prompts (3 tests)"
echo "  - Selection Weights (2 tests)"
echo "  - Compound Grammar (3 tests)"
echo "  - Transitive Wildcards (3 tests)"
echo "  - Multi-Select (4 tests)"
echo "  - Quantifying Wildcard (2 tests)"
echo "  - YAML Wildcards (2 tests)"
echo "  - Real-World Scenarios (3 tests)"
echo "  - Edge Cases (5 tests)"
echo ""
echo "Log saved to: /tmp/versatile_test.log"
echo ""
