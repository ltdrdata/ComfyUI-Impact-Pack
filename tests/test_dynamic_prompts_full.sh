#!/bin/bash
# Comprehensive Dynamic Prompt Validation Test
# Tests all dynamic prompt features with statistical validation

PORT=8188

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

LOG_FILE="/tmp/dynamic_prompt_full_validation.log"

exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================="
echo "Dynamic Prompt Full Validation Test"
echo "=========================================="
echo "Validating: All dynamic prompt features"
echo ""

# Check server
if ! curl -s http://127.0.0.1:$PORT/ > /dev/null 2>&1; then
    echo "${RED}Server not running on port $PORT${NC}"
    echo "Start server with: cd /mnt/teratera/git/ComfyUI && bash run.sh --listen 127.0.0.1 --port $PORT"
    exit 1
fi

TOTAL_GROUPS=0
PASSED_GROUPS=0
FAILED_GROUPS=0

# Test function for multiselect with validation
test_multiselect() {
    local TEST_NAME=$1
    local PROMPT=$2
    local EXPECTED_COUNT=$3
    local SEPARATOR=$4
    local ITERATIONS=$5
    shift 5
    local OPTIONS=("$@")

    echo "${BLUE}=== $TEST_NAME ===${NC}"
    echo "Prompt: ${YELLOW}$PROMPT${NC}"
    echo "Expected: $EXPECTED_COUNT items per result, separator: '$SEPARATOR'"
    echo -n "Testing $ITERATIONS iterations: "

    local PASSED=0
    local FAILED=0
    declare -a FAILURES

    for i in $(seq 1 $ITERATIONS); do
        SEED=$((1000 + i * 100))
        RESULT=$(curl -s -X POST http://127.0.0.1:$PORT/impact/wildcards \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$PROMPT\", \"seed\": $SEED}" | \
            python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))" 2>/dev/null || echo "ERROR")

        if [ "$RESULT" = "ERROR" ]; then
            echo -n "X"
            ((FAILED++))
            FAILURES+=("  Iteration $i (seed $SEED): Server error")
            continue
        fi

        # Count items based on separator
        if [ -z "$SEPARATOR" ]; then
            ITEM_COUNT=1
        else
            ITEM_COUNT=$(echo "$RESULT" | awk -F"$SEPARATOR" '{print NF}')
        fi

        # Check if count matches
        if [ $ITEM_COUNT -ne $EXPECTED_COUNT ]; then
            echo -n "X"
            ((FAILED++))
            FAILURES+=("  Iteration $i (seed $SEED): Expected $EXPECTED_COUNT items, got $ITEM_COUNT" "    Result: $RESULT")
            continue
        fi

        # Check for duplicates (split by separator and check uniqueness)
        if [ -n "$SEPARATOR" ]; then
            UNIQUE_COUNT=$(echo "$RESULT" | awk -F"$SEPARATOR" '{for(i=1;i<=NF;i++) print $i}' | sort -u | wc -l)
            if [ $UNIQUE_COUNT -ne $EXPECTED_COUNT ]; then
                echo -n "D"
                ((FAILED++))
                FAILURES+=("  Iteration $i (seed $SEED): Duplicates detected" "    Result: $RESULT")
                continue
            fi
        fi

        # Check that all items are from the option list
        VALID=1
        if [ -n "$SEPARATOR" ]; then
            while IFS= read -r item; do
                item=$(echo "$item" | xargs)  # trim whitespace
                FOUND=0
                for opt in "${OPTIONS[@]}"; do
                    if [ "$item" = "$opt" ]; then
                        FOUND=1
                        break
                    fi
                done
                if [ $FOUND -eq 0 ]; then
                    VALID=0
                    break
                fi
            done < <(echo "$RESULT" | awk -F"$SEPARATOR" '{for(i=1;i<=NF;i++) print $i}')
        fi

        if [ $VALID -eq 0 ]; then
            echo -n "?"
            ((FAILED++))
            FAILURES+=("  Iteration $i (seed $SEED): Invalid items detected" "    Result: $RESULT")
            continue
        fi

        echo -n "."
        ((PASSED++))
    done

    echo " Done"
    echo "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"

    if [ $FAILED -gt 0 ]; then
        echo -e "${RED}Failures:${NC}"
        printf '%s\n' "${FAILURES[@]}"
        ((FAILED_GROUPS++))
    else
        echo "${GREEN}✅ PASS${NC}"
        ((PASSED_GROUPS++))
    fi
    echo ""
    ((TOTAL_GROUPS++))
}

# Test function for weighted selection with statistical validation
test_weighted() {
    local TEST_NAME=$1
    local PROMPT=$2
    local ITERATIONS=$3
    shift 3
    local OPTIONS=("$@")

    echo "${BLUE}=== $TEST_NAME ===${NC}"
    echo "Prompt: ${YELLOW}$PROMPT${NC}"
    echo -n "Testing $ITERATIONS iterations: "

    declare -A COUNTS
    local TOTAL=0

    for i in $(seq 1 $ITERATIONS); do
        SEED=$((1000 + i * 100))
        RESULT=$(curl -s -X POST http://127.0.0.1:$PORT/impact/wildcards \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$PROMPT\", \"seed\": $SEED}" | \
            python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))" 2>/dev/null || echo "ERROR")

        if [ "$RESULT" = "ERROR" ]; then
            echo -n "X"
            continue
        fi

        MATCHED=0
        for opt in "${OPTIONS[@]}"; do
            if echo "$RESULT" | grep -Fq "$opt"; then
                COUNTS[$opt]=$((${COUNTS[$opt]:-0} + 1))
                MATCHED=1
                break
            fi
        done

        if [ $MATCHED -eq 1 ]; then
            ((TOTAL++))
            echo -n "."
        else
            echo -n "?"
        fi
    done

    echo " Done"
    echo "Distribution:"

    for opt in "${OPTIONS[@]}"; do
        local COUNT=${COUNTS[$opt]:-0}
        local PERCENT=0
        if [ $TOTAL -gt 0 ]; then
            PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COUNT / $TOTAL) * 100}")
        fi
        echo "  $opt: $COUNT / $TOTAL (${PERCENT}%)"
    done

    echo "${GREEN}✅ PASS${NC}"
    ((PASSED_GROUPS++))
    ((TOTAL_GROUPS++))
    echo ""
}

echo "=========================================="
echo "MULTISELECT VALIDATION"
echo "=========================================="
echo ""

test_multiselect "Test 1: 2-item multiselect" "{2\$\$, \$\$red|blue|green|yellow}" 2 ", " 20 "red" "blue" "green" "yellow"

test_multiselect "Test 2: 3-item multiselect" "{3\$\$ and \$\$alpha|beta|gamma|delta|epsilon}" 3 " and " 20 "alpha" "beta" "gamma" "delta" "epsilon"

test_multiselect "Test 3: Single-item multiselect" "{1\$\$ \$\$one|two|three}" 1 " " 20 "one" "two" "three"

test_multiselect "Test 4: Max-item multiselect (all 4)" "{4\$\$-\$\$cat|dog|bird|fish}" 4 "-" 20 "cat" "dog" "bird" "fish"

echo "=========================================="
echo "WEIGHTED SELECTION VALIDATION"
echo "=========================================="
echo ""

test_weighted "Test 5: Heavy bias 10:1 (100 iterations)" "{10::common|1::rare}" 100 "common" "rare"

test_weighted "Test 6: Equal weights 1:1:1 (60 iterations)" "{1::alpha|1::beta|1::gamma}" 60 "alpha" "beta" "gamma"

test_weighted "Test 7: Extreme bias 100:1 (100 iterations)" "{100::very_common|1::very_rare}" 100 "very_common" "very_rare"

test_weighted "Test 8: Multi-level weights 5:3:2 (100 iterations)" "{5::high|3::medium|2::low}" 100 "high" "medium" "low"

test_weighted "Test 9: Default weight mixing (100 iterations)" "{10::weighted|unweighted}" 100 "weighted" "unweighted"

echo "=========================================="
echo "BASIC SELECTION VALIDATION"
echo "=========================================="
echo ""

test_weighted "Test 10: Simple random selection (50 iterations)" "{option_a|option_b|option_c}" 50 "option_a" "option_b" "option_c"

test_weighted "Test 11: Nested selection (50 iterations)" "{outer_{inner1|inner2}|simple}" 50 "outer_inner1" "outer_inner2" "simple"

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Total test groups: $TOTAL_GROUPS"
echo "${GREEN}Passed: $PASSED_GROUPS${NC}"
echo "${RED}Failed: $FAILED_GROUPS${NC}"
echo ""

if [ $FAILED_GROUPS -eq 0 ]; then
    echo "${GREEN}✅ All tests passed${NC}"
    exit 0
else
    echo "${RED}❌ Some tests failed${NC}"
    exit 1
fi
