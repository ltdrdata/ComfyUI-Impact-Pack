#!/bin/bash
# UTF-8 Encoding Test Suite
# Tests multi-language support (Korean, Arabic smoke, emoji, special symbols)

set -euo pipefail

# Use an isolated server configured with tests/wildcards/samples.
# Server startup and configuration are described in docs/E2E_TEST_STRATEGY.md.
BASE_URL="${1:-http://127.0.0.1:8198}"
BASE_URL="${BASE_URL%/}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Expected-set ground truth for strict E2E assertions on pure-wildcard tests
# (tests 01/04/07). Derived from tests/wildcards/samples/*.txt; update here if
# the fixture changes. Empty lines and '#'-prefixed comment lines are filtered
# by modules/impact/wildcards.py:load_txt_wildcard (same filter this helper
# models via set membership).
KOREAN_EXPECTED=$'수놓은 별빛\n벚꽃 핑크\n강코랄\n옌로우\n챈메랄드\n챔무\n백설민주\n나부키하늘\n토미베이지\n율렌지\n블루지니\n캔디핑크'
EMOJI_EXPECTED=$'🌸🌺🌼🌻🌷🌹🥀🏵️💐🌾\n🌲🌳🌴🌵🌾🌿☘️🍀🍁🍂\n🌞🌝🌛🌜🌚🌕🌖🌗🌘🌑\n🐶🐱🐭🐹🐰🦊🐻🐼🐨🐯\n🦁🐮🐷🐸🐵🙈🙉🙊🐒🐔\n🍎🍊🍋🍌🍉🍇🍓🍈🍒🍑\n🍕🍔🍟🌭🍿🧂🥓🥚🍳🧇\n❤️🧡💛💚💙💜🖤🤍🤎💔\n💕💞💓💗💖💘💝💟☮️✝️\na 🌸 beautiful 🌺 garden 🌼 with 🌻 many 🌷 flowers 🌹'
SPECIAL_EXPECTED=$'∀∂∃∅∆∇∈∉∋∏∑−∓∔∕∖∗∘∙√∛∜∝∞∟\nα β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω\n$ € £ ¥ ₹ ₽ ₩ ₪ ₫ ₱ ₦ ₡ ₨ ₤ ₵ ₲ ₴\n┌─┬─┐\n│ │ │\n├─┼─┤\n└─┴─┘\nCafé résumé naïve Zürich\nàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ\n"double quotes" \'single quotes\' «guillemets» „German quotes"\n(parentheses) [brackets] {braces} ⟨angle brackets⟩\n… — – • · ° † ‡ § ¶ © ® ™ ℠ № ℮'

# Strict set-membership assertion: fails if RESULT does not byte-equal any
# line in EXPECTED_SET. Replaces the previous grep '[\x80-\xFF]' check, which
# passed for mojibake (any non-ASCII byte, correct or corrupt, satisfied it).
assert_set_membership() {
    local RESULT=$1
    local EXPECTED_SET=$2
    while IFS= read -r expected_line; do
        if [ "$RESULT" = "$expected_line" ]; then
            return 0
        fi
    done <<< "$EXPECTED_SET"
    return 1
}

echo "=========================================="
echo "UTF-8 Encoding Test Suite"
echo "=========================================="
echo "Server: $BASE_URL"
echo "Testing: Multi-language encoding support"
echo ""

curl --fail --silent --show-error --connect-timeout 5 --max-time 10 "$BASE_URL/" > /dev/null

# Test function
#   $1 TEST_NUM, $2 DESCRIPTION, $3 PROMPT, $4 SEED
#   $5 (optional) EXPECTED_SET — when non-empty, enforces strict byte-equal
#      set membership against the fixture ground truth. When empty (e.g. for
#      composite/dynamic prompts 02/03/05/08/...), falls back to a non-empty
#      check since full expected output cannot be enumerated ahead of time.
test_encoding() {
    local TEST_NUM=$1
    local DESCRIPTION=$2
    local PROMPT=$3
    local SEED=$4
    local EXPECTED_SET=${5:-}

    echo "${BLUE}=== Test $TEST_NUM: $DESCRIPTION ===${NC}"
    echo "Prompt: ${YELLOW}$PROMPT${NC}"
    echo "Seed: $SEED"

    RESULT=$(curl --fail --silent --show-error --connect-timeout 5 --max-time 30 -X POST "$BASE_URL/impact/wildcards" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$PROMPT\", \"seed\": $SEED}" | \
        python3 -c "import sys, json; print(json.load(sys.stdin).get('text','ERROR'))" 2>/dev/null || echo "ERROR")

    echo "Result: ${GREEN}$RESULT${NC}"

    if [ -n "$EXPECTED_SET" ]; then
        # Strict: RESULT must byte-equal one of the fixture lines. Catches mojibake.
        if assert_set_membership "$RESULT" "$EXPECTED_SET"; then
            echo "Status: ${GREEN}✅ PASS - exact match against fixture set${NC}"
        else
            echo "Status: ${RED}❌ FAIL - RESULT is not a member of the expected set (possible mojibake or phantom-split)${NC}"
            FAIL_COUNT=$((FAIL_COUNT+1))
        fi
    elif [ "$RESULT" = "ERROR" ] || [ -z "$RESULT" ]; then
        echo "Status: ${RED}❌ FAIL - Server error or no response${NC}"
        FAIL_COUNT=$((FAIL_COUNT+1))
    else
        echo "Status: ${GREEN}✅ PASS - non-empty response received${NC}"
    fi
    echo ""
}

echo "=========================================="
echo "Test Suite Execution"
echo "=========================================="
echo ""

# Script-level failure counter. test_encoding() FAIL branches increment this;
# the end-of-script guard converts any nonzero count to `exit 1` so the
# harness becomes gating in CI / `&&` chains. Previously all test_encoding
# FAIL branches only printed red text and the script exited 0 regardless,
# defeating the strict set-membership assertions added for #1196.
FAIL_COUNT=0

# Korean Tests (K-pop theme with Korean filename)
test_encoding "01" "Korean Hangul (아름다운색)" \
    "__아름다운색__" \
    100 \
    "$KOREAN_EXPECTED"

test_encoding "02" "Korean with emoji" \
    "🌸 __아름다운색__" \
    200

test_encoding "03" "Korean in dynamic prompt" \
    "{붉은|하얀|노란} __아름다운색__" \
    300

# Emoji Tests
test_encoding "04" "Emoji wildcard" \
    "__test_encoding_emoji__" \
    400 \
    "$EMOJI_EXPECTED"

test_encoding "05" "Multiple emojis" \
    "🌸 beautiful 🌺 garden 🌼" \
    500

test_encoding "06" "Emoji in dynamic prompt" \
    "{🌸|🌺|🌼|🌻|🌷}" \
    600

# Special Characters Tests
test_encoding "07" "Mathematical symbols" \
    "__test_encoding_special__" \
    700 \
    "$SPECIAL_EXPECTED"

test_encoding "08" "Currency symbols" \
    "Price: {$|€|£|¥|₩} 100" \
    800

# Mixed Language Tests
test_encoding "09" "Korean wildcard in composite prompt" \
    "아름다운 __아름다운색__" \
    900

test_encoding "10" "Korean + Emoji + English" \
    "🌸 beautiful 아름다운 __아름다운색__" \
    1000

# RTL (Right-to-Left) Tests
test_encoding "11" "Arabic RTL text" \
    "زهرة جميلة" \
    1100

# Edge Cases
test_encoding "12" "Korean in quantifier (아름다운색)" \
    "3#__아름다운색__" \
    1200

test_encoding "13" "Korean in multi-select (아름다운색)" \
    "{2\$\$, \$\$__아름다운색__|장미|벚꽃}" \
    1300

test_encoding "14" "Mixed UTF-8 in weighted selection" \
    "{5::🌸|3::장미|2::花}" \
    1400

test_encoding "15" "Very long Korean text (아름다운색)" \
    "아름다운 {붉은|하얀|노란|분홍|보라} __아름다운색__ 꽃밭에서" \
    1500

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "${GREEN}✅ Encoding tests completed${NC}"
echo ""
echo "All tests verified UTF-8 encoding support:"
echo "  1. Korean (Hangul) characters — strict set-membership ✓"
echo "  2. Emoji support — strict set-membership ✓"
echo "  3. Arabic (RTL) text — smoke only (non-empty echo; no strict assertion)"
echo "  4. Mathematical and special symbols — strict set-membership ✓"
echo "  5. Mixed multi-language content — smoke ✓"
echo "  6. UTF-8 in dynamic prompts — smoke ✓"
echo "  7. UTF-8 with quantifiers and multi-select — smoke ✓"
echo ""

if [ $FAIL_COUNT -ne 0 ]; then
    echo ""
    echo "${RED}FAILED: $FAIL_COUNT test(s) failed${NC}"
    exit 1
fi
