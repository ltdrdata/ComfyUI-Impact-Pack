#!/bin/bash
# Quick test for wildcard lazy loading

echo "=========================================="
echo "Wildcard Lazy Load Quick Test"
echo "=========================================="
echo ""

# Test 1: Get wildcard list (before accessing any wildcards)
echo "=== Test 1: Wildcard List (BEFORE access) ==="
curl -s http://127.0.0.1:8188/impact/wildcards/list > /tmp/wc_list_before.json
COUNT_BEFORE=$(cat /tmp/wc_list_before.json | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('data', [])))")
echo "Total wildcards: $COUNT_BEFORE"
echo ""

# Test 2: Simple wildcard
echo "=== Test 2: Simple Wildcard ==="
curl -s -X POST http://127.0.0.1:8188/impact/wildcards \
  -H "Content-Type: application/json" \
  -d '{"text": "__samples/flower__", "seed": 42}' > /tmp/wc_simple.json
RESULT2=$(cat /tmp/wc_simple.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('text', 'ERROR'))")
echo "Input:  __samples/flower__"
echo "Output: $RESULT2"
echo ""

# Test 3: Depth 3 transitive
echo "=== Test 3: Depth 3 Transitive (TXT→TXT→TXT) ==="
curl -s -X POST http://127.0.0.1:8188/impact/wildcards \
  -H "Content-Type: application/json" \
  -d '{"text": "__adnd__ creature", "seed": 222}' > /tmp/wc_depth3.json
RESULT3=$(cat /tmp/wc_depth3.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('text', 'ERROR'))")
echo "Input:  __adnd__ creature"
echo "Output: $RESULT3"
echo "Chain:  adnd → (dragon/beast/...) → (dragon_spirit/...)"
echo ""

# Test 4: YAML transitive
echo "=== Test 4: YAML Transitive ==="
curl -s -X POST http://127.0.0.1:8188/impact/wildcards \
  -H "Content-Type: application/json" \
  -d '{"text": "__colors__", "seed": 333}' > /tmp/wc_yaml.json
RESULT4=$(cat /tmp/wc_yaml.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('text', 'ERROR'))")
echo "Input:  __colors__"
echo "Output: $RESULT4"
echo "Chain:  colors → (cold|warm) → (blue|red|orange|yellow)"
echo ""

# Test 5: Get wildcard list (AFTER accessing wildcards)
echo "=== Test 5: Wildcard List (AFTER access) ==="
curl -s http://127.0.0.1:8188/impact/wildcards/list > /tmp/wc_list_after.json
COUNT_AFTER=$(cat /tmp/wc_list_after.json | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('data', [])))")
echo "Total wildcards: $COUNT_AFTER"
echo ""

# Compare
echo "=========================================="
echo "Results"
echo "=========================================="
echo ""
if [ "$COUNT_BEFORE" -eq "$COUNT_AFTER" ]; then
    echo "✅ Wildcard list unchanged: $COUNT_BEFORE = $COUNT_AFTER"
else
    echo "❌ Wildcard list changed: $COUNT_BEFORE != $COUNT_AFTER"
fi

if [ "$RESULT2" != "ERROR" ] && [ "$RESULT3" != "ERROR" ] && [ "$RESULT4" != "ERROR" ]; then
    echo "✅ All wildcards resolved successfully"
else
    echo "❌ Some wildcards failed"
fi

echo ""
echo "Check /tmp/comfyui_ondemand.log for loading mode"
grep -i "wildcard.*mode" /tmp/comfyui_ondemand.log | tail -1
