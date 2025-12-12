#!/bin/bash
# Find transitive wildcard references in the wildcard directories

# Auto-detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPACT_PACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WILDCARDS_DIR="$IMPACT_PACK_DIR/wildcards"
CUSTOM_WILDCARDS_DIR="$IMPACT_PACK_DIR/custom_wildcards"

echo "=========================================="
echo "Transitive Wildcard Reference Scanner"
echo "=========================================="
echo ""

echo "Scanning for wildcard references (pattern: __*__)..."
echo ""

# Function to find references in a file
find_references() {
    local file=$1
    local relative_path=${file#$IMPACT_PACK_DIR/}

    # Find all __wildcard__ patterns in the file
    local refs=$(grep -o '__[^_]*__' "$file" 2>/dev/null | sort -u)

    if [ -n "$refs" ]; then
        echo "📄 $relative_path"
        echo "   References:"
        echo "$refs" | while read -r ref; do
            # Remove __ from both ends
            local clean_ref=${ref#__}
            clean_ref=${clean_ref%__}

            # Check if referenced file exists
            local found=false

            # Check in wildcards/
            if [ -f "$WILDCARDS_DIR/$clean_ref.txt" ]; then
                echo "   → $ref (wildcards/$clean_ref.txt) ✓"
                found=true
            elif [ -f "$WILDCARDS_DIR/$clean_ref.yaml" ]; then
                echo "   → $ref (wildcards/$clean_ref.yaml) ✓"
                found=true
            elif [ -f "$WILDCARDS_DIR/$clean_ref.yml" ]; then
                echo "   → $ref (wildcards/$clean_ref.yml) ✓"
                found=true
            fi

            # Check in custom_wildcards/
            if [ -f "$CUSTOM_WILDCARDS_DIR/$clean_ref.txt" ]; then
                echo "   → $ref (custom_wildcards/$clean_ref.txt) ✓"
                found=true
            elif [ -f "$CUSTOM_WILDCARDS_DIR/$clean_ref.yaml" ]; then
                echo "   → $ref (custom_wildcards/$clean_ref.yaml) ✓"
                found=true
            elif [ -f "$CUSTOM_WILDCARDS_DIR/$clean_ref.yml" ]; then
                echo "   → $ref (custom_wildcards/$clean_ref.yml) ✓"
                found=true
            fi

            if [ "$found" = false ]; then
                echo "   → $ref ❌ (not found)"
            fi
        done
        echo ""
    fi
}

# Scan TXT files
echo "=== TXT Files with References ==="
echo ""
find "$WILDCARDS_DIR" "$CUSTOM_WILDCARDS_DIR" -name "*.txt" 2>/dev/null | while read -r file; do
    find_references "$file"
done

# Scan YAML files
echo ""
echo "=== YAML Files with References ==="
echo ""
find "$WILDCARDS_DIR" "$CUSTOM_WILDCARDS_DIR" -name "*.yaml" -o -name "*.yml" 2>/dev/null | while read -r file; do
    find_references "$file"
done

echo ""
echo "=========================================="
echo "Recommended Test Cases"
echo "=========================================="
echo ""
echo "1. Simple TXT wildcard:"
echo "   Input: __samples/flower__"
echo "   Type: Direct reference (no transitive)"
echo ""

# Find a good transitive TXT example
echo "2. TXT → TXT transitive:"
find "$CUSTOM_WILDCARDS_DIR" -name "*.txt" -exec grep -l "__.*__" {} \; 2>/dev/null | head -1 | while read -r file; do
    local basename=$(basename "$file" .txt)
    local first_ref=$(grep -o '__[^_]*__' "$file" 2>/dev/null | head -1)
    echo "   Input: __${basename}__"
    echo "   Resolves to: $first_ref (and others)"
    echo "   File: ${file#$IMPACT_PACK_DIR/}"
done
echo ""

echo "3. YAML transitive:"
echo "   Input: __colors__"
echo "   Resolves to: __cold__ or __warm__ → blue|red|orange|yellow"
echo "   File: custom_wildcards/test.yaml"
echo ""

echo "=========================================="
echo "Scan Complete"
echo "=========================================="
