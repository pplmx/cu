#!/bin/bash
set -e

BUILD_DIR="${1:-$PWD/build}"
COVERAGE_DIR="$BUILD_DIR/coverage"

echo "========================================"
echo "  Nova Coverage Gap Analysis"
echo "========================================"
echo "Build directory: $BUILD_DIR"
echo ""

if [ ! -f "$COVERAGE_DIR/lcov/total.info" ]; then
    echo "Error: Coverage data not found. Run generate_coverage.sh first."
    exit 1
fi

mkdir -p "$COVERAGE_DIR/gaps"

echo "Finding untested functions..."
grep -E "^SF:.*\.hpp?$" "$COVERAGE_DIR/lcov/total.info" | while read -r line; do
    file=$(echo "$line" | cut -d: -f2)
    if [ -f "$file" ]; then
        # Extract function names that have 0 hits
        grep -B1 "FNF:0" "$COVERAGE_DIR/lcov/total.info" | \
            grep "FN:" | \
            cut -d: -f2- | \
            while read -r fn_line; do
                fn_name=$(echo "$fn_line" | cut -d, -f2)
                fn_file=$(echo "$fn_line" | cut -d, -f1)
                echo "$fn_name ($fn_file)" >> "$COVERAGE_DIR/gaps/untested_functions.txt"
            done
    fi
done 2>/dev/null || true

# Alternative: Use lcov to extract uncovered functions
echo "Extracting uncovered functions..."
lcov --extract "$COVERAGE_DIR/lcov/total.info" \
    "$PWD/include/*" \
    --output-file "$COVERAGE_DIR/gaps/uncovered.info" 2>/dev/null || true

# Generate summary by module
echo ""
echo "========================================"
echo "  Coverage Gaps by Module"
echo "========================================"

MODULES=("memory" "algo" "neural" "fft" "graph" "raytrace" "stream")
for module in "${MODULES[@]}"; do
    if [ -f "$COVERAGE_DIR/module/${module}.info" ]; then
        coverage=$(grep -o 'lines\.\.\.\.: [0-9.]*' "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | head -1 | grep -o '[0-9.]*' || echo "N/A")
        printf "%-12s %s%%\n" "$module:" "$coverage"
    fi
done

echo ""
echo "Detailed gaps saved to: $COVERAGE_DIR/gaps/"
echo "========================================"
