#!/bin/bash
set -e

BUILD_DIR="${1:-$PWD/build}"
COVERAGE_DIR="$BUILD_DIR/coverage"

echo "========================================"
echo "  Nova Coverage Summary"
echo "========================================"
echo "Build directory: $BUILD_DIR"
echo ""

if [ ! -f "$COVERAGE_DIR/lcov/total.info" ]; then
    echo "Error: Coverage data not found. Run generate_coverage.sh first."
    exit 1
fi

# Extract overall coverage
TOTAL_LINES=$(grep "lines......:" "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
TOTAL_BRANCHES=$(grep "branches....:" "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
TOTAL_FUNCTIONS=$(grep "functions..." "$COVERAGE_DIR/lcov/total.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")

echo "## Overall Coverage"
printf "| Metric     | Coverage |\n"
printf "|------------|----------|\n"
printf "| Lines      | %s%% |\n" "$TOTAL_LINES"
printf "| Branches   | %s%% |\n" "$TOTAL_BRANCHES"
printf "| Functions  | %s%% |\n" "$TOTAL_FUNCTIONS"

echo ""
echo "## Coverage by Module"
printf "| Module     | Lines     | Branches  | Functions |\n"
printf "|------------|-----------|-----------|-----------|\n"

MODULES=("memory" "algo" "neural" "fft" "graph" "raytrace" "stream" "device" "distributed" "nccl" "pipeline" "mesh")
for module in "${MODULES[@]}"; do
    if [ -f "$COVERAGE_DIR/module/${module}.info" ]; then
        lines=$(grep "lines......:" "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
        branches=$(grep "branches....:" "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
        funcs=$(grep "functions..." "$COVERAGE_DIR/module/${module}.info" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
        printf "| %-10s | %9s | %9s | %9s |\n" "$module" "$lines" "$branches" "$funcs"
    fi
done

echo ""
echo "========================================"

# Check minimum coverage threshold
MIN_COVERAGE="${NOVA_COVERAGE_MIN:-80}"
if [ "$TOTAL_LINES" != "N/A" ]; then
    PASSING=$(echo "$TOTAL_LINES >= $MIN_COVERAGE" | bc -l 2>/dev/null || echo "0")
    if [ "$PASSING" = "1" ]; then
        echo "✓ Coverage ($TOTAL_LINES%) meets minimum ($MIN_COVERAGE%)"
    else
        echo "✗ Coverage ($TOTAL_LINES%) below minimum ($MIN_COVERAGE%)"
        exit 1
    fi
fi
