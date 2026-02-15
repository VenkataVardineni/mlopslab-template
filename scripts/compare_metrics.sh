#!/bin/bash
# Script to compare metrics between git commits using dvc metrics diff

set -e

# Default: compare workspace with HEAD
BASE_REV="${1:-HEAD}"
TARGET_REV="${2:-}"

echo "=== DVC Metrics Comparison ==="
echo ""

if [ -z "$TARGET_REV" ]; then
    echo "Comparing workspace with $BASE_REV..."
    echo ""
    dvc metrics diff "$BASE_REV"
else
    echo "Comparing $BASE_REV with $TARGET_REV..."
    echo ""
    dvc metrics diff "$BASE_REV" "$TARGET_REV"
fi

echo ""
echo "=== Metrics Summary ==="
echo ""
echo "To view detailed plots comparison:"
echo "  dvc plots diff $BASE_REV ${TARGET_REV:-workspace}"
echo ""
echo "To view plots in browser:"
echo "  dvc plots show plots/forecast.csv"

