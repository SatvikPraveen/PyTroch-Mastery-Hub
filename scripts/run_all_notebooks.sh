#!/usr/bin/env bash
# scripts/run_all_notebooks.sh
# Execute all Jupyter notebooks in order and report pass/fail.
# Usage: bash scripts/run_all_notebooks.sh [--timeout 120] [--section 01]

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

TIMEOUT=120   # seconds per notebook
SECTION=""    # filter by section prefix
PASSED=0
FAILED=0
FAILED_LIST=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --section) SECTION="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/run_all_notebooks.sh [--timeout 120] [--section 01]"
      exit 0 ;;
    *) echo "Unknown arg: $1"; shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NOTEBOOKS_DIR="$PROJECT_ROOT/notebooks"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         PyTorch Mastery Hub — Notebook Runner        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Timeout : ${TIMEOUT}s per notebook"
echo "  Section : ${SECTION:-all}"
echo ""

# Find notebooks (sorted for deterministic order)
if [[ -n "$SECTION" ]]; then
  NOTEBOOKS=$(find "$NOTEBOOKS_DIR" -name "*.ipynb" | grep "/$SECTION" | sort)
else
  NOTEBOOKS=$(find "$NOTEBOOKS_DIR" -name "*.ipynb" | sort)
fi

TOTAL=$(echo "$NOTEBOOKS" | wc -l | tr -d ' ')
echo "  Found $TOTAL notebooks to execute."
echo ""

for notebook in $NOTEBOOKS; do
  rel_path="${notebook#"$PROJECT_ROOT/"}"
  printf "  %-60s" "$rel_path"

  output_file=$(mktemp /tmp/executed_nb_XXXXXX.ipynb)
  if timeout "$TIMEOUT" jupyter nbconvert \
       --to notebook \
       --execute "$notebook" \
       --output "$output_file" \
       --ExecutePreprocessor.timeout="$TIMEOUT" \
       --ExecutePreprocessor.kernel_name=python3 \
       > /dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
  else
    echo -e "${RED}FAIL${NC}"
    ((FAILED++))
    FAILED_LIST+=("$rel_path")
  fi
  rm -f "$output_file"
done

echo ""
echo "── Results ──────────────────────────────────────────────"
echo -e "  Passed : ${GREEN}${PASSED}${NC}"
echo -e "  Failed : ${RED}${FAILED}${NC}"
echo "  Total  : $TOTAL"

if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
  echo ""
  echo "  Failed notebooks:"
  for nb in "${FAILED_LIST[@]}"; do
    echo -e "    ${RED}✗${NC} $nb"
  done
  exit 1
else
  echo ""
  echo -e "  ${GREEN}All notebooks executed successfully! 🎉${NC}"
fi
