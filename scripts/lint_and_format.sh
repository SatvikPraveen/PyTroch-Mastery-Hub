#!/usr/bin/env bash
# scripts/lint_and_format.sh
# Run all linting and formatting tools in one command.
# Usage: bash scripts/lint_and_format.sh [--check]

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

CHECK_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=true; shift ;;
    -h|--help)
      echo "Usage: bash scripts/lint_and_format.sh [--check]"
      echo "  --check  Only check, do not modify files (for CI)"
      exit 0 ;;
    *) echo "Unknown arg: $1"; shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

ERRORS=0

run_tool() {
  local name="$1"; shift
  printf "  %-20s" "$name"
  if "$@" > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
  else
    echo -e "${RED}FAIL${NC}"
    ((ERRORS++))
  fi
}

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        PyTorch Mastery Hub — Lint & Format           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

if $CHECK_ONLY; then
  echo "  Mode: CHECK ONLY (no files modified)"
  echo ""
  run_tool "black (check)"   black --check src tests
  run_tool "isort (check)"   isort --check-only src tests
  run_tool "flake8"          flake8 src tests
  run_tool "bandit"          bandit -r src -x tests -q
else
  echo "  Mode: FORMAT + LINT"
  echo ""
  run_tool "black"    black src tests
  run_tool "isort"    isort src tests
  run_tool "flake8"   flake8 src tests
  run_tool "bandit"   bandit -r src -x tests -q
fi

echo ""
if [[ $ERRORS -eq 0 ]]; then
  echo -e "  ${GREEN}All checks passed! ✓${NC}"
else
  echo -e "  ${RED}${ERRORS} tool(s) reported issues.${NC}"
  exit 1
fi
