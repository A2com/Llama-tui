#!/usr/bin/env bash
# TDD tests for qwen wrapper script
# Run: bash tests/test_qwen_wrapper.sh

set -uo pipefail

WRAPPER="${QWEN_WRAPPER:-/Users/a2com/.local/bin/qwen}"
LLAMA_URL="${LLAMA_URL:-http://localhost:8081}"
PASS=0
FAIL=0

green() { printf '\033[32m✓ %s\033[0m\n' "$1"; }
red()   { printf '\033[31m✗ %s\033[0m\n' "$1"; }
assert_eq() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    green "$label"
    ((PASS++))
  else
    red "$label — expected '$expected', got '$actual'"
    ((FAIL++))
  fi
}
assert_contains() {
  local label="$1" needle="$2" haystack="$3"
  if echo "$haystack" | grep -q "$needle" 2>/dev/null; then
    green "$label"
    ((PASS++))
  else
    red "$label — '$needle' not found in output"
    ((FAIL++))
  fi
}
assert_exit_code() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" -eq "$actual" ]; then
    green "$label"
    ((PASS++))
  else
    red "$label — expected exit $expected, got $actual"
    ((FAIL++))
  fi
}

# ─── Test 1: wrapper exists ───
echo "=== Test 1: wrapper exists ==="
if [ -x "$WRAPPER" ]; then
  green "wrapper is executable"
  ((PASS++))
else
  red "wrapper not found or not executable at $WRAPPER"
  ((FAIL++))
fi

# ─── Test 2: --version returns success ───
echo "=== Test 2: --version ==="
OUTPUT=$("$WRAPPER" --version 2>&1) && RC=$? || RC=$?
assert_exit_code "--version exits cleanly" 0 "$RC"
assert_contains "--version output mentions qwen" "qwen" "$OUTPUT"

# ─── Test 3: --help returns usage ───
echo "=== Test 3: --help ==="
OUTPUT=$("$WRAPPER" --help 2>&1) && RC=$? || RC=$?
assert_contains "--help shows usage" "Usage" "$OUTPUT"

# ─── Test 4: stdin prompt produces output ───
echo "=== Test 4: stdin prompt ==="
OUTPUT=$(echo "Say exactly: hello world" | "$WRAPPER" --yolo 2>/dev/null) && RC=$? || RC=$?
assert_exit_code "--yolo exits cleanly" 0 "$RC"
assert_contains "stdin prompt produces output" "hello" "$OUTPUT"

# ─── Test 5: --model flag is accepted ───
echo "=== Test 5: --model flag ==="
OUTPUT=$(echo "Say exactly: test model" | "$WRAPPER" --yolo --model qwen3-coder-plus 2>/dev/null) && RC=$? || RC=$?
assert_exit_code "--model flag accepted" 0 "$RC"
assert_contains "--model flag produces output" "test" "$OUTPUT"

# ─── Test 6: invalid flag errors out ───
echo "=== Test 6: invalid flag ==="
OUTPUT=$("$WRAPPER" --invalid-flag 2>&1) && RC=$? || RC=$?
assert_exit_code "invalid flag returns non-zero" 1 "$RC"

# ─── Summary ───
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1