#!/usr/bin/env bash

# Counters
UPDATED=0
SKIPPED=0
FAILED=0
NO_UPDATES=0

record_result() {
  case $1 in
    0) ((UPDATED++)) ;;
    1) ((NO_UPDATES++)) ;;
    2) ((SKIPPED++)) ;;
    3) ((FAILED++)) ;;
  esac
}

print_summary() {
  echo
  echo "=============================="
  echo "Summary:"
  echo "  Updated: $UPDATED"
  echo "  No updates: $NO_UPDATES"
  echo "  Skipped (dirty): $SKIPPED"
  echo "  Needs attention: $FAILED"
  echo "=============================="
}