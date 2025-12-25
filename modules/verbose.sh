#!/usr/bin/env bash

VERBOSE=false

parse_verbose_flag() {
  if [[ "${1:-}" == "-v" || "${1:-}" == "--verbose" ]]; then
    VERBOSE=true
    return 0
  fi
  return 1
}