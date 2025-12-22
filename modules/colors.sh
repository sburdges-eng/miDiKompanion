#!/usr/bin/env bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
color_echo() {
  local color=$1
  shift
  echo -e "${color}$*${NC}"
}

color_red() { color_echo "$RED" "$@"; }
color_green() { color_echo "$GREEN" "$@"; }
color_yellow() { color_echo "$YELLOW" "$@"; }
color_blue() { color_echo "$BLUE" "$@"; }