PROFILE_NAME="Full Featured"
OUTPUT_NAME="git-update-full.sh"

# All modules
MODULES=(
  "colors"
  "config"
  "verbose"
  "summary"
)

# Configuration
CONFIG_SECTION='
# Defaults
DEFAULT_ROOT="."
DEFAULT_BRANCHES=("main" "dev")
DEFAULT_EXCLUDE=()

# Parse arguments
parse_verbose_flag "${1:-}"

# Load config if exists
if [ -f ".git-update-config" ]; then
  source ".git-update-config"
else
  ROOT="${ROOT:-$DEFAULT_ROOT}"
  BRANCHES=("${BRANCHES[@]:-${DEFAULT_BRANCHES[@]}}")
  EXCLUDE=("${EXCLUDE[@]:-${DEFAULT_EXCLUDE[@]}}")
fi
'