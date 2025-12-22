PROFILE_NAME="Standard"
OUTPUT_NAME="git-update.sh"

# Standard modules
MODULES=(
  "colors"
  "config"
)

# Configuration
CONFIG_SECTION='
# Defaults
DEFAULT_ROOT="."
DEFAULT_BRANCHES=("main" "dev")
DEFAULT_EXCLUDE=()

# Load config if exists
if [ -f ".git-update-config" ]; then
  source ".git-update-config"
else
  ROOT="${ROOT:-$DEFAULT_ROOT}"
  BRANCHES=("${BRANCHES[@]:-${DEFAULT_BRANCHES[@]}}")
  EXCLUDE=("${EXCLUDE[@]:-${DEFAULT_EXCLUDE[@]}}")
fi

VERBOSE=false
'