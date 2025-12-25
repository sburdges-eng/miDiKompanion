PROFILE_NAME="Custom"
OUTPUT_NAME="git-update-custom.sh"

# Pick your modules
MODULES=(
  "colors"
  # "config"
  # "verbose"
  # "summary"
)

# Custom configuration
CONFIG_SECTION='
ROOT="."
BRANCHES=("main" "develop" "staging")
EXCLUDE=("old-project" "archived-repo")
VERBOSE=false
'