#!/usr/bin/env bash
set -euo pipefail

PROFILE=${1:-profiles/standard.profile}
OUTPUT_DIR="dist"

if [ ! -f "$PROFILE" ]; then
  echo "Error: Profile not found: $PROFILE"
  exit 1
fi

# Load profile
source "$PROFILE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine output filename
OUTPUT_FILE="$OUTPUT_DIR/${OUTPUT_NAME:-git-update.sh}"

echo "Building: $PROFILE_NAME"
echo "Output: $OUTPUT_FILE"
echo ""

# Start building
{
  # Header
  cat core/header.sh
  echo ""
  
  # Modules
  if [ ${#MODULES[@]} -gt 0 ]; then
    echo "# ============================================"
    echo "# MODULES"
    echo "# ============================================"
    echo ""
    
    for module in "${MODULES[@]}"; do
      if [ -f "modules/$module.sh" ]; then
        echo "# --- Module: $module ---"
        # Skip shebang and set lines from modules
        grep -v '^#!/usr/bin/env bash' "modules/$module.sh" | grep -v '^set -euo pipefail' || true
        echo ""
      else
        echo "Warning: Module not found: modules/$module.sh" >&2
      fi
    done
  fi
  
  # Configuration
  echo "# ============================================"
  echo "# CONFIGURATION"
  echo "# ============================================"
  echo ""
  
  if [ -n "${CONFIG_SECTION:-}" ]; then
    echo "$CONFIG_SECTION"
    echo ""
  fi
  
  # Main loop
  cat core/main-loop.sh
  
  # Footer (if exists)
  if [ -f core/footer.sh ]; then
    echo ""
    cat core/footer.sh
  fi
  
} > "$OUTPUT_FILE"

# Make executable
chmod +x "$OUTPUT_FILE"

echo "✓ Build complete: $OUTPUT_FILE"
echo "  Profile: $PROFILE_NAME"
echo "  Modules: ${MODULES[*]:-none}"
echo ""
echo "Run with: ./$OUTPUT_FILE"