# ============================================
# MAIN LOOP
# ============================================

for repo in "$ROOT"/*; do
  [ -d "$repo/.git" ] || continue
  
  REPO_NAME=$(basename "$repo")
  
  # Skip excluded repos (if EXCLUDE is defined)
  if [ ${#EXCLUDE[@]} -gt 0 ]; then
    for excluded in "${EXCLUDE[@]}"; do
      [[ "$REPO_NAME" == "$excluded" ]] && continue 2
    done
  fi
  
  (
    cd "$repo" || exit 1
    
    echo
    if command -v color_blue >/dev/null 2>&1; then
      color_blue "===== $REPO_NAME ====="
    else
      echo "===== $REPO_NAME ====="
    fi
    
    # Check for dirty working tree
    if ! git diff --quiet || ! git diff --cached --quiet || \
       [ -n "$(git ls-files --others --exclude-standard)" ]; then
      if command -v color_yellow >/dev/null 2>&1; then
        color_yellow "⚠️  Dirty working tree. Skipping."
      else
        echo "⚠️  Dirty working tree. Skipping."
      fi
      exit 2
    fi
    
    # Save current state (handles detached HEAD)
    CURRENT_REF=$(git symbolic-ref -q HEAD || git rev-parse HEAD)
    
    # Fetch updates
    echo "Fetching updates..."
    if ${VERBOSE:-false}; then
      git fetch --all --prune
    else
      git fetch --all --prune --quiet
    fi
    
    ANY_UPDATED=false
    
    # Update each branch
    for b in "${BRANCHES[@]}"; do
      if git show-ref --verify --quiet "refs/remotes/origin/$b"; then
        echo "Updating branch: $b"
        
        if git checkout "$b" 2>&1 | grep -qv "Already on"; then
          echo "  → Switched to $b"
        fi
        
        if ${VERBOSE:-false}; then
          if git pull --ff-only; then
            if command -v color_green >/dev/null 2>&1; then
              color_green "  ✓ Updated"
            else
              echo "  ✓ Updated"
            fi
            ANY_UPDATED=true
          else
            if command -v color_yellow >/dev/null 2>&1; then
              color_yellow "  ⚠️  Requires manual merge"
            else
              echo "  ⚠️  Requires manual merge"
            fi
            exit 3
          fi
        else
          if git pull --ff-only --quiet; then
            if command -v color_green >/dev/null 2>&1; then
              color_green "  ✓ Updated"
            else
              echo "  ✓ Updated"
            fi
            ANY_UPDATED=true
          else
            if command -v color_yellow >/dev/null 2>&1; then
              color_yellow "  ⚠️  Requires manual merge"
            else
              echo "  ⚠️  Requires manual merge"
            fi
            exit 3
          fi
        fi
      fi
    done
    
    # Restore original state
    if [ -n "$CURRENT_REF" ]; then
      git checkout "$CURRENT_REF" --quiet 2>&1 || {
        if command -v color_red >/dev/null 2>&1; then
          color_red "⚠️  Failed to restore original branch"
        else
          echo "⚠️  Failed to restore original branch"
        fi
        exit 3
      }
    fi
    
    if command -v color_green >/dev/null 2>&1; then
      color_green "✓ Done"
    else
      echo "✓ Done"
    fi
    
    $ANY_UPDATED && exit 0 || exit 1
  )
  
  # Record result (if function exists)
  if command -v record_result >/dev/null 2>&1; then
    record_result $?
  fi
done