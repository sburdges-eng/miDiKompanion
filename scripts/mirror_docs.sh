#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DOCS_MUSIC_BRAIN="${ROOT_DIR}/docs_music-brain"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

EXCLUDES=(
  "--exclude=lost-and-found/"
  "--exclude=final kel/"
  "--exclude=dist/"
  "--exclude=build/"
  "--exclude=__pycache__/"
  "--exclude=node_modules/"
  "--exclude=.git/"
)

RSYNC_FLAGS=("-a" "--delete" "--prune-empty-dirs")
if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_FLAGS+=("--dry-run" "--itemize-changes")
fi

copy_with_rsync() {
  local src="$1"
  local dest="$2"
  shift 2
  rsync "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" "$@" "$src" "$dest"
}

copy_with_python() {
  local src="$1"
  local dest="$2"
  shift 2

  python3 - <<'PY'
import os
import shutil
import sys

src = sys.argv[1]
dest = sys.argv[2]
patterns = sys.argv[3:]

def is_excluded(path):
    parts = path.replace("\\", "/").split("/")
    excluded = {"lost-and-found", "final kel", "dist", "build", "__pycache__", "node_modules", ".git"}
    return any(p in excluded for p in parts)

for root, dirs, files in os.walk(src):
    if is_excluded(root):
        dirs[:] = []
        continue
    rel = os.path.relpath(root, src)
    for name in files:
        if any(name.endswith(p) for p in patterns):
            src_path = os.path.join(root, name)
            rel_path = os.path.normpath(os.path.join(rel, name))
            dest_path = os.path.join(dest, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
PY
  "$src" "$dest" "$@"
}

sync_tree() {
  local src="$1"
  local dest="$2"
  if command -v rsync >/dev/null 2>&1; then
    copy_with_rsync "$src" "$dest"
  else
    copy_with_python "$src" "$dest" ".md" ".txt" ".json"
  fi
}

sync_flat_files() {
  local src="$1"
  local dest="$2"
  if command -v rsync >/dev/null 2>&1; then
    copy_with_rsync "$src" "$dest" "--include=*.md" "--include=*.txt" "--include=*.json" "--exclude=*"
  else
    copy_with_python "$src" "$dest" ".md" ".txt" ".json"
  fi
}

mkdir -p "$DEST_DOCS_MUSIC_BRAIN/Songwriting_Guides"

# 1) Mirror vault (docs only) into docs_music-brain/
sync_tree "$ROOT_DIR/vault/" "$DEST_DOCS_MUSIC_BRAIN/"

# 2) Mirror top-level Songwriting_Guides into docs_music-brain/Songwriting_Guides/
sync_tree "$ROOT_DIR/Songwriting_Guides/" "$DEST_DOCS_MUSIC_BRAIN/Songwriting_Guides/"

# 3) Mirror root-level doc/data files into docs_music-brain/ (flat)
sync_flat_files "$ROOT_DIR/" "$DEST_DOCS_MUSIC_BRAIN/"

echo "Mirror complete."${DRY_RUN:+" (dry run)"}
