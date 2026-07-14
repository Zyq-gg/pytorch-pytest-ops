#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SKILLS_DIR=${CODEX_HOME:-$HOME/.codex}/skills
DEST=$SKILLS_DIR/pytorch-pytest-ops

mkdir -p "$SKILLS_DIR"

if [[ -L "$DEST" ]]; then
  CURRENT=$(readlink -f "$DEST")
  if [[ "$CURRENT" == "$ROOT" ]]; then
    echo "Skill already installed: $DEST -> $ROOT"
  else
    echo "ERROR: $DEST already points to $CURRENT" >&2
    echo "Remove or rename it explicitly before installing this clone." >&2
    exit 2
  fi
elif [[ -e "$DEST" ]]; then
  echo "ERROR: $DEST already exists and is not a symlink." >&2
  echo "Move it explicitly before installing this clone." >&2
  exit 2
else
  ln -s "$ROOT" "$DEST"
  echo "Installed skill: $DEST -> $ROOT"
fi

python3 "$ROOT/scripts/self_check.py"
echo "Open a new Codex session if the skill is not visible in the current session."
