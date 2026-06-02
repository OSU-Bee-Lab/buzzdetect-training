#!/usr/bin/env bash
# Usage: bash setup_worktree.sh <slug>
# Creates a new experiment worktree at .local/worktrees/<slug> on branch exp/<slug>
# and symlinks all shared data/binary dirs into it.
set -euo pipefail

if [ -z "${1-}" ]; then
    echo "Usage: bash setup_worktree.sh <slug>"
    exit 1
fi

EXP="$1"
ROOT="$(git rev-parse --show-toplevel)"
WT="$ROOT/.local/worktrees/$EXP"

echo "Creating worktree at $WT on branch exp/$EXP"
git worktree add "$WT" -b "exp/$EXP"

# Embedder dirs: replace git-checked-out dirs with symlinks so binary
# weights and other gitignored files are available
echo "Symlinking embedders..."
for d in "$ROOT/embedders/"/*/; do
    name="$(basename "$d")"
    rm -rf "$WT/embedders/$name"
    ln -s "$d" "$WT/embedders/$name"
done
ln -sf "$ROOT/embedders/embedding.py" "$WT/embedders/embedding.py"

# Set embeddings: symlink gitignored embeddings dir for every set that exists
echo "Symlinking set embeddings..."
for set_dir in "$ROOT/02_set/sets/"/*/; do
    setname="$(basename "$set_dir")"
    src="$ROOT/02_set/sets/$setname/embeddings"
    dst="$WT/02_set/sets/$setname/embeddings"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
    fi
done

# Test audio and embeddings
echo "Symlinking test data..."
for name in audio embeddings; do
    src="$ROOT/04_test/$name"
    dst="$WT/04_test/$name"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
    fi
done

# Sanity check
echo ""
echo "Sanity check:"
python_bin="$(conda run -n buzzdetect-train which python 2>/dev/null || echo '')"
if [ -n "$python_bin" ]; then
    conda run -n buzzdetect-train python -c "
import sys; sys.path.insert(0, '$WT')
import config
print('  config.ROOT:', config.ROOT)
import embedders.embedding
print('  embedders.embedding: OK')
" && echo "  All good." || echo "  WARNING: import check failed."
else
    echo "  (skipping import check — conda env not found)"
fi

echo ""
echo "Worktree ready: $WT"
echo "Branch: exp/$EXP"
