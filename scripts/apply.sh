#!/usr/bin/env bash
# apply.sh [DIR]: fetch kindlingai/gr8 at the pinned commit into DIR (default ./gr8) and apply patches/*.patch with
# `git am`. Ends by checking that the result is exactly the source tree the measurements were built from.
#
# The Dockerfile runs this script to get its sources; you can also run it on its own to read or build the patched tree.
set -euo pipefail

GR8_URL=${GR8_URL:-https://github.com/kindlingai/gr8}
GR8_COMMIT=c53b5b869818950a96a9b72c647d4b156b057127      # upstream main, 2026-10-18
EXPECTED_TREE=67de5b7ce4d44db24352cc323f636eae906f2209   # git tree id after the 31 patches; the measured engines were built from trees that differ only in comments, test messages and build.sh's test list (0030, 0031)

here=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:-gr8}

if [ -e "$dir" ]; then
  echo "apply: $dir already exists; remove it or pass another directory" >&2
  exit 1
fi

git init -q "$dir"
cd "$dir"
# One commit, no history. Falls back to a full fetch if the server refuses a fetch by commit id.
git fetch -q --depth 1 "$GR8_URL" "$GR8_COMMIT" 2>/dev/null || git fetch -q "$GR8_URL"
git checkout -q --detach "$GR8_COMMIT"

# git am records a committer; use the caller's identity if it has one, else a local placeholder.
name=$(git config user.name || echo "gr8 recipe")
email=$(git config user.email || echo "recipe@localhost")
git -c user.name="$name" -c user.email="$email" am -q --committer-date-is-author-date "$here"/patches/*.patch

tree=$(git rev-parse 'HEAD^{tree}')
if [ "$tree" != "$EXPECTED_TREE" ]; then
  echo "apply: tree $tree != expected $EXPECTED_TREE" >&2
  exit 1
fi
echo "apply: $(pwd) = gr8 ${GR8_COMMIT:0:7} + $(ls "$here"/patches/*.patch | wc -l | tr -d ' ') patches, tree ${tree:0:12} OK"
