#!/usr/bin/env bash
# apply.sh [DIR]: fetch kindlingai/gr8 at the pinned commit into DIR (default ./gr8) and apply patches/*.patch with
# `git am`. Ends by checking that the result is exactly the source tree the measurements were built from.
#
# The Dockerfile runs this script to get its sources; you can also run it on its own to read or build the patched tree.
set -euo pipefail

GR8_URL=${GR8_URL:-https://github.com/kindlingai/gr8}
GR8_COMMIT=6448a16661abd81e9540bc073b3cfa92d7234565      # upstream main, 2026-10-14
EXPECTED_TREE=be4496db2ea043cbea7dc5a0acb49e084784dccf   # git tree id after the 33 patches; differs from the built tree (bec488b62de9ec852dbf2a49eda566f4ef9005b3, branch v07) only in 4 test/tool files where upstream main drifted past the pin (engine/attn/tests/attnk.cu, engine/kernels/tests/k1_audit.cu, engine/runtime/s2/tests/batch_dselmodel_test.cpp, tools/s2_dsel_table.py) - no runtime files differ

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
