#!/usr/bin/env bash
# make-patches.sh GR8_CLONE [REF]: regenerate patches/ from a gr8 clone whose REF (default HEAD) sits on the pinned
# upstream commit. This is how patches/0001..0022 were made; the series is plain `git format-patch` output:
#
#   git format-patch --no-signature --zero-commit -o patches <GR8_COMMIT>..<REF>
#
# --zero-commit writes 0000... in each patch's first line instead of a local commit id, and --no-signature drops the
# git version trailer, so the files depend only on the commits' content, authors, dates and messages.
# After regenerating, scripts/apply.sh checks that the series rebuilds the expected source tree (EXPECTED_TREE there).
set -euo pipefail

GR8_COMMIT=6448a16661abd81e9540bc073b3cfa92d7234565   # kindlingai/gr8 main, "S2: size the notice target ... (#66)"

src=${1:?usage: make-patches.sh GR8_CLONE [REF]}
ref=${2:-HEAD}
here=$(cd "$(dirname "$0")/.." && pwd)

git -C "$src" merge-base --is-ancestor "$GR8_COMMIT" "$ref" \
  || { echo "make-patches: $ref is not on top of $GR8_COMMIT" >&2; exit 1; }

rm -f "$here"/patches/*.patch
git -C "$src" format-patch --no-signature --zero-commit -o "$here/patches" "$GR8_COMMIT..$ref"
echo "tree of $ref: $(git -C "$src" rev-parse "$ref^{tree}")  (update EXPECTED_TREE in scripts/apply.sh if it changed)"
