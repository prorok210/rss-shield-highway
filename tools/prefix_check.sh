#!/usr/bin/env bash
# Runs regression tests of the working tree against the code of an earlier (pre-fix) commit.
#
#   tools/prefix_check.sh <commit> <transcript> <overlay path>... -- <pytest node id>...
#
# The tree of <commit> is exported to a temporary directory, the overlay paths (the new tests
# and their scenario builders) are copied over it from the working tree, and pytest runs the
# listed tests there. A regression test is valid if it fails in this run and passes in the
# working tree. Exit code 0 = no listed test passed on <commit>.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
commit="$1"; out="$2"; shift 2
overlay=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do overlay+=("$1"); shift; done
shift
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git archive "$commit" | tar -x -C "$tmp"
for p in "${overlay[@]}"; do
  mkdir -p "$tmp/$(dirname "$p")"
  cp -r "$p" "$tmp/$(dirname "$p")/"
done
mkdir -p "$(dirname "$out")"
{
  echo "pre-fix commit: $(git rev-parse --short "$commit") ($(git log -1 --format=%s "$commit" | cut -c1-80))"
  echo "overlay from the working tree: ${overlay[*]}"
  echo "tests: $*"
  echo
  (cd "$tmp" && "$PY" -m pytest -q -p no:cacheprovider -rf --tb=line "$@" 2>&1 \
     | sed -e "s#$tmp/##g" -e 's/ in [0-9.]*s.*$//')
} > "$out"
tail -1 "$out"
tail -1 "$out" | grep -Eq "failed|error" && ! tail -1 "$out" | grep -q "passed"
