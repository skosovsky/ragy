#!/bin/sh
set -eu
# Run from repository root; destination must not exist to preserve captures.
destination=${1:-/tmp/ragy-task18-before-source}
if test -e "$destination"; then
  echo "Destination already exists: $destination" >&2
  exit 2
fi
mkdir -p "$destination"
git archive dd3b0f1b379ce0d5fbcfd543336b2d6093228a5b | tar -x -C "$destination"
for package in lexical lexical/managed graph/managed lifecycle/filestore dense/persistent tensor/persistent; do
  cp "docs/task18/harness-before/$package/task18_benchmark_test.go.txt" "$destination/$package/task18_benchmark_test.go"
done
