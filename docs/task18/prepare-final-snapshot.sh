#!/bin/sh
set -eu
# Freeze tracked changes and untracked Go source for a read-only performance run.
# Run from repository root after production validation and before review fixes.
destination=${1:-/tmp/ragy-task18-after-source}
if test -e "$destination"; then
  echo "Destination already exists: $destination" >&2
  exit 2
fi
mkdir -p "$destination"
git archive HEAD | tar -x -C "$destination"
git diff --binary HEAD | (cd "$destination" && git apply)
git ls-files --others --exclude-standard | while IFS= read -r path; do
  case "$path" in
    *.go|go.mod|go.sum|*/go.mod|*/go.sum)
      mkdir -p "$destination/$(dirname "$path")"
      cp "$path" "$destination/$path"
      ;;
  esac
done
