#!/bin/bash
set -euo pipefail
export GOWORK=off
seconds=${1:-30}
case "$seconds" in ''|*[!0-9]*) echo 'FUZZ_SECONDS must be an integer from 1 to 300' >&2; exit 1;; esac
(( seconds >= 1 && seconds <= 300 )) || exit 1
root=$(cd "$(dirname "$0")/.." && pwd)
while IFS= read -r module; do
  cd "$root/$module"
  packages=$("${GO:-go}" list -tags=fuzz ./...)
  for package in $packages; do
    names=$("${GO:-go}" test -tags=fuzz -list='^Fuzz' "$package")
    while IFS= read -r name; do
      case "$name" in Fuzz*) ;; *) continue;; esac
      case "$name" in *[[:space:]]*) continue;; esac
      "${GO:-go}" test -tags=fuzz -run='^$' -fuzz="^${name}$" -fuzztime="${seconds}s" -timeout="$((seconds+60))s" -parallel=2 "$package"
    done <<< "$names"
  done
done < "$root/scripts/check-modules.txt"
