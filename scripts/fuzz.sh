#!/bin/bash
set -euo pipefail
export GOWORK=off
(( $# >= 2 )) || { echo 'usage: fuzz.sh SECONDS MODULE...' >&2; exit 1; }
seconds=$1
shift
case "$seconds" in ''|*[!0-9]*) echo 'FUZZ_SECONDS must be a positive integer' >&2; exit 1;; esac
seconds=$((10#$seconds))
(( seconds > 0 )) || { echo 'FUZZ_SECONDS must be positive' >&2; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
for module in "$@"; do
  cd "$root/$module"
  packages=$("${GO:-go}" list -tags=fuzz ./...)
  for package in $packages; do
    names=$("${GO:-go}" test -tags=fuzz -list='^Fuzz' "$package")
    while IFS= read -r name; do
      case "$name" in Fuzz*) ;; *) continue;; esac
      case "$name" in *[[:space:]]*) continue;; esac
      "${GO:-go}" test -tags=fuzz -run='^$' -fuzz="^${name}$" -fuzztime="${seconds}s" -timeout="$((seconds+60))s" "$package"
    done <<< "$names"
  done
done
