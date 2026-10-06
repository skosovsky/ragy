#!/bin/sh
set -eu
# Run from repository root. Stable profile; race instrumentation is separate.
label=${1:?usage: sh docs/task18/run-benchmarks.sh before-or-after}
case "$label" in before|after) ;; *) exit 2 ;; esac
export GOCACHE=${GOCACHE:-/tmp/ragy-task18-go-cache}
mkdir -p docs/task18/results
go test ./lexical -run '^$' -bench Task18 -benchmem -benchtime=100ms -count=3 -cpu=4 > "docs/task18/results/$label-core.txt"
go test ./lifecycle/filestore -run '^$' -bench Task18 -benchmem -benchtime=100ms -count=3 -cpu=4 > "docs/task18/results/$label-history.txt"
go test ./lexical/managed ./graph/managed -run '^$' -bench Task18 -benchmem -benchtime=100ms -count=3 -cpu=4 > "docs/task18/results/$label-managed.txt"
go test ./dense/persistent ./tensor/persistent -run '^$' -bench Task18 -benchmem -benchtime=100ms -count=3 -cpu=4 > "docs/task18/results/$label-persistent.txt"
