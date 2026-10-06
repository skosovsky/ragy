#!/bin/sh
set -eu
# Run from final repository root, with no other test/benchmark processes running.
# Baseline source is an archive of dd3b0f1 plus the benchmark harness from before
# production changes. Its managed configuration uses the then-current API.
label=${1:?usage: sh docs/task18/run-serial-benchmarks.sh before-or-after source-directory}
case "$label" in before|after) ;; *) exit 2 ;; esac
output_root=$(pwd)/docs/task18/results
source_root=${2:-$(pwd)}
export GOCACHE=${GOCACHE:-/tmp/ragy-task18-go-cache}
export TMPDIR=/tmp
mkdir -p "$output_root"
cd "$source_root"
/usr/bin/time -l go test ./lexical -run '^$' -bench Task18 -benchmem -benchtime=100ms -count=3 -cpu=4 > "$output_root/serial-$label-core.txt" 2> "$output_root/serial-$label-core-process-stats.txt"
/usr/bin/time -l go test ./lifecycle/filestore -run '^$' -bench '^BenchmarkTask18Filestore/' -benchmem -benchtime=100ms -count=3 -cpu=4 > "$output_root/serial-$label-history.txt" 2> "$output_root/serial-$label-history-process-stats.txt"
/usr/bin/time -l go test ./lexical/managed ./graph/managed -run '^$' -bench Task18 -benchmem -benchtime=100ms -count=3 -cpu=4 > "$output_root/serial-$label-managed.txt" 2> "$output_root/serial-$label-managed-process-stats.txt"
/usr/bin/time -l go test ./dense/persistent ./tensor/persistent -run '^$' -bench Task18 -benchmem -benchtime=100ms -count=3 -cpu=4 > "$output_root/serial-$label-persistent.txt" 2> "$output_root/serial-$label-persistent-process-stats.txt"
