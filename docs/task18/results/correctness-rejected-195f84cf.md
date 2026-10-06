# TASK-18 independent correctness review

Verdict: **REJECT** for candidate `195f84cfd9514f1d477acca75e1e1bf4868f2e64762d37f8058af8fbadc76da7` (baseline `dd3b0f1b379ce0d5fbcfd543336b2d6093228a5b`). All 76 listed SHA-256 file hashes matched before and after review. Production files were read only; the reviewer did not participate in implementation.

## Confirmed defect

**S07: batch raw CAS can alias an exact target artifact identity.**

`lifecycle/maintenance.go:validateManifestReplacements` checks each new manifest against inventories/fences in the previous snapshot only. `Snapshot.Validate` checks unique manifest IDs and keys but does not reject duplicate `(target, source.Reference)` reservations across manifests introduced together. Consequently `ValidateReplacement` and the real filestore accept two fresh planned manifests with different IDs, keys, content fingerprints and payload fingerprints, reserving the same exact target/reference. This bypasses the no-alias rule enforced by serial Executor.Prepare and can persist incompatible operation plans for a single target record.

Independent AAA reproduction used an empty valid namespace and two such plans, then called the exported replacement validator and actual durable CompareSwap. Under `GOWORK=off GOCACHE=/tmp/ragy-task18-review-cache go test -mod=mod -race -count=1 -v ./...`, both operations returned nil and the durable Load contained generation 1 with two manifests. The test required ErrConflict and unchanged empty durable state, and failed. This is a correctness failure, not a race detector report.

Raw result: [independent-batch-repro.txt](results/independent-batch-repro.txt). Retained source: [independent-batch-repro.go.txt](results/independent-batch-repro.go.txt). The isolated module used a local replace to this repository and no provider/network service.

## Review coverage and limits

Read S01–S10 and task/common contracts; inspected BM25 aggregate/query snapshot ownership, scoped lexical cache identity and freshness/privacy gates, graph admission/adjacency/conflicts, lifecycle retirement closures and pins, immutable old plans/fences/receipts, generation CAS, bounded descriptor reads and atomic flock/fsync/rename persistence. No additional confirmed production defect arose from this source review. Existing final logs report 14 module race runs passing and core lint zero issues; those are implementation evidence, not substitute independent tests. Root reports an outstanding nested OpenAI formatting lint failure; acceptance also requires resolving it.

The serial before/after benchmark window was respected: only readonly inspection and small hash verification occurred during measurement. The independent reproduction ran after QUIETCOMPLETE. Performance measurements were not independently rerun, and no external/live provider verification is claimed. The candidate cannot be accepted until the confirmed alias defect is fixed and the replacement candidate, affected tests, lint and performance evidence are reviewed.
