# TASK-18 independent correctness review

Verdict: **ACCEPT** for frozen candidate `af0961f3fbea174049fd61b504d85571068921652880439fb04b43ba9ba1e99f` (77 files), baseline `dd3b0f1b379ce0d5fbcfd543336b2d6093228a5b`. Every listed SHA-256 file hash matched before review and after independent tests. The reviewer did not implement production changes; no confirmed correctness defect remains open.

## Independent execution

The original independent AAA raw-CAS reproduction was rerun unchanged against the final candidate under Go 1.27.1, `GOWORK=off`, separate `GOCACHE=/tmp/ragy-task18-review-cache`, `go test -mod=mod -race -count=1 -v ./...`. Both exported `ValidateReplacement` and the real filestore reject overlapping new plans with `ErrConflict`; returned and durable generations remain 0, with no persisted manifests. [Raw result](results/independent-final-batch-repro.txt), [retained harness](results/independent-batch-repro.go.txt).

Independent `go test -race -count=1 ./lexical ./lexical/managed ./graph/managed ./lifecycle ./lifecycle/filestore` passed all five packages: BM25 1.450s; managed lexical 4.119s; managed graph 6.928s; lifecycle 6.080s; filestore 4.169s. This includes atomic rebuild and concurrent reads, cache capacity/freshness/revocation, graph private bridges/cycles/conflicts/supports, retirement protection, durable pin replay/release, concurrent CAS, crash/lock and persistence fault tests. [Raw result](results/independent-final-correctness-race.txt). No race detector report occurred.

Root final-round evidence was inspected separately: [all 14 module statuses](results/round2-final-module-status.json) have exit 0, backed by `round2-final-race-*.txt`. Some unchanged nested-module results use Go's cache; core and conformance ran for 101.225s and 46.024s. Actual configured PDF parser tests passed; no live database/provider credentials or remote service verification is inferred. Core, OpenAI and conformance lint logs report zero issues. Historical failed reproduction and formatting lint evidence remain retained.

## Contract review

| Contract | Reviewed evidence and conclusion |
|---|---|
| S01 | Frozen baseline/harness, serial raw samples, allocation/latency/storage tables and process CPU/RSS logs retained. Final history/retirement measurements were rerun after the batch correction; unchanged lexical/graph/persistent timed paths retain their earlier exact source lineage. Reports separate timed operations from setup/process costs. |
| S02 | BM25 maintains total length under the same lock as document/posting mutation. Rebuild publishes complete staged maps only on success. Query copies matching postings and candidate rows under one read lock, retaining complete admitted corpus statistics. |
| S03 | Managed cache keys include binding, prepared filter, exact inventory and mutation generation. LRU entry capacity is mandatory. Hits still check ledger/publication/inventory and authority freshness, with owned output metadata cloning. Filtered documents are removed before scoped BM25 statistics are built. |
| S04 | Required admission cardinality counts selected facts before filtering/deduplication. Adjacency is built after scope and conflict admission, requires both endpoints, and preserves directed/undirected cycles and self-loops. Full-view and traversal measurements expose admission cost separately from shape-only AdmitTraversal. |
| S05 | Explicit retirement selects exact IDs and protects current publication, unknown outcomes, active operation ancestry, unfinished cleanup closures and live registered exact-target pins. Registration/release share namespace CAS and survive restart. |
| S06 | Maintenance uses the existing whole-snapshot lock/CAS/fsync/rename path. Independent package tests include concurrent writers, cancellation, rename faults and process-death lock release. Maintenance dispatches no source/target deletion. |
| S07 | Retired skeletons/fences and pin identities are immutable; previous plans and receipts remain reserved. The final batch ownership index reserves current inventory/fences, then each new row before the next. The unchanged independent reproduction confirms public/custom-store validation and real CAS both enforce the corrected invariant. Shared supports and distinct target ownership remain allowed. |
| S08 | Descriptor-bounded reads and serialized byte limits return explicit ErrCapacity, without automatic deletion. Skeletons/fences/jobs/pins consume finite storage; reopening under a sufficient budget and offline migration/fresh-root reindex are documented. |
| S09 | Capability matrix distinguishes volatile lexical/graph, persistent exact dense scan, candidate-only exact MaxSim, and local APFS flock/fsync semantics. No ANN/distributed guarantee is implied. |
| S10 | Independent five-package race checks plus inspected final 14-module/lint evidence pass. Source inspection found no new provider/OTel core dependency or agent runtime; public metadata/cloning/identity contracts remain host-owned BYOT. |

## Resolved finding and limits

Round 1 (`195f84cfd9514f1d477acca75e1e1bf4868f2e64762d37f8058af8fbadc76da7`) was rejected: a batch could persist two new plans with different content/payload fingerprints and the same exact `(target, source.Reference)`. The original validator checked each only against the prior snapshot. The implementation correction uses one incremental ownership index across current and fresh rows, and adds AAA negative/positive regressions. [Historical rejection](results/round1-correctness.md), [original failing log](results/independent-batch-repro.txt).

Acceptance covers the stated local contracts and fixtures, not an unbounded-storage or universal-performance claim. Narrow managed lexical workloads and filestore CAS have measured regressions; graph admission still loads/scans retained metadata, and cache/admission cardinalities do not bound arbitrary host payload bytes. Dense and tensor remain exact algorithms. Measurements use three small samples on one APFS host with uncontrolled system load and provide no tail percentile or production SLO. During both quiet measurement windows this reviewer performed only readonly inspection/hash checks; independent tests ran after explicit QUIETCOMPLETE. Performance logs were inspected, not independently rebenchmarked.
