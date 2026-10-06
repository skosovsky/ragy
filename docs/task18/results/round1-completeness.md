# TASK18 independent completeness — round 1

Candidate: `195f84cfd9514f1d477acca75e1e1bf4868f2e64762d37f8058af8fbadc76da7`; baseline `dd3b0f1b379ce0d5fbcfd543336b2d6093228a5b`. All 76 frozen file hashes were independently checked and matched. Reviewer did not implement production changes. Original TASK18 and common execution conditions remain the scope; no mandatory requirement was removed.

**90% confirmed: 9/10 mandatory criteria. Rejected pending S07 repair.**

| Criterion | Result | Evidence |
|---|---|---|
| S01 | Confirmed | Eight serial before/after raw logs PASS with identical 54 profile names and three samples per profile; scaling.csv has 108 rows. Workload/harness review confirms corpus, limits and timing match. Process CPU/RSS and separate current-format retirement layout measured. |
| S02 | Confirmed | Aggregate totalLength/updateAvgLength is O(1); query captures only matching postings/candidates while retaining global statistics. Atomic replacement and immutable score snapshots tested. |
| S03 | Confirmed | Explicit bounded LRU; binding, prepared predicate, exact inventory and mutation generation key; fresh ledger and authority checks on hits and delivery. Scoped snapshots exclude private records before statistics. |
| S04 | Confirmed | Admitted-only endpoint adjacency, required selected-record bound and honest metadata/BYOT byte exclusions. Full admission measured through FindByIDs; cycle/self-loop/private bridge/conflict/shared-support suites retained. |
| S05 | Confirmed | Host-selected retirement protects current/unknown/unfinished operations, cleanup reference closure and durable live publication pins; exact cleanup confirmation required. |
| S06 | Confirmed | Maintenance uses existing flock/generation CAS/fsync/rename path; restart/crash/cancellation/concurrent CAS suites pass. Maintenance has no payload-deletion port. |
| S07 | **Not confirmed** | Existing-history reservations work, but ValidateReplacement compares new rows only with current history. Two fresh plans added in a single raw CAS can share an exact target/source.Reference with different payloads. Independent repro persisted both at generation1; independent-batch-repro.txt records failure. |
| S08 | Confirmed | Read/write finite total snapshot capacity and explicit ErrCapacity; skeleton/fence/receipt growth remains finite-profile limitation. Single v2 schema rejects old format and documents backup/offline migration or reindex. |
| S09 | Confirmed | Actual volatile lexical/graph, persistent exact dense and candidate-only tensor, local APFS/flock/fsync capability matrix with limits/restart/reindex rules. |
| S10 | Confirmed | Root final-module-status.json records all14 GOWORK=off race modules exit0 and core lint0. Independent focused race run of lexical, managed lexical/graph, lifecycle and filestore passes all5; completeness-focused-race.txt. |

Spec-First contracts/wire schema precede implementation; one persistent format and explicit required configuration fields implement the clear break. Domain metadata and ownership remain BYOT; retention/IAM, scheduling and source payload belong to host. Reviewed new tests use AAA and meaningful concurrent/revocation/crash barriers.

Measurements are not universal speedup evidence: selective managed lexical reads and uncompacted CAS have elapsed/byte regressions; graph adjacency adds bytes while reducing scan latency. Dense/tensor elapsed variation does not establish algorithm change. ns/op is wall throughput/mean operation time; whole-process CPU/RSS includes setup and compilation. No SLO, tail percentile, external-service verification or constant-space claim is accepted.
