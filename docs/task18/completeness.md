# TASK18 independent completeness — final round 2

Candidate: `af0961f3fbea174049fd61b504d85571068921652880439fb04b43ba9ba1e99f`; baseline `dd3b0f1b379ce0d5fbcfd543336b2d6093228a5b`. All 77 frozen file hashes were independently checked and matched. Reviewer did not implement production changes. Original TASK18 and common execution conditions remain the scope; no mandatory requirement was removed.

**100% confirmed: 10/10 mandatory criteria. Accepted for this exact frozen candidate.**

| Criterion | Result | Evidence |
|---|---|---|
| S01 | Confirmed | Eight serial before/after raw logs PASS with identical 54 profile names and three samples per profile; scaling.csv has 108 rows. Workload/harness review confirms corpus, limits and timing match. Process CPU/RSS and separate current-format retirement layout measured. |
| S02 | Confirmed | Aggregate totalLength/updateAvgLength is O(1); query captures only matching postings/candidates while retaining global statistics. Atomic replacement and immutable score snapshots tested. |
| S03 | Confirmed | Explicit bounded LRU; binding, prepared predicate, exact inventory and mutation generation key; fresh ledger and authority checks on hits and delivery. Scoped snapshots exclude private records before statistics. |
| S04 | Confirmed | Admitted-only endpoint adjacency, required selected-record bound and honest metadata/BYOT byte exclusions. Full admission measured through FindByIDs; cycle/self-loop/private bridge/conflict/shared-support suites retained. |
| S05 | Confirmed | Host-selected retirement protects current/unknown/unfinished operations, cleanup reference closure and durable live publication pins; exact cleanup confirmation required. |
| S06 | Confirmed | Maintenance uses existing flock/generation CAS/fsync/rename path; restart/crash/cancellation/concurrent CAS suites pass. Maintenance has no payload-deletion port. |
| S07 | Confirmed | Retired IDs/pins/fences cannot disappear or rebind; old operational handles fail ErrRetired. Round1 fresh-batch alias defect was repaired: one reservation index checks current inventories and each preceding fresh row. Unchanged independent repro now returns ErrConflict and leaves generation0/zero manifests; independent-final-batch-repro.txt. AAA tests cover empty/existing namespace atomic rejection and legitimate shared supports/distinct targets. |
| S08 | Confirmed | Read/write finite total snapshot capacity and explicit ErrCapacity; skeleton/fence/receipt growth remains finite-profile limitation. Single v2 schema rejects old format and documents backup/offline migration or reindex. |
| S09 | Confirmed | Actual volatile lexical/graph, persistent exact dense and candidate-only tensor, local APFS/flock/fsync capability matrix with limits/restart/reindex rules. |
| S10 | Confirmed | Root round2-final-module-status.json records all14 GOWORK=off race modules exit0, including actual PDF parser execution, and final core/OpenAI/conformance lint0. Independent focused race run passes lexical, managed lexical/graph, lifecycle and filestore; completeness-focused-race.txt. After the narrow round2 CAS guard, independent lifecycle/filestore race rerun passes both; completeness-round2-race.txt. Some root modules reuse valid Go test cache; core/conformance/PDF reran. |

Spec-First contracts/wire schema precede implementation; one persistent format and explicit required configuration fields implement the clear break. Domain metadata and ownership remain BYOT; retention/IAM, scheduling and source payload belong to host. Reviewed new tests use AAA and meaningful concurrent/revocation/crash barriers.

Measurements are not universal speedup evidence: selective managed lexical reads and uncompacted CAS have elapsed/byte regressions; graph adjacency adds bytes while reducing scan latency. Dense/tensor elapsed variation does not establish algorithm change. ns/op is wall throughput/mean operation time; whole-process CPU/RSS includes setup and compilation. No SLO, tail percentile, external-service verification or constant-space claim is accepted.

Final measurement lineage: round2 CAS/history and retirement layouts were rerun against final production paths, six profiles each and three samples, all PASS. The other 48 query profiles retain round1 measurements because their timed code and workload are unchanged by the narrow new-inventory CAS guard; their process CPU/RSS includes round1 untimed setup and is explicitly attributed to round1. Reports and CSV source columns preserve this distinction. No historical failure or original overlap log was rewritten into success.

Round1 completeness remains independently retained at results/round1-completeness.md (90%, S07 rejected). Final acceptance is conditional only on committing this verified candidate unchanged; there are no outstanding mandatory scope gaps. The earlier rejection was resolved with production validation and adversarial positive/negative regressions, not by narrowing the contract.
