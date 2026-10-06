# T13 completeness acceptance

Baseline: `4846cd00f71a0c031416b5f9856e146b2ca19e26`. Independent nonimplementing acceptor. Reviewed current unstaged/untracked T13 candidate against supplied AGENTS.md, master D15–D22, lifecycle review, arch-docs:02, backlog criteria, traceability and preimplementation normative contract. No implementation files changed by this acceptor.

## Verdict

**PASS — criteria 5/5 = 100%; assigned source coverage 28/28 = 100%.** No SKIP or future intent counted. Custom Store hardening L-H01 is not promoted to a supported-native cross-tenant defect; confirmed-F count remains unchanged.

## Criteria

| Criterion | Result | Actual evidence |
|---|---|---|
| T13.C01 — 5 domains of published lifecycle contract and explicit 8 master/18 lifecycle dispositions | PASS | Current lifecycle/README.md: glossary, Capture/Acquire, namespace-wide CAS/reuse, operation-specific recovery, finite-step Cleaner/Complete. Traceability dispositions individually checked against actual source and guide. |
| T13.C02 — Namespace admission and unchanged successful capture | PASS | publication.go compares requested namespace immediately after Load; helper tautology removed. New 16-case strict/partial × empty/nonempty × matching/misrouted/schema/malformed matrix checks protocol+zero+no writes; caller-invalid checks no Load. Existing strict/partial unavailable/empty behavior retained. |
| T13.C03 — Reservations/replay/CAS/fences/context | PASS | Current guide documents permanent ABA identities, semantic target/artifact order, trusted low-level CAS and non-nil context. New durable order replay/CAS test plus existing pins/maintenance/fence tests prove retained contracts. |
| T13.C04 — Scale/profile and optimization decision | PASS | Guide documents history CPU/bytes/nested scans, finite snapshot capacity not RSS/CPU, supported trusted darwin/linux filesystem and power-loss limits. Clone/index algorithms unchanged in git diff; no speedup claim, before/after N/A; future measured+differential validation required. |
| T13.C05 — Targeted fresh lifecycle race/replay/fence checks | PASS | Independent full and focused race logs; exact results recorded below. |

## Assigned source rows

Each disposition below has been checked against current implementation, current guide and actual test scope, not merely the presence of a traceability entry. Retain/contract items require no algorithm rewrite.

| Row | Disposition | Rationale and evidence |
|---|---|---|
| D15 | contract — PASS | Keep names to avoid gratuitous churn; current guide distinguishes source pointer, immutable read capture and durable metadata reservation. Capture not lease; pin metadata not payload/authorization. Evidence: `lifecycle/README.md`, `lifecycle/manifest.go`, `lifecycle/publication.go`, `lifecycle/pins_test.go`. |
| D16 | change — PASS | Retain requested namespace across Store.Load before deriving publication; mismatched or malformed response yields protocol zero/no writes. Remove helper tautology. Custom Store hardening, not a new native/F defect. Evidence: `lifecycle/publication.go`, `lifecycle/capture_namespace_test.go`, `lifecycle/publication_test.go`. |
| D17 | contract — PASS | Namespace generation remains conservative across sources/pins. Only positive reuse rechecks generation; all decisions are observations, not leases/permission. Host reloads/reconciles rather than hidden retry. Evidence: `lifecycle/README.md`, `lifecycle/reuse.go`, `lifecycle/reuse_test.go`, `lifecycle/pins_test.go`. |
| D18 | contract — PASS | Retain finite Executor/Cleaner steps. Current guide separates dispatch unknown, read-only inspection unknown and CAS acknowledgment unknown, plus host clock/backoff/overdue recovery and Complete scope. Evidence: `lifecycle/README.md`, `lifecycle/executor_test.go`, `lifecycle/cleanup_test.go`, `lifecycle/cleanup.go`. |
| D19 | retain — PASS | Permanent retired skeletons/digests/released IDs/bootstrap+cleanup receipts prevent ABA/rebinding and preserve replay. Capacity never evicts automatically; metadata retention and payload cleanup separate. Evidence: `lifecycle/README.md`, `lifecycle/maintenance_test.go`, `lifecycle/pins_test.go`, `lifecycle/filestore/maintenance_unix_test.go`. |
| D20 | retain — PASS | Finite call count is not CPU/bytes/RSS bound. Whole history validation/scans/serialization and JSON ownership clone remain unchanged. No optimization/speedup claimed; before/after N/A. Future optimization requires measured profile and differential ownership/order validation. Evidence: `lifecycle/README.md`, `lifecycle/executor.go`, `lifecycle/cleanup.go`, `lifecycle/maintenance.go`, `lifecycle/maintenance_test.go`. |
| D21 | retain — PASS | Plan target/artifact order remains semantic to reserved replay; capture sorting does not authorize reordering plans. Raw CAS is trusted checkpoint port with mandatory ValidateReplacement, not command authorization. Evidence: `lifecycle/README.md`, `lifecycle/store.go`, `lifecycle/order_contract_unix_test.go`, `lifecycle/maintenance_test.go`. |
| D22 | retain — PASS | Sorted simultaneous inventory fences and exactly-once synchronous cooperative callback preserve exact inventory. No goroutine detachment, implicit workers or distributed commitment claim. Evidence: `lifecycle/README.md`, `lifecycle/inventory_verifier.go`, `lifecycle/inventory_verifier_test.go`. |
| lifecycle:01 | retain — PASS | Executor/Cleaner encode CAS/publication/unknown invariants; keep finite protocol and host scheduling, no new agent/workflow runtime. Evidence: `lifecycle/README.md`, `lifecycle/executor_test.go`, `lifecycle/cleanup_test.go`, `lifecycle/cleanup.go`. |
| lifecycle:02 | contract — PASS | Keep names to avoid gratuitous churn; current guide distinguishes source pointer, immutable read capture and durable metadata reservation. Capture not lease; pin metadata not payload/authorization. Evidence: `lifecycle/README.md`, `lifecycle/manifest.go`, `lifecycle/publication.go`, `lifecycle/pins_test.go`. |
| lifecycle:03 | contract — PASS | Keep names to avoid gratuitous churn; current guide distinguishes source pointer, immutable read capture and durable metadata reservation. Capture not lease; pin metadata not payload/authorization. Evidence: `lifecycle/README.md`, `lifecycle/manifest.go`, `lifecycle/publication.go`, `lifecycle/pins_test.go`. |
| lifecycle:04 | contract — PASS | Namespace generation remains conservative across sources/pins. Only positive reuse rechecks generation; all decisions are observations, not leases/permission. Host reloads/reconciles rather than hidden retry. Evidence: `lifecycle/README.md`, `lifecycle/reuse.go`, `lifecycle/reuse_test.go`, `lifecycle/pins_test.go`. |
| lifecycle:05 | contract — PASS | Namespace generation remains conservative across sources/pins. Only positive reuse rechecks generation; all decisions are observations, not leases/permission. Host reloads/reconciles rather than hidden retry. Evidence: `lifecycle/README.md`, `lifecycle/reuse.go`, `lifecycle/reuse_test.go`, `lifecycle/pins_test.go`. |
| lifecycle:06 | contract — PASS | Namespace generation remains conservative across sources/pins. Only positive reuse rechecks generation; all decisions are observations, not leases/permission. Host reloads/reconciles rather than hidden retry. Evidence: `lifecycle/README.md`, `lifecycle/reuse.go`, `lifecycle/reuse_test.go`, `lifecycle/pins_test.go`. |
| lifecycle:07 | contract — PASS | Retain finite Executor/Cleaner steps. Current guide separates dispatch unknown, read-only inspection unknown and CAS acknowledgment unknown, plus host clock/backoff/overdue recovery and Complete scope. Evidence: `lifecycle/README.md`, `lifecycle/executor_test.go`, `lifecycle/cleanup_test.go`, `lifecycle/cleanup.go`. |
| lifecycle:08 | contract — PASS | Retain finite Executor/Cleaner steps. Current guide separates dispatch unknown, read-only inspection unknown and CAS acknowledgment unknown, plus host clock/backoff/overdue recovery and Complete scope. Evidence: `lifecycle/README.md`, `lifecycle/executor_test.go`, `lifecycle/cleanup_test.go`, `lifecycle/cleanup.go`. |
| lifecycle:09 | retain — PASS | Finite call count is not CPU/bytes/RSS bound. Whole history validation/scans/serialization and JSON ownership clone remain unchanged. No optimization/speedup claimed; before/after N/A. Future optimization requires measured profile and differential ownership/order validation. Evidence: `lifecycle/README.md`, `lifecycle/executor.go`, `lifecycle/cleanup.go`, `lifecycle/maintenance.go`, `lifecycle/maintenance_test.go`. |
| lifecycle:10 | retain — PASS | Permanent retired skeletons/digests/released IDs/bootstrap+cleanup receipts prevent ABA/rebinding and preserve replay. Capacity never evicts automatically; metadata retention and payload cleanup separate. Evidence: `lifecycle/README.md`, `lifecycle/maintenance_test.go`, `lifecycle/pins_test.go`, `lifecycle/filestore/maintenance_unix_test.go`. |
| lifecycle:11 | contract — PASS | Retain finite Executor/Cleaner steps. Current guide separates dispatch unknown, read-only inspection unknown and CAS acknowledgment unknown, plus host clock/backoff/overdue recovery and Complete scope. Evidence: `lifecycle/README.md`, `lifecycle/executor_test.go`, `lifecycle/cleanup_test.go`, `lifecycle/cleanup.go`. |
| lifecycle:12 | retain — PASS | Finite call count is not CPU/bytes/RSS bound. Whole history validation/scans/serialization and JSON ownership clone remain unchanged. No optimization/speedup claimed; before/after N/A. Future optimization requires measured profile and differential ownership/order validation. Evidence: `lifecycle/README.md`, `lifecycle/executor.go`, `lifecycle/cleanup.go`, `lifecycle/maintenance.go`, `lifecycle/maintenance_test.go`. |
| lifecycle:13 | retain — PASS | Plan target/artifact order remains semantic to reserved replay; capture sorting does not authorize reordering plans. Raw CAS is trusted checkpoint port with mandatory ValidateReplacement, not command authorization. Evidence: `lifecycle/README.md`, `lifecycle/store.go`, `lifecycle/order_contract_unix_test.go`, `lifecycle/maintenance_test.go`. |
| lifecycle:14 | retain — PASS | Plan target/artifact order remains semantic to reserved replay; capture sorting does not authorize reordering plans. Raw CAS is trusted checkpoint port with mandatory ValidateReplacement, not command authorization. Evidence: `lifecycle/README.md`, `lifecycle/store.go`, `lifecycle/order_contract_unix_test.go`, `lifecycle/maintenance_test.go`. |
| lifecycle:15 | retain — PASS | Finite call count is not CPU/bytes/RSS bound. Whole history validation/scans/serialization and JSON ownership clone remain unchanged. No optimization/speedup claimed; before/after N/A. Future optimization requires measured profile and differential ownership/order validation. Evidence: `lifecycle/README.md`, `lifecycle/executor.go`, `lifecycle/cleanup.go`, `lifecycle/maintenance.go`, `lifecycle/maintenance_test.go`. |
| lifecycle:16 | retain — PASS | Sorted simultaneous inventory fences and exactly-once synchronous cooperative callback preserve exact inventory. No goroutine detachment, implicit workers or distributed commitment claim. Evidence: `lifecycle/README.md`, `lifecycle/inventory_verifier.go`, `lifecycle/inventory_verifier_test.go`. |
| lifecycle:17 | change — PASS | Stable current lifecycle/README.md now contains states, legal replay, unknown recovery, errors, pin distinctions, capacity/scale/filesystem/schema; top-level README points here rather than task chronology. Evidence: `lifecycle/README.md`, `lifecycle/manifest.go`, `lifecycle/publication.go`, `lifecycle/pins_test.go`. |
| lifecycle:18 | contract — PASS | All context-taking APIs require non-nil Go context. Existing defensive pin rejection retained, no promise of package-wide nil handling. Nil-context hygiene is not promoted to defect. Evidence: `lifecycle/manifest.go`, `lifecycle/README.md`. |
| arch-docs:02 | retain — PASS | Sorted simultaneous inventory fences and exactly-once synchronous cooperative callback preserve exact inventory. No goroutine detachment, implicit workers or distributed commitment claim. Evidence: `lifecycle/README.md`, `lifecycle/inventory_verifier.go`, `lifecycle/inventory_verifier_test.go`. |
| L-H01 | change — PASS | Retain requested namespace across Store.Load before deriving publication; mismatched or malformed response yields protocol zero/no writes. Remove helper tautology. Custom Store hardening, not a new native/F defect. Evidence: `lifecycle/publication.go`, `lifecycle/capture_namespace_test.go`, `lifecycle/publication_test.go`. |

## Independent checks

`GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./lifecycle/...` — PASS; lifecycle, filestore and integration, fresh uncached test executions. Raw log: `T13-completeness-race.log`.

`GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 -v ./lifecycle -run 'TestCapture|TestReservedPlanOrder|TestFencedVerifier|TestReuse|TestPublicationPin|TestExecutorUnknown|TestCleanupFakeClock|TestUnknownCleanup'` — PASS. Raw log: `T13-completeness-focused.log`; all 16 capture matrix children and retained replay/fence/unknown/cleanup cases passed.

An initial command accidentally named nonexistent `./integration`; that setup failure was not counted and was replaced with the actual `./lifecycle/...` scope. No live external backend or hardware power-loss result claimed. Scope is T13 targeted completeness, not the later repository-wide T22 gate.

## Candidate SHA256

These hashes freeze every changed implementation/test, normative contract and current public guide file. Bookkeeping, mutable task journal and acceptance reports are excluded. Source changes after these hashes require reacceptance.

| File | SHA256 |
|---|---|
| `README.md` | `53adcf246d1942e74960dcec83844778ebd7af4fa3cce23c0c34f48fb648eda5` |
| `docs/contracts/remediation.md` | `1cc024e03fb7a6e96f0b2d29e9ec7e2de6d7199b716d9756dbc5cce169ca8060` |
| `lifecycle/cleanup.go` | `ab73c9393cc48f4d68049da6c5cec55ee0b4413d9e49678546bb0a58073a5aae` |
| `lifecycle/executor.go` | `a97525f6c3b3ff9607e7fb328dec0b5f59652cd673078f7a16ad0506e8fbce69` |
| `lifecycle/manifest.go` | `5feaceee40df8128acfcbb2294beb755aa11001c1808e0de37549582450663fa` |
| `lifecycle/publication.go` | `64a2a48f79506500751ebf1278eaccf088e0d6f83377e0a2804df08cc9e6c173` |
| `lifecycle/reuse.go` | `7a0d94bbff1c92a60bce9c6b557023206d15ad3b205feb28ca32932d94c961f4` |
| `lifecycle/store.go` | `4b3358c97711c122084827e6f0c9680df3c5db1f1f62dc06fa2eaf86cbab0211` |
| `lifecycle/README.md` | `39b9b177b6e16ba5652b78fe6597f189b151115e1c96d97400a3e7ba760f5523` |
| `lifecycle/capture_namespace_test.go` | `c32dfaa5a267bd36a97577f21d0026e68a1e0904db8817730d1080ba4614ad14` |
| `lifecycle/order_contract_unix_test.go` | `ca15da7c00688c973da73d3f9828bf5c23fc8f5e7e46052c4c266b77611dc49c` |
