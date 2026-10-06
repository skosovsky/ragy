# T13 — independent correctness acceptance

Verdict: **PASS**. No remaining correctness findings in the T13 candidate.
Baseline: `4846cd00f71a0c031416b5f9856e146b2ca19e26`.
Reviewer: `/root/t13_correctness`; no implementation changes or commits made.
Acceptance criteria reviewed: **5/5 (100%)**. Assigned source rows reviewed: **28/28 (100%)**.

## Independent checks

- Fresh `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./lifecycle ./lifecycle/filestore ./lifecycle/integration`: PASS. All three packages, including actual durable/managed-index integration, completed; integration 91.375s. Log: `T13-correctness-race.log`.
- Fresh `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run ./lifecycle/...`: PASS, 0 issues. Log: `T13-correctness-lint.log`. Initial invocation without the GOCACHE override could not load packages and is not counted as a pass.
- Independent temporary overlay with fresh race: PASS. Both capture modes exercised pre-cancellation with zero Load, cancellation during successful matching Load with zero publication, joined Store transport/unavailable causes preserved, typed-nil Store rejected without dispatch, and admitted namespace with dangling publication/missing manifest rejected as protocol (independent of namespace mismatch). Overlay also reran capture, reserved-plan, fences and publication-pin regressions. Log: `T13-correctness-overlay.log`.

## Criteria and source audit

T13.C01: Published current guide consistently distinguishes source pointer, immutable read capture and registered durable metadata reservation. Namespace-wide generation, positive-only CheckReuse reload, operation-specific unknown recovery, one-step Cleaner/backoff and Complete scope match actual source.

T13.C02: Caller validation remains before Load; requested namespace survives Load and is compared before helper derivation. Strict/partial 16-case matrix covers matching/wrong namespace, unsupported schema and malformed responses across empty/nonempty fixtures, with zero invalid publication and no writes. Independent dangling-publication overlay verifies structural admission after namespace succeeds. Existing strict/partial/complete-empty and ownership tests remain valid. Store errors preserve their original causes.

T13.C03: Shared helper used by AcquirePublicationPin remains behind explicit namespace/schema admission; removing the tautology does not weaken that caller. New order regression uses real durable Store, rejects target/artifact reordering both through Prepare replay and CompareSwap, and proves exact unchanged snapshot/generation. Retired IDs/digests, live/released pin reservations, receipt monotonicity, owned containers, sorted simultaneous fences and synchronous exactly-once observer checks are unchanged and covered by fresh race suites. Nil context is explicitly unsupported package-wide; existing defensive rejection does not establish an additional guarantee.

T13.C04: Executor/Cleaner comments and guide correctly scope finite port-call counts without CPU/bytes/RSS guarantees. JSON clone/history validation/scans retained, so before/after optimization measurement is N/A. Filestore guide matches darwin/linux build profile, nonblocking flock, atomic rename/file+root-directory fsync and finite full-snapshot read/write byte limit; ancestor provisioning and hardware power-loss durability are expressly outside process-restart evidence. Trusted CAS is not described as authorization.

T13.C05: Fresh race covers lifecycle, filestore, real managed integration, restart/replay, ownership, reservations, cancellation/unknown checkpoints, exact inventory and fence contracts. No relevant test skipped is reported as pass. No native cross-namespace defect or additional F finding is claimed.

## Assigned source coverage

All dispositions were checked against the original task/review requirement, current implementation, retained tests and current guide, rather than accepting traceability text alone.

| Source row | Correctness disposition checked |
|---|---|
| `D15` | contract; source/code/tests/guide consistent, PASS |
| `D16` | change; source/code/tests/guide consistent, PASS |
| `D17` | contract; source/code/tests/guide consistent, PASS |
| `D18` | contract; source/code/tests/guide consistent, PASS |
| `D19` | retain; source/code/tests/guide consistent, PASS |
| `D20` | retain; source/code/tests/guide consistent, PASS |
| `D21` | retain; source/code/tests/guide consistent, PASS |
| `D22` | retain; source/code/tests/guide consistent, PASS |
| `lifecycle:01` | retain; source/code/tests/guide consistent, PASS |
| `lifecycle:02` | contract; source/code/tests/guide consistent, PASS |
| `lifecycle:03` | contract; source/code/tests/guide consistent, PASS |
| `lifecycle:04` | contract; source/code/tests/guide consistent, PASS |
| `lifecycle:05` | contract; source/code/tests/guide consistent, PASS |
| `lifecycle:06` | contract; source/code/tests/guide consistent, PASS |
| `lifecycle:07` | contract; source/code/tests/guide consistent, PASS |
| `lifecycle:08` | contract; source/code/tests/guide consistent, PASS |
| `lifecycle:09` | retain; source/code/tests/guide consistent, PASS |
| `lifecycle:10` | retain; source/code/tests/guide consistent, PASS |
| `lifecycle:11` | contract; source/code/tests/guide consistent, PASS |
| `lifecycle:12` | retain; source/code/tests/guide consistent, PASS |
| `lifecycle:13` | retain; source/code/tests/guide consistent, PASS |
| `lifecycle:14` | retain; source/code/tests/guide consistent, PASS |
| `lifecycle:15` | retain; source/code/tests/guide consistent, PASS |
| `lifecycle:16` | retain; source/code/tests/guide consistent, PASS |
| `lifecycle:17` | change; source/code/tests/guide consistent, PASS |
| `lifecycle:18` | contract; source/code/tests/guide consistent, PASS |
| `arch-docs:02` | retain; source/code/tests/guide consistent, PASS |
| `L-H01` | change; source/code/tests/guide consistent, PASS |

## Candidate SHA-256

Acceptance logs, bookkeeping and task-status files excluded from candidate digest. Candidate implementation/tests/contracts/current guidance frozen at the hashes below.

| File | SHA-256 |
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

Independent overlay source SHA-256: `f56a3c00a30828e9542566e75ff1b74d9017ea2e95eefb9afa377727d9bdc5ec`. Overlay was temporary and did not alter repository implementation/test files.
