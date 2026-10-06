# Implemented capability profiles

This matrix describes current contracts and observed verification, not certification
of arbitrary host callbacks, external services or storage drivers. A capability flag
is admission data; the associated implementation and test scope remain essential.
Full task acceptance and both final independent audits are outstanding.

## Filters and admission

`access.Scoped` accepts mandatory scalar Eq/In predicates combined with And
(`filter/scope_profile.go`). NotEq, ranges, Or and Not are query operators, not
mandatory authorization policies. Fields/kinds must match each target's finalized
schema. `retrieval.PrepareRead` intersects mandatory, query and plan conditions;
unsupported admission is protected and classified before target payload work.

Optional query filters use schema-validated Eq/NotEq/In, ordered comparisons and
And/Or/Not where the declared field kind/operator permits them. In-memory leaves
use `filter.MatchCondition`/`filter.MatchIR`; wire adapters render the validated tree through their typed
walkers. This does not promise arbitrary engine collation, null, missing-field or
external driver semantics. Exact int64 JSON and integer membership transport
boundaries are separately verified in [wire-metadata.md](wire-metadata.md).

| Implementation | Scope and query execution | Storage / pinned publication | Lifecycle / inventory | Observed verification and limits |
|---|---|---|---|---|
| `dense/persistent.Adapter` | Thin catalog attributes matched before bounded payload reader; compatible dense space and native dot-product; bounded exact scan within declared MaxScanRecords | Local filesystem payload/catalog; pinned read required; exact reference/checksum; retained revision has no latest fallback | Stage/Inspect/exact cleanup plus fenced bounded inventory observer; durable publication belongs to Executor/store | `examples/conformance/persistent_read_unix_test.go` nine leaf and three planner cases; integer storage consumer cases; `dense/persistent` payload, inventory and process-crash tests; actual single/joint lifecycle. darwin/linux filesystem profile only |
| `tensor/persistent.Adapter` | Same admission boundary; typed token matrix space; native MaxSim; explicit bounded candidate set, exact within candidates, not exhaustive | Local filesystem payload/catalog; pinned read required; retained exact revision | Stage/Inspect/exact cleanup and fenced bounded inventory observer | Same external leaf/planner/integer suites; tensor candidate-loss oracle and actual persistent composition; `results/tensor-comparison.json`. Saved synthetic tensors prove no live embedding quality |
| `lexical/managed.Adapter` | Mandatory/query/plan filter before payload clone; captured readonly BM25 snapshot; exact binding fingerprint | Records in process memory; pinned read required; durable ledger does not restore missing records | Stage/Inspect/exact cleanup and fenced inventory observer; registered dispatch/exact support validation | Managed BM25 publication/tombstone/revocation tests; actual joint/bootstrap/cleanup suites. Fresh empty adapter returns unavailable until host restores records |
| `graph/managed.Adapter` and `Backend` | Node and edge scope before BFS expansion and projection; FindByIDs same admitted view; schema checked on both node/edge predicates; paging rejected; bounded depth/nodes/edges | Facts and explicit host bases in process memory; exact pinned inventory; missing records unavailable. Backend projects admitted facts with original supports and rank-only score | Stage/Inspect/exact source-support cleanup and fenced inventory observer; shared facts survive until final source support disappears, or an explicit host basis remains | Private bridge, unpublished/lost inventory, cycles/budgets, conflicts/revocation, support projection and host-basis tests; actual dense+graph integration. External consumer public nine-case read suite, three planner cases and eight direct FindByIDs cases now exercise this actual managed profile; arbitrary custom graph engines remain unverified |
| `lexical.BM25Index` | Scalar mandatory scope and broader schema-valid query filters; final freshness gate; atomic rebuild | Live in-memory profile. A captured readonly `BM25Snapshot` has a binding-specific pin; a general mutable index does not provide managed publication | Direct Index/Upsert are explicit host operations; managed source transitions use the separate managed adapter | BUG-002 reader/writer race and rebuild failure suite; external custom-adapter conformance reference uses actual BM25. No durable corpus claim |
| `adapters/elasticsearch.Store` | Declares scalar scope; filters rendered before injected Search and final delivery gate | Injected search client; current publication only (`PinnedPublication=false`); no retained cross-target snapshot claim | Raw index/storage operations; no managed Stage/Cleanup/inventory guarantee | Adapter module and exact integer wire tests. Actual external service behavior not certified |
| `adapters/qdrant.Store` | Declares scalar scope; typed filter translation before injected Search; final delivery gate | Injected client; current publication only | Raw dense/document operations; no managed lifecycle guarantee | Adapter tests and exact integer read/write/membership/invalid-batch cases; no live service claim |
| `adapters/pgvector.Store` | Declares scalar scope; parameterized schema-valid predicates before DB Query; final delivery gate | Injected DB port; current publication only | Raw dense/document operations; no managed lifecycle guarantee | Adapter tests and exact integer stored JSON/arguments/invalid-batch cases; no live database/driver precision claim |
| Raw graph Runner/store ports | Typed BYOT graph transport; raw traversal is not admitted managed traversal | Host-owned engine state; no implicit scoped/pinned profile | Raw mutation contract only | Their graph/adapter module tests do not confer managed output/traversal/publication capabilities |
| `documents.RawStore` | Explicit unscoped storage access; never a scoped hydration capability | Host storage retention/consistency | Raw deletion methods do not authorize lifecycle cleanup | `documents` contract separates RawStore from scoped hydration. Admission must happen through the scoped path before payload retrieval |

Persistent implementations require a local filesystem supporting flock, atomic
rename and directory fsync. Target lock and catalog/digest checks are part of that
profile. A process-crash test is not a hardware power-loss test or a distributed
transaction guarantee. `lifecycle/filestore.Store` durably persists bounded namespace
snapshots with generation CAS; it does not make volatile lexical/graph payload durable.

## Publication and joint profiles

| Contract | Supported behavior | Evidence / limits |
|---|---|---|
| Executor + captured publication | Dense, lexical, tensor and graph single-target; dense+lexical, dense+tensor, dense+graph joint target lists; Ready required before publication, expected source and generation CAS | `lifecycle/integration/joint_unix_test.go`, stage/operation checkpoint suites, single target suites. Publication is a logical index-artifact snapshot, not source-storage retention authority |
| Complete capture | Captures exact Ready inventory; tombstones excluded; empty inventory is pinned complete-empty | Publication and tombstone tests; no live fallback |
| Explicit partial capture | Freezes omitted target labels; unsupported branch skipped before I/O only under explicit profile; read coverage retained | Actual joint partial-publication/capture suites; strict mode refuses missing targets. Execution/freshness failures are not skippable capability failures |
| Bootstrap inventory | Complete and delta observation with exact artifacts/supports, bounded entries/records and nested mutation fences; unknown records not adopted/deleted | Actual persistent and volatile single/joint bootstrap suites. Confirmation is a point-in-time observation, not a lease for future mutation |
| Cleanup/recovery | Durable unknown-before-dispatch, one-shot Inspect reconciliation, exact registered retired inventory; shared support and newer-publication fencing | Actual restart/deadline/checkpoint/newer-publication/shared-graph suites. Host supplies scheduler and retry timing; no hidden background worker |

## Optional composition capabilities

| Implementation | Contract / boundary | Evidence / outstanding acceptance |
|---|---|---|
| Route/aggregate/fallback/rescue + explicit partial node | Full reachable-tree admission before planner/leaf payload; mandatory binding preserved; protection errors fail closed; partial coverage explicit | Core nested negotiation/partial/protection suites plus actual external joint_read profiles: dense+lexical/tensor/graph across aggregate/route/fallback/rescue, foreign/empty/unsupported plan, strict/partial negotiation, and revocation during actual dense payload I/O. Declared scalar Eq/In/And scope and common pinned publication; arbitrary external adapters require their own tests |
| Scoped hydration / `source.Reader` | Thin attributes/admission before payload; exact source/revision/representation/transformation; final freshness and retention check; batch all-or-nothing | Documents/source tests and actual PDF retained r1/r2/deleted/denied integration. Host owns storage and authorization decisions |
| Owned memory cache decorator | Keys include query/options/scope epoch/publication/config; owned snapshots; expiry/freshness before hit/delivery; failures and partial results not cached | Cache mutation/race/fake-clock/epoch tests and actual hit/miss partition fixtures for query Eq/In, plan filters/ranges/cache identity/text and graph seeds/direction/depth/node/edge/page; no external cache service guarantee |
| Text recipes | Explicit opt-in, bounded sequential stages, one ledger per Run, cloned BYOT snapshots, source-linked RRF contributors; no final answer generation | Scripted contracts plus actual BM25/HTTP protocol integration and repeated concurrent-attempt tests. Complete calibrated CLI comparison saved (20rows/30actualcalls); advisory bounds/unknown price/isolation limitations explicit; default baseline |
| Local graph expansion | Scoped model-free bounded BFS with shared attempt ledger and original supports | Actual managed graph/source tests; complete actual hybrid-vs-graph comparison saved; local support recall1→1 |
| Community/global summary | Host membership and original-source admission; bounded map/reduce; explicit shared ledger; derived support-only precision | Actual source reader and HTTP protocol tests, last-call concurrent budget test. Complete actual extraction/map/reduce comparison and external source-grounded rubric saved; community model output rejected, global original supports retained; negative quality recorded |
| Structured model transport | Optional one-POST adapter, strict executable response schemas, host tokenizer/pricing/model, reserved token limits and no hidden retry | HTTP request/privacy/refusal/malformed/usage/cancellation tests. Fixtures do not certify live tokenizer/model behavior |
| PDF parser adapter | Actual engine behind optional adapter; physical page index/printed labels, points/rotation, table/cell/image locators; OCR-derived/partial coverage explicit | Actual parser→chunk→BM25→resolve tests using synthetic PDF. OCR engine accuracy and blob durability remain host concerns |
| Evidence export/recording | Immutable strict schema, caller policy redaction, score/stage/revision association, optional labels; disabled/best-effort/required recording without rerunning retrieval | Schema and recorder privacy/failure/failed-journal/contribution tests plus immutable exports from actual persistent tensor candidate/rerank and actual parsed PDF persistent lifecycle/reopen. Saved pdf-record.json/tensor-record.json pass strict Go decode and independent JSON Schema. Tensor stages execute under one actual joint publication; tensor-recorded-run.json separately preserves actual CandidateBudget100, CandidateIDs order and dense candidate IDs with executable association validation. PDF parsing coverage is host-derived Partial/MissingEvidence, separate from complete branch admission. No implicit sink durability or correctness of judged labels |
| Public `contracttest` suite | Generic external module, explicit capabilities, admission/freshness/deadline/empty-denial/ownership observations | `examples/conformance` GOWORK=off reference and rejection tests plus persistent leaves. Actual managed graph Backend now also passes nine direct read and three planner cases, plus eight direct managed FindByIDs cases; actual joint_read mixed compositions are additionally verified; test scope is recorded in requirements.md |

Evidence checkpoints include [audit-regressions-test.txt](results/audit-regressions-test.txt), [external-joint-conformance-test.txt](results/external-joint-conformance-test.txt), [pdf-recorder-test.txt](results/pdf-recorder-test.txt) and [tensor-recorder-test.txt](results/tensor-recorder-test.txt). Historical mechanical full gates: results/joint-recorder-full-lint.txt and joint-recorder-full-test.txt. Current final gates after all consumer trace/accounting/Unicode fixes terminated exit0: results/cli-final-lint.txt and cli-final-test.txt. Independent current correctness overlay8cases and76actual-trace replays passed; final audits are linked in implementation-report.md.
These commands verify the current repository tests, not arbitrary external services or
unlisted external infrastructures. Later implementation changes require new checks.
