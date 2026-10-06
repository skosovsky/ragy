# ragy lifecycle review — b63d5e19

Scope: lifecycle/{executor,publication,pins,reuse,cleanup,cleanup_validation,maintenance,bootstrap,inventory_verifier,manifest,filestore}; publication/pins/cleanup/reuse tests and docs/task18/lifecycle-maintenance.md. Source unchanged. Current design is intentionally a finite-step ingestion/publication protocol, not an agent harness. No scheduler, product policy, automatic retry, or implicit retention should be added.

## Confirmed hardening gap L-H01 (separate design item): Capture accepts a misrouted namespace from Store

References: lifecycle/publication.go:55–59, 68,77; contrast lifecycle/pins.go:42 and lifecycle/executor.go:309.

CapturePublication and CapturePartialPublication call store.Load(ctx, requestedNamespace), then discard requestedNamespace. publicationFromSnapshot sets `namespace := snapshot.Namespace`, and its `snapshot.Namespace != namespace` protocol check is therefore a tautology. A structurally valid snapshot for another namespace produces a successful pinned publication for that namespace. All neighboring loader paths retain and compare the requested namespace.

This is a protocol-response validation inconsistency at the pluggable Store boundary. The public Capture comments do not explicitly promise validation of malicious or misrouted Store responses; Store owns its namespace snapshot. Therefore count this as hardening/design, not as a supported-native runtime defect. The supplied filestore already rejects incorrect on-disk namespaces. This is NOT evidence that the built-in store crosses tenants, or that downstream access scopes can be bypassed. A misrouted/custom Store is required.

Repro: `/tmp/ragy-lifecycle-capture.go`; log `/tmp/ragy-lifecycle-capture.log`. From ragy: `GOCACHE=/tmp/ragy-review-go-cache go run /tmp/ragy-lifecycle-capture.go`. Store.Load("requested-tenant") returns a valid published snapshot in "other-tenant". Strict and partial captures both succeed and expose other-tenant target tuples; AcquirePublicationPin against the same Store returns ErrProtocol. CompareSwap is a panic sentinel and never runs.

Fix: check `snapshot.Namespace == requested namespace` immediately after Load (before deriving publication); retain shared helper only for callers that have already validated namespace. Remove tautological helper comparison. Keep malformed *response* errors ErrProtocol, distinct from invalid caller arguments.

AAA acceptance:
- Arrange valid wrong-namespace snapshots, empty and nonempty, and table of strict/partial capture; Act capture expected namespace; Assert ErrProtocol and zero publication, no writes.
- Arrange valid matching namespace snapshot; Act each capture; Assert identical successful behavior and immutable owned inventory.
- Arrange wrong schema/malformed response; Assert ErrProtocol consistently; argument validation still returns ErrInvalidArgument.

## Separate architectural/naming/complexity decisions (not bugs)

L-D01 — Keep Executor and Cleaner. They implement publication and managed-index cleanup invariants; removing them as “orchestration” would force services to recreate atomic CAS and unknown-outcome protocol. Do not grow into flowy or agent-harness scheduling. Reference executor.go:51, cleanup.go:77–91.

L-D02 — Clarify publication terminology in current API docs. lifecycle.Publication is a source→manifest pointer; access.Publication is an immutable read observation; PublicationPin is a durable metadata reservation. These are three distinct concepts with similar names. Consider SourcePublication / PublicationSnapshot / MetadataPin in a clean break, or provide one diagram/glossary without gratuitous rename churn. References manifest.go:89, pins.go:12, maintenance.go:22.

L-D03 — Keep Capture versus Acquire distinction explicit in quickstart. Value capture is not a discoverable lease. Acquire protects metadata only; it cannot ensure availability of target payloads or source bytes. No TTL or cleanup blocking should be inferred. docs/task18/lifecycle-maintenance.md already documents this correctly; propagate to top-level lifecycle examples.

L-D04 — Namespace-wide CAS is deliberately conservative. Pin/release or an unrelated source mutation can conflict with CheckReuse/Prepare/Publish. Document caller retry/reconcile recipes, not an internal fallback retry loop. Large-host storage adapters may optimize internally while preserving generation contracts. reuse.go:78–84; store.go.

L-D05 — CheckReuse's comment “Any concurrent lifecycle generation change invalidates the decision” overstates behavior: early Absent/Changed/Incomplete returns do not reload. Only a positive Confirmed decision has the second-generation check. Narrow the wording to positive reuse confirmation; no need to make negative observations more expensive. reuse.go:29–31,57–85.

L-D06 — CheckReuse is an observation point, not a permission or lease. Even positive reuse can become stale immediately after final Load. Host must control lifecycle publication and subsequent use. Preserve explicit comment and require integrations not to cache CanSkip indefinitely. reuse.go:20–27.

L-D07 — ErrOutcomeUnknown also wraps failures from read-only Inspect/InspectCleanup/CheckReuse. This is defensible for unknown target state, but error wording “target outcome unknown” can be misread as proof that a destructive operation was issued. Document operation-specific recovery table; avoid a generic retry middleware. reuse.go:118–123; cleanup.go:265; executor.go:495.

L-D08 — Keep one-step Cleaner, explicit host clock, backoff and overdue recovery flag. Backoff values are schedule outputs, not sleeping retries. Consider named recovery mode instead of bool only if extra modes arise; do not build a worker inside ragy. cleanup.go:77–100,171–235.

L-D09 — Scope “bounded explicit operations” accurately: bounded *number of port calls* is not bounded CPU/bytes. Snapshot validation, cloning, retainedArtifacts, ancestry and full snapshot serialization scale with namespace history. Existing maintenance scale docs correctly acknowledge this. Link those limits from Executor/Cleaner public package docs; don't claim constant-work maintenance. executor.go:51; cleanup.go:396; maintenance.go:231; filestore/store_unix.go:147.

L-D10 — Retired skeletons, artifact digests, released IDs, bootstrap receipts are intentional permanent reservations. Do not “simplify” by deleting them on TTL; that permits ABA/rebinding and breaks replay. At capacity host must move profile/store or explicitly migrate, not evict silently. maintenance.go:15–24,134,304; docs/task18/lifecycle-maintenance.md.

L-D11 — Distinguish operation state Complete from total historical data destruction: pins protect metadata, target cleanup is separate, and source bytes are host-owned. Cleaner Complete means all captured target-cleanup items confirmed; it does not certify privacy erasure or tenant deletion. cleanup.go:31,315–330.

L-D12 — Existing full-snapshot JSON roundtrip in CompactHistory is an ownership technique, not unexplained fallback. Keep unless profiling supports an explicit deep clone shared with other clone paths. A rewrite must preserve nested slice ownership and nil/order semantics used in replacement validation. maintenance.go:145–156; executor.go:453.

L-D13 — Plan equality is order-sensitive (JSON samePlan, target/artifact slices in sameReservedPlan), while publication target selection sorts input. Decide/document whether ingestion target/artifact order is semantic to idempotency. Do not silently canonicalize already-reserved plan identity during upgrade. executor.go:465; maintenance.go:272; publication.go:95–99.

L-D14 — Store.CompareSwap is a trusted low-level checkpoint interface, not a public authorization boundary. ValidateReplacement preserves immutable reservations and certain monotonic receipts, but does not validate every transition as if calls came through Executor/Cleaner. Document this host/adapter responsibility; do not claim direct arbitrary snapshots are equivalent to authenticated workflow commands. maintenance.go:304–324,425–440. No demonstrated supported Executor path bypass was found.

L-D15 — ValidateReplacement compares cleanup/pin/receipt history with scans; direct correctness wins over adding a second persisted index. If workload demands optimization, build temporary indexes within validation and verify differential equivalence. No new service/storage framework merely to optimize local profile. maintenance.go:442–507.

L-D16 — FencedInventoryVerifier acquires observer fences in sorted target order and requires callback exactly once. Keep these anti-deadlock/protocol checks. Observers are cooperative synchronous ports; do not introduce goroutines/timeouts to forcibly detach callbacks. Document that target/source snapshots after the verification point are host-governed. inventory_verifier.go:13,32–41,80–107.

L-D17 — There is no standalone lifecycle README; current migration/retention contract lives in docs/task18. Move or mirror stable guidance into lifecycle/README.md with state transitions, legal replay, failures and recovery, error taxonomy, pin distinctions, local-filesystem profile and capacity limits. Keep historical task docs as evidence instead of requiring users to discover runtime contracts through task numbers.

L-D18 — Consistent nil context behavior is optional API hygiene, not a supported-call correctness bug: pins explicitly reject nil while Capture/Executor/filestore use ctx.Err and ordinary Go convention disallows nil contexts. Choose a package-wide documented convention; do not count every missing nil check as a distinct defect.

## Positive contracts / rejected suspicions

- Stage persists unknown checkpoint before dispatch; after uncertain dispatch it requires explicit Inspect, not blind Stage replay.
- Cleanup likewise persists unknown before destructive call; Confirm uses current CAS and exact IDs.
- Partial capture excludes an entire missing target branch; all-excluded differs from complete-empty.
- Metadata pin replay retains exact original inventory after publication advances; released IDs cannot revive.
- Cleanup and metadata compaction are separate, with reference closures and exact artifact fences preserved.
- No confirmed bug in legal concurrent lifecycle state transitions emerged in this scoped audit. The hardening gap uses a malformed Store response; there are no promoted supported-native runtime defects in this report.

## Verification limits

Ran the minimal compiled repro against current repository via go run, no repository edits. Did not run full test suite, race suite, or external backends; parent handles global validation. Reproduction output may contain diagnostic successful exits; success means the faulty behavior was observed, not that expected contract assertions passed.
