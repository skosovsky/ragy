# Task12 clear-break migration guide

This guide describes the implemented contracts and required host-code changes for all seven issue cards. It is an API migration guide, not a completed acceptance certificate. Live text/graph quality experiments and independent final audits must be confirmed in the verification report before acceptance or issue closure.

## Required host-code migration sequence

1. Capture the host authorization decision before planning; use mandatory scoped binding, freshness authority and an explicit publication. Choose unrestricted current reads explicitly only for that host-approved profile.
2. Update custom backend, binder, projector, processor and hydration signatures to preserve the captured context/binding. Declare capabilities and execute the public conformance suite; raw storage lookup is not scoped retrieval.
3. Replace manual cross-target upsert/delete coordination with Executor prepare/stage/reconcile/publish and Cleaner operations. Supply namespace, exact revisions, fingerprints, expected publication and stable idempotency keys; retain the durable CAS ledger and host scheduler.
4. Bootstrap only from fenced confirmed complete/delta inventory, or rebuild into a managed namespace. Preserve unknown records. Handle conflict, unknown outcome, snapshot unavailable, partial and cleanup overdue explicitly.
5. Update numeric producers, thresholds, fusion and exporters to declared ScoreState/ScoreSemantics. Native values are not implicitly normalized. Use explicit RRF or a selected normalization policy for heterogeneous scales.
6. Supply tensor model/configuration/space identity, normalized token matrices and candidate budget. Configure optional text recipes with explicit typed planner/assessor, quotes, ledger and deadline; baseline requires no model.
7. Provide graph schema, entity identity/aliases/conflict policies and exact community membership. Preserve original supports and invalidate/recompute derived summaries after source mutation or revocation; names alone do not identify entities.
8. Consume immutable evidence rather than reconstructing an old attempt from current logs/index state. Configure identifier/numeric/text/location allowlists and recording mode. Ground-truth labels, quality graders and artifact retention remain external.
9. Update source projectors/renderers to exact revision/representation locators and UTF-8 byte coordinates. Preserve original/derived distinction and all contributors through grouping/dedup/truncation, or declare precision unavailable.
10. Reparse documents with unknown historical geometry from retained authoritative original bytes. Until then expose only document-level evidence; never invent page/word/cell/image coordinates. Publish rebuilt typed projections with changed transformation identity through lifecycle, verify scoped retrieval/resolution and keep old r1 source material under the host retention policy until no retained citation needs it. A missing/denied r1 returns unavailable even when r2 exists.
11. Remove replaced wrappers/types and migrate compile-time consumers, fixtures and persisted payload tags. Clear break is checked through compilation and conformance, not compatibility aliases.
12. Reindex rounded integer metadata from exact authoritative values. Recover an earlier lost/partial BM25 corpus with a successful authoritative rebuild and keep both integer precision and atomic rebuild regressions in consumer tests.

For parsed layout, preserve source coverage in typed payload metadata and its manifest fingerprint. Admission ReadCoverage describes branches/inventory; it does not certify complete parsing/OCR. A consumer exporting a partial parsed attempt reports Partial/MissingEvidence from the retrieved coverage while retaining complete read admission when all requested targets were admitted.

## Integer metadata

JSON metadata decoded as `filter.RawAttributes` now retains numbers until schema normalization. Integer fields round-trip without a float64 intermediary. Graph metadata, retrieval metadata, and persisted JSON decoding share this boundary. `elasticsearch.Hit.Source` now uses `filter.RawAttributes`; client implementations should decode directly into this type, rather than first decoding identifiers into a `map[string]any` through float64. Already rounded numeric IDs cannot be reconstructed by the codec.

If previous ingestion persisted corrupted numeric metadata, rebuild the affected records and filter/ACL attributes from authoritative source data. Do not guess the original ID from its rounded value. Validate adjacent integer IDs and the complete signed int64 range before considering data recovered.

## BM25 replacement

`BM25Index.Index` builds replacement state separately and only publishes after the entire replacement succeeds. Failed validation, tokenization or codec projection keeps the previously published corpus and statistics. Rebuild and Upsert serialize at the same lock; a successful `Index(nil)` intentionally clears the corpus.

BM25 configuration rejects NaN, infinity, negative K1/B and B above one. Zero K1/B select defaults; explicit K1 must be positive and B within (0,1]. Invalid computed non-finite scores fail with protocol error and no results. Update callers that previously relied on negative values silently selecting defaults.

If an earlier failed rebuild already emptied or partially replaced an in-memory index, restore it by rebuilding successfully from authoritative documents. Updating the code does not restore missing records.

## Migration order

Update read bindings and score producers first, then source identities/retention and lifecycle targets, followed by optional recipes, evidence export and locator consumers. Rebuild derived records when their original identity, coordinates or transformation cannot be recovered. Replaced contracts have no compatibility wrappers. The sections below describe the working APIs; capability declarations and experiment acceptance are tracked separately.

## Tensor matrix and space contract

Every tensor record now supplies `tensor.Space`: model identity, model revision,
configuration identity (including tokenization/preprocessing), vector-space identity
and dimension. The host must populate these from the embedding pipeline; equal
matrix dimensions do not justify inventing a common space for different models.

`Record.Validate` rejects empty/ragged/non-finite/non-unit matrices and incomplete
spaces. No implicit normalization is performed. Recompute or explicitly normalize
old tensors in the host embedding pipeline, record the resulting configuration,
and rebuild the affected index. Unknown historical identities require backfill,
not guessed compatibility.

`tensor.MaxSim` returns native scores; `tensor.Rerank` keeps their declared
semantics and candidate-local rank. Do not clamp these scores to [0,1] or add them
to dense/sparse scores without an explicit comparable-scale policy. The supplied
candidate set can miss the best document. Preserve `CandidateIDs` and the budget
in evidence, even when TopK hides some candidates.

For durable retrieval use `tensor/persistent` and the pinned candidate path described below. Its reopened-file integration verifies native MaxSim, bounded candidates and publication admission. The saved-vector experiment measures this reference profile; it does not qualify another embedding model or corpus.

## Scores and thresholds — clear break

`Document.ScoreState` now defaults to `ScoreAbsent`. Every numeric producer must
set `ScorePresent` or `ScoreNormalized` plus a nonempty `ScoreSemantics` identifying
a comparable algorithm/model/configuration scale. Never reuse the old numeric enum
ordinals in saved payloads: migrate tagged source records from authoritative data
or rebuild their derived retrieval artifacts. Absent score means no numeric evidence,
not numeric zero. Native values can be negative or exceed one. Normalized values
require [0,1] and an explicitly selected policy.

Replace `RetrieveOptions.MinSimilarity` with `Threshold: &ScoreThreshold{Value: ...,
State: ..., Semantics: ...}`. Nil means disabled; zero/negative values are real
thresholds. A threshold cannot be applied to rank-only evidence or another scale.
There is no compatibility alias or silent coercion.

Remove assumptions that search adapters or BM25 return logistic/clamped relevance.
They retain native values. `ClampScore` was removed. Configure an explicit host
normalization policy when a consumer truly requires normalized values; do not
silently truncate invalid policy output. `RankToScoreNormalizer` implementations
must now expose `Semantics()`.

Numeric merge, default grouping and numeric TopK require matching state/semantics.
Use explicit rank fusion for heterogeneous sources. RRF identifies its selected
relative-max rank policy and configured k. Caller-comparator `Rerank` now emits
rank-only ordering so terminal TopK does not undo the comparator. Original numeric
observations are retained in `ScoreHistory`; update custom renderers/exporters to
preserve contributor IDs, values, states and semantics. History slices are copied;
this does not establish deep ownership of arbitrary BYOT domain metadata.

Score history alone is not an evidence record. Use the binding-aware `evidence` recorder described below for stage observations, exact original-source associations and explicit privacy policy.

## Read access and publication binding — clear break

Every retrieval request now supplies `Read`. A zero binding is invalid and is
rejected before planning or target dispatch. Select `retrieval.UnrestrictedRead()`
explicitly for live unrestricted reads. This profile provides no pinned publication
or cross-target revision guarantee.

For protected reads construct `access.Scoped` from a schema-checked mandatory
predicate, immutable policy snapshot, expiry, host freshness authority, injected
clock and publication. The reference mandatory profile supports scalar Eq/In/And.
Optional query/planner filters are intersected per target schema; they cannot
replace mandatory constraints. Epoch revocation is checked through the host authority
before TTL expires. Expired bindings require a new trusted host decision/attempt;
ragy does not grant fresh privileges by rewriting an old snapshot.

Planner/binder/projector request copies retain the binding captured at entry.
Move authorization out of mutable request metadata/query filters. Update custom
backends to declare `ReadCapabilityProvider` and enforce the effective filter before
loading/exporting payload. Declaring capabilities is not sufficient certification;
public external-adapter conformance is described below. Scope/publication
errors are `access.ProtectionError` and must not be rescued into ordinary success.

Freshness failure suppresses final result sets and BYOT execution outputs, including
partial/error returns. Keep host authority/clock implementations safe for concurrent
fan-out. Authority failure is fail-closed; do not introduce hidden policy refresh or
fallback to unrestricted reads.

`access.PinPublication` copies a logical inventory; capture actual Ready target revisions through lifecycle before constructing the trusted binding. Managed dense/tensor/lexical/graph adapters support pinned reads with the capability limits listed below. Raw transport stores declare current-only reads and reject a required pin. Scoped graph traversal admits nodes and edges before expansion. Explicit partial negotiation skips only configured unsupported branches and retains coverage; external mixed-path conformance is exercised by the supplied actual compositions and public adapter suite; it certifies those declared profiles, not arbitrary external backends.

## Context and binding through processors/model reranking — clear break

Implement `PostProcessor.Process(ctx, read, rs)` instead of `Process(rs)`.
`PostProcessorChain.Process` now receives the binding separately from tuning
options: `Process(ctx, read, opts, rs)`. Pass the original trusted binding;
construct a new unrestricted binding only for an explicitly unrestricted operation.
Processors must propagate the supplied context/deadline to all I/O and check the
binding immediately before external payload consumers. Do not copy authorization
into mutable request metadata or replace an expired binding inside a processor.

Query-aware rerankers implement `Rerank(ctx, read, query, rs)`. The shipped model
adapter checks required freshness before dispatch and at every return, including
HTTP/protocol errors. Observability forwards the same context and binding. Update
custom wrappers and fixtures to this one contract; legacy signatures are removed.

The chain checks before/after each processor and before final delivery. Revocation,
expiry, cancellation or a protected failure suppresses result payloads and stops
later processors. Ordinary errors may preserve partial results only while the same
binding remains valid. Joined sibling errors are discarded when carrying a
protection failure, preventing unrelated typed partial payloads from escaping that
boundary. The host authority still owns authorization decisions and must not return
sensitive payloads as authority-error causes.

Pass the same captured binding to evidence recording and recipe-owned model calls. These boundaries check freshness before callbacks and final delivery. The optional cache path is described below.

## Optional cache configuration and ownership

Use `NewCachedBackend` with an explicit `CacheConfig`, including `SnapshotRequest`,
`HostIdentity`, `Identity`, `CloneMeta`, `Now` and a positive TTL. The snapshotter
must independently own BYOT intent/request metadata and planned intent; core option
slices/pointers are copied by the decorator. Do not concurrently mutate inputs
during request capture. Supply thread-safe ports and a metadata cloner that owns
all nested mutable values; a shallow map copy is insufficient for nested metadata.

HostIdentity must include every host-owned request value affecting retrieval.
Identity must include the actual index/revision, recipe, configuration and requested
capabilities. Rotate the index revision when live content changes. Identity is
checked around cache I/O and target retrieval; do not make its callback itself
refresh authorization. Supply the original immutable binding on every request.
At authorization expiry obtain a new binding from the host, even if a cache entry
is still available. Authority failure/revocation is fail-closed with no refresh.

Use `NewMemoryCache` with explicit bounded capacity, clock and the same metadata
ownership policy, or implement `ResultCache` with independent Store/Load snapshots.
Partial/failed retrieval is not cached; ordinary storage failures are returned
without retrying target retrieval. This live cache does not provide a durable
publication snapshot; unsupported pinned targets continue to reject execution.

Core snapshot/target revision, page, range and threshold fields now carry explicit
lowercase JSON field names. Update consumers that serialize these structs using
their former Go field-name keys. No legacy wire aliases are provided.

## Composition admission before dispatch

Custom execution nodes used with scoped or pinned bindings must implement
`RequestReadAdmission[TIntent, TRequestMeta].AdmitRead(ctx, req)` returning
`(ReadCoverage, error)`. Negotiate every
reachable leaf, including fallback/rescue/default and conditional branches, before
dispatch. Use `PreflightRead` for nested nodes and `PrepareRead` for declared backend
providers. Admission must not run planners, predicates, model calls or target
payload I/O, mutate requests, or refresh authorization. Required host freshness
validation is permitted and remains fail-closed.

Built-in composition negotiates all configured reachable branches. A currently
unselected branch with an unsupported target still rejects the strict profile;
route or conditional selection is not a way to bypass admission. Pipeline admission
runs before planning and again after binding; target enforcement runs at dispatch.
Unknown custom nodes are rejected before the planner runs for scoped/pinned reads.
Explicit unrestricted live operations remain explicit unrestricted operations.

Scope failures cannot be rescued into success. Route policy revocation stops
decision recording and subsequent rescue predicates/targets, suppressing revoked
payload and execution outputs. Use the explicit partial profile described below;
do not turn strict failures into implicit partial success in consumer code.

## Explicit partial branch and coverage — clear break

Wrap a branch in `PartialReadNode` (or `RequestPartialReadNode`) only when the host
explicitly permits omitting that branch for unsupported read capabilities. Give it
a stable configuration label in `Name`. Labels contain only letters, numbers,
underscore, hyphen or dot, up to 128 characters. Never derive labels from source,
document, policy or request values. The original binding is still required.

Only direct typed admission failures from `access.UnsupportedCapability` are
skippable. Use this classification solely for unsupported negotiation before target
I/O. Do not classify runtime errors, authorization decisions or cancellation this
way. Host authority errors are made non-skippable even when their cause is
unsupported. Joined admission errors are also non-skippable. Generic
`errors.Is(err, ErrUnsupported)` is insufficient to decide whether to skip.

Custom `AdmitRead` implementations return `CompleteReadCoverage()` only after all
reachable leaves have negotiated successfully; inspect nested nodes with
`InspectRead` and retain their reports. Unobserved coverage is rejected as an
admission success. Scope gates and actual target enforcement are still mandatory.
Update execution result constructors for the new `RetrievalResult.Coverage` field.
Direct backend interfaces do not become partial-capability execution interfaces.

A successful partial result may be empty. Check `Coverage.State()`/`IsPartial()`;
a nil error does not imply complete coverage. Built-in public composition retains
coverage for all configured reachable branches, including unused optional fallback
branches. `SkippedBranches()` returns an independent slice of static labels.
Protected delivery failure clears both payload and coverage/side outputs.

Coverage JSON carries its schema identity and distinguishes `unobserved`,
`complete`, `partial` and `unrestricted`. Unknown schema/state, unknown fields,
missing required fields, duplicate/invalid branch labels and inconsistent state
are rejected without mutating the previous decoded value. Consumers should retain
this immutable report in their own envelopes until the forthcoming evidence
recorder path carries it automatically. JSON Schema and a roundtrip fixture are in
`docs/task12/schemas` and `docs/task12/fixtures`.

## Public custom-adapter conformance

Replace imports of `github.com/skosovsky/ragy/internal/contracttest` with the public
`github.com/skosovsky/ragy/contracttest`. The internal package was removed; no
compatibility re-export remains. Existing codec/index/graph/pipeline/processor
helpers moved to the public package; shipped consumers were updated.

For scoped backend certification, implement a fresh generic `ScopedReadFixture`
factory and call `RunScopedReadSuite(t, factory)`. Supply host-owned intent/request
metadata/source metadata types, a valid Eq/In/And binding, a corpus with allowed
and forbidden payloads, contradictory/unsupported query conditions, host
revocation/expiry hooks and actual payload I/O/materialization observations. Record
loaded IDs before any output projection/filtering; observations reconstructed from
final output cannot establish enforcement. Do not reuse mutable fixture state
between scenarios. Scope gates must exist in the raw adapter, not only in an outer
execution wrapper. Custom adapters declare their capability/schema contract and
use PrepareRead/DeliverRead around I/O as appropriate.

The checker rejects incompatible declarations before dispatch and checks direct
leaf enforcement, pre-I/O cancellation/authority denial/expiry, revocation during
I/O and exact incoming deadline propagation. It reports stable violation codes
without forbidden payload IDs/counts or raw adapter errors. Read these QA reports
as fixture-specific evidence, not runtime permission decisions or universal
certification. Hosts own policy and target-specific persistent integration.

The independent consumer module is `examples/conformance` with a module namespace
outside ragy. Run it with `GOWORK=off go test -race -v ./...` from that directory.
It proves public accessibility and BYOT composition without internal helper imports,
including rejection of postfilter-only and premature-I/O implementations.

## Scoped exact-revision hydration — clear break

The old storage contract is now `documents.RawStore[TMeta]`, with no alias under
`documents.Store`. Its lookup/delete methods are explicitly raw storage operations,
without scope/publication/retained-revision guarantees. Update raw storage consumers
and interface assertions to the new name; never use RawStore as a scoped hydration
fallback. Existing adapters retain raw access for host storage operations.

For supported hydration use `documents.NewHydrator` and `LookupRequest`, carrying
the original binding and exact `source.Reference` values. Each reference identifies
namespace, source, revision, transformation/access fingerprints, artifact and source
representation. Do not substitute latest or reinterpret representations on a miss.
Pinned publication inventory must match the requested target/revision/fingerprints
before even permission catalog I/O is allowed.

Provide a typed thin Catalog, exact-reference PayloadLoader, finalized schema,
TAccess-to-scalar-attributes projector and explicit BYOT payload metadata cloner.
Catalog descriptors contain permission metadata only, with owned immutable snapshots;
they cannot load source content. Attribute projection and metadata cloning must be
pure/thread-safe, without hidden I/O. Loader must materialize only the admitted exact
references, preserving identities and caller context. Catalog/loader requests are
separately captured so port mutations cannot alter the next boundary.

The complete batch is admitted before the first payload load. Missing/deleted/denied
entries fail closed without identifying the rejected source in diagnostics. Payload
results are validated as a whole before metadata consumers; unexpected/duplicate
identities, revision substitution or mismatched document IDs produce protected
protocol failure. Partial/error loader returns expose no payload, and no retry or
latest fallback occurs. Host ports must never place payloads in error causes.

Source retention and original blob durability remain host responsibilities. Exact locator resolution uses `layout.Resolver` or text hydration; managed target storage uses the lifecycle adapters described below. Generic hydration does not certify an arbitrary raw store for retained reads or publication.

## Locators, retained text resolution and artifact rendering

Use `source.Reference` for exact source revision/representation identity and
`source.Locator` for text/page/region/cell/image locations. Text spans use zero-based
half-open UTF-8 bytes, with code point boundaries, in the named representation;
they are not PDF binary offsets. Physical page index and printed label are distinct.
Supply geometry in points from the unrotated top-left page and explicit clockwise
rotation. Do not infer invalid geometry or split one merged logical cell into
multiple independent citations.

Use `source.OriginalText` with retained text when exact mapping is available.
Use `DerivedText` for context or descriptions and retain all supports. Truncate
with `MappedText.Slice` and combine with `JoinMapped`; neither operation shifts the
original coordinate system or attributes derived separators to a source span.
Use `Hydrator.ResolveText` for exact retained text citations. For page/region/cell/image kinds use `layout.Resolver` with typed retained original payloads; the text resolver explicitly rejects those kinds. Missing/deleted/denied revisions have no latest fallback.

Replace `ArtifactRenderer.Render(rs, opts)` with `Render(ctx, read, rs, opts)` using
the same binding as retrieval. Supply `CloneMeta` with a real deep copy of mutable
host metadata. Callbacks must not embed payloads in error causes. `Mapping` returns
owned source mapping and cannot be combined with the string-only `Snippet` callback.
When no mapping is supplied, precision remains unobserved; display provenance is
not evidence of exact geometry or bytes. Budget units remain Unicode code points,
while mapping intervals remain UTF-8 bytes. Consume snippet `Supports` to retain
all dedup contributors. A custom formatter receives isolated copies.

Preserve mappings/supports through store projection, grouping and reranking. The actual PDF path tests parser output, typed layout resolution, original/derived rendering and persistent dense lifecycle publication. Reparse old documents with unknown coordinates; no API change certifies historical raw records automatically.

## Optional PDF/layout parser profile

Configure `adapters/pdf.New` with an external Python interpreter, required
transformation fingerprint and explicit input/output/page/element/time limits.
Optional parser dependencies are confined to that interpreter; core has none.
Supply already authorized retained bytes with the exact revision and `pdf-binary`
representation. Include engine/configuration/normalization/dependency identity in
the host fingerprint; do not reuse it after changing the transformation.

Consume the validated `layout.Document` envelope and propagate partial coverage
and typed diagnostics into host source/chunk metadata. Normalized page text has
its own representation and UTF-8 spans. Keep original image and cell identities
separate from derived descriptions/OCR. A limited parse is partial even
when its emitted pages were individually processed. Unsupported crop/rotation/
table-grid geometry is an explicit failure. Choose another declared adapter/profile
rather than guessing coordinates or treating missing OCR as complete coverage.

The actual parser integrations cover page/cell/image projection, BM25 retrieval, scoped text/layout resolution and original/derived artifact rendering. A durable integration publishes parsed r1/r2 records through `dense/persistent` and Executor/filestore, then reopens both index and ledger. Partial coverage is host typed metadata bound to the payload fingerprint. Original layout/blob retention in the fixture is in memory; it does not establish a durable blob service. OCR unreadable observations are explicitly simulated, not an OCR quality experiment.

## Shared typed source.Reader materialization

Document-specific permission/load contracts were removed rather than aliased.
Move implementations from documents.LookupRequest/Descriptor/PayloadLoader/Hydrated
into source.LookupRequest, source.Descriptor, source.Loader and source.Materialized.
Materialized.Payload carries your own typed value; it is not restricted to a
retrieval document/string. For document hydration that value is
retrieval.Document[TMeta]. Update loader signatures and access .Payload instead of
.Document. documents.Hydrator now delegates all admission to the same source.Reader.

For binary, cell/layout or other source payloads, construct source.NewReader with
ReadConfig[TAccess, TPayload]. Provide thin owned permission metadata, exact-reference
loading, schema/attribute projection, payload validation and a real payload clone
for mutable bytes/slices/maps. Callbacks must be pure/thread-safe and must never
embed payloads in error causes. Full returned identity/count admission happens before
any payload consumer. Validation/cloning and every delivery have freshness gates.
Do not convert image bytes to strings to reuse document hydration. Storage,
retention and truthful association with the requested reference remain host-owned.

Shared admission does not itself validate geometry. Use `layout.Resolver` to validate exact original geometry against the admitted retained payload. Image/cell UI highlighting, blob storage and authorization decisions remain host responsibilities.

## Retained layout resolution

Use layout.NewResolver with source Catalog/Loader ports carrying layout.Retained.
The host retains canonical original locations, original text/media, word geometry,
cell extents, observed coverage/diagnostics and optional separate derived mapping.
Validate exact-reference loading and supply truthful payload/geometry association.
Do not replace original image bytes with a description or base64 document string.

Resolve(ctx, layout.ResolveRequest) uses the original read binding. Text offsets
validate against retained normalized UTF-8 text; page/region geometry must agree
with the retained page; cell identity must match exactly. Image region must lie
inside the original extent. Region text consists of intersecting word evidence
with original spans, rather than a fabricated mapping for the entire page.
Empty region evidence has no text mapping. Media bytes are original, not implicitly
cropped; UI highlighting/cropping remains external. OriginalRegion preserves cell
or image extent separately from the requested location. Mutable bytes/diagnostics
are independently copied, and partial coverage survives resolution.

Derived descriptions must use support-only DerivedContent and reference the same
canonical original artifact. The source host owns generation/retention and must not
place payloads into error causes. Missing/denied/revoked/inconsistent entries fail
the whole batch without latest substitution or partial export. Use explicit OCR observations and layout projection below. The durable PDF integration verifies managed indexing/publication and retained r1/r2 resolution; original source retention and OCR accuracy remain host responsibilities.

## OCR observations and layout projection

Provide explicit layout.OCRObservation for the exact original image and the host
OCR transformation fingerprint. Recognized carries derived text; unreadable and
unsupported carry none; zero/unobserved is not a successful OCR outcome. ApplyOCR
returns an owned document and retains partial coverage/original page text/geometry.
It replaces the image's canonical OCR diagnostic and never invents PDF byte spans.
OCR accuracy and engine implementation remain external. Simulation is conformance
evidence and must be reported separately from actual parser/engine execution.

Use layout.Project with the original read binding to obtain typed evidence before
host metadata/index projection. Page spans are exact normalized UTF-8 mappings;
logical cell quotes use SupportedOriginalText with original support and unavailable
byte precision; image OCR/descriptions use derived support-only mapping. ImageText
is optional and gated before/after; it must return evidence supported by exactly
the image supplied. Unreadable/no-description images do not become fake searchable
text. Carry Projected.Coverage/Diagnostics and source supports in your own stored
metadata/artifact association. Host chooses the index and owns retention. For managed indexing preserve both artifact references and original supports in the manifest, and bind projected coverage/diagnostics into the typed payload fingerprint. The actual PDF durable integration exercises this association without introducing PDF-specific fields into generic lifecycle contracts.

## Durable lifecycle inventory

Hosts implementing lifecycle persistence use lifecycle.Store with owned namespace
Snapshots and expected generation CAS. The optional lifecycle/filestore adapter
requires a local Linux/macOS filesystem supporting flock, atomic rename and fsync;
it is not a shared-network-filesystem guarantee. Host storage directories, backup,
retention and scheduling remain external. Keep all old managed artifact/support
inventory; replacing chunk IDs alone is insufficient for future cleanup.

A CAS error after I/O may have committed: load and reconcile durable state before
retrying. ErrConflict is explicit contention/stale generation, not permission to
publish without checking the active source. Use the Executor and fenced bootstrap contracts below in addition to this persistence port. A durable manifest alone does not prove a backend artifact exists or certify a volatile target after restart.

## Explicit ingestion/publication operations

Register typed StagePort targets in lifecycle.Executor. Provide owned payload
capture and a validator that checks the payload against the manifest fingerprint.
Prepare persists the complete planned inventory before I/O. Reuse the same
idempotency key only for the entire unchanged plan; changing expected publication,
source fingerprints, target profile/inventory or payload requires a new operation.

Call Stage explicitly for each target. ErrOutcomeUnknown requires Reconcile,
which invokes Inspect once; do not repeat uncertain writes blindly. Target ports
must stage revision-bound records outside published read visibility and inspect
actual outcomes. Publish uses expected active source publication plus durable
namespace generation CAS. A published manifest is frozen against late stage writes.
Errors after publication CAS may mean committed state: Load/replay reconciles;
neither cancellation nor timeout means rollback. Scheduling and target retry policy
remain host-owned. Use Cleaner for explicit cleanup and capture a publication before all pinned target reads. Dense/tensor targets reopen durable payloads; lexical/graph reference targets retain in-process state and report unavailable after inventory loss.

## Cleanup dispatch and source acknowledgement clock

Published manifests carry PublishedAt from ExecutorConfig.Now (default wall clock;
use an injected clock for deterministic workers). Cleaner receives explicit deadline
and capped backoff policy. Begin persists full retired manifest/target inventory;
Attempt runs one due operation and never sleeps. Unknown cleanup requires Reconcile,
which only inspects. Overdue stops ordinary dispatch; pass explicit recovery after
restoring the target. A timeout does not reopen a tombstoned source.

CleanupPort must provide idempotent exact-inventory support removal, concurrent write
fencing and retained-read guarantees; it must preserve shared and host-owned graph
facts. Returning complete without these guarantees is a nonconforming adapter.
Unknown/unmanaged bootstrap records are not inferred from missing delta sources.
Revision-bound artifact references cannot be reused by a different operation plan:
choose a new revision/transformation/access identity rather than overwriting old
published records. The target adapters implement cleanup inspection against durable registered jobs and exact inventory. Actual integrations verify shared supports, stale cleanup fencing and restart reconciliation; fenced bootstrap observers verify backend inventory before ledger import.

## Managed lexical reads

Register lexical/managed in Executor and Cleaner. Supply the full typed record
inventory, a real payload fingerprint validator, finalized metadata schema and pure
metadata clone function. Host clone callbacks must not mutate their input metadata.
Returned managed document IDs are canonical exact-reference hashes; keep original
artifact/source locators in your typed metadata or projector.

Capture one publication before planner/fan-out. Build the trusted scope binding from
that capture and pass it unchanged. RequirePinnedPublication now declares targets
that reject live reads during admission. A pinned empty inventory is complete-empty;
it does not grant live access. Strict capture refuses missing requested targets in
partial publications rather than substituting old revisions.

Readonly BM25Snapshot keeps an owned corpus and the original binding fingerprint,
with no mutation API. Supply CloneMeta that owns nested domain data; each query result is cloned from the retained corpus before delivery, so consumer mutation cannot change future snapshot queries. Generic ResultSet copies ragy-owned fields and does not deep-clone arbitrary BYOT metadata. The managed adapter uses the same binding through scoring;
there is no unrestricted scoring fallback. Lexical records are in-process memory
in this reference profile: reconstructing a fresh adapter cannot claim retained
snapshot availability solely because durable manifests exist.

## Bootstrap inventory

Use Bootstrapper with a host InventoryVerifier for existing records. Supply an owned
namespace inventory, watermark, target coverage and confirmed published manifests.
Complete requires full coverage; delta may be partial and never proposes deletion
of absent sources. Keep keys with unknown revision/ownership as UnmanagedRecord;
do not invent source manifests, fingerprints or coordinates for them.

The verifier must confirm actual backend ownership and exact artifact/support/revision
inventory under its fenced watermark. Successful import changes only durable
manifest/publication state. Complete inventory produces Missing proposals with the
original expected publication; explicitly create/acknowledge a tombstone before
cleanup. Replay of an old receipt keeps its original expectations, preventing deletion
of a newer publication. Changed inventory under the same kind/watermark conflicts.
Construct `NewFencedInventoryVerifier` with target observers that inspect actual artifacts and support inventory. Backfill of unverifiable historical records remains an explicit host reindex operation; unknown ownership cannot be adopted by guessing.


## Persistent bounded tensor reads

Construct `tensor/persistent.Config[TMeta]` with namespace, target, lifecycle Store,
BYOT Schema/MetadataCodec, explicit `tensor.Space`, CloneMeta and positive bounded
catalog/payload/record limits. CloneMeta must return owned metadata, including all
nested slices/maps/pointers; the adapter gates host callbacks using the captured
read binding. Keep target roots under host-controlled local filesystem permissions.
The declared filesystem durability profile requires flock, atomic rename and fsync.

Build exact `source.Reference` values for staged artifacts and register the target
with the lifecycle Executor. Do not call Stage with an unregistered plan; Stage is
intentionally fenced by durable target-unknown state and expected publication.
CapturePublication before fan-out and put the captured Binding into the request.
Current/unpinned reads are unsupported by this target, including unrestricted reads.

Use `query.Intent` with a compatible normalized Embedding, candidate
references from an admitted dense/sparse path and an explicit CandidateBudget.
Set positive TopK no larger than the budget. The adapter owns query matrices and
reference slices, intersects planned/query filters with mandatory scope, and loads
only admitted candidate payloads. Reference fixture limits are 100/10; configuration
may set other explicit bounded limits. Query does not generate candidates or scan
all payloads. Use Query for owned candidate evidence or Retrieve for ordinary
ResultSet integration. Compare numeric scores only under identical declared score
semantics; different embedding profiles require explicit rank fusion/normalization.

Register the same adapter as a Cleaner target. A captured old revision is readable
while its exact files are retained; physical cleanup makes it unavailable. Never
substitute a newer revision on unavailable. Cleanup operates only on durable managed
inventory and must not be used to infer ownership of unrelated or orphan files.


### Portable candidate composition

Import `tensor/query` for `Intent`, `Result`, `Target` and `Search`; query contracts
belong to this portable package, not to `tensor/persistent`. There are no aliases for
the replaced adapter-specific query types. Set Search.Config.Candidates to an
admitted retrieval Backend with ReadCapabilityProvider, Target to a bounded tensor
query port, and explicit BYOT CloneCandidateMeta/Reference callbacks. A
ProjectedBackend can adapt the request intent for an existing lexical/dense leaf.

Leave Intent.Candidates empty on Search input: Search supplies the actual candidate
references from one bounded retrieval call. It negotiates both leaves before
candidate dispatch, sets candidate TopK/FetchLimit to CandidateBudget, rejects
backend overflow instead of truncating, deduplicates exact reference values, and
retains the captured read binding. Ordinary dense Vector tuning is forwarded to
the candidate leaf and cleared before tensor query; a token matrix remains a
separate compatible Embedding. Candidate numeric scores are not added to MaxSim.

Each callback and delivery is freshness-gated. Candidate requests own their token
matrices and core options separately from the final tensor request. Required clone
callbacks must own all nested metadata. Use an actual captured candidate index
snapshot; declaring a capability is not a substitute for backend enforcement.

The persistent target reserves `.stage-<operation digest>` inside its owned target
collection for the exact durable registered operation. Interrupted staging is
recreated only after the same immutable plan/publication fence is checked. Cleanup
and InspectCleanup account for this reserved path; opaque unknown names are not
inferred to be managed records and remain untouched.


## Persistent managed dense reads

Use `dense/persistent.Record[TMeta]` with exact source Reference, `dense.Record`
Value and explicit `dense.Space`. Configure an owned local filesystem root,
namespace/target, lifecycle Store, schema/codec, CloneMeta, catalog/payload byte
limits, MaxRecords per source and MaxScanRecords across admitted query records.
This profile requires finite, dimension-compatible unit vectors and never silently
normalizes. Generic raw dense indexes do not inherit its publication guarantee.

Register the adapter as a lifecycle target and use Stage/Inspect through Executor,
then publish the required target inventory. Retrieve takes
`retrieval.Query[persistent.Intent]` with an explicit normalized dense Embedding;
leave Options.Vector empty to avoid two conflicting embedding inputs. Require a
captured pinned binding even for an explicitly unrestricted access profile.
The reference profile uses positive TopK and does not accept graph tuning or score
thresholds; unsupported requests fail before target I/O. Exact scan overflow is an
error, not a silent sample/truncated ranking.

Keep score semantics with the model/configuration-specific normalized-dot identity.
Zero and negative scores are present native evidence. Use explicit fusion policies
when mixing them with lexical/tensor scores. CloneMeta returns fully owned metadata;
mandatory/query/planned predicates run before payload loading or clone callbacks.
Register the same adapter with Cleaner for exact retired-inventory deletion.

For a joint source operation, construct one manifest with all required target
inventories and one immutable typed payload/fingerprint, and register typed ports
that project the relevant records into each real target. CapturePublication once
with all requested target names before fan-out. A timeout after target commit means
unknown outcome: call Reconcile/Inspect, do not blindly replay the write. Default
publication remains on the previous revision until all required targets are ready.
Physical cleanup follows the logical swap/tombstone, preserves unrelated sources,
and makes cleaned retained snapshots explicitly unavailable.


## Managed graph source inventories

Use `graph/managed.Payload[TMeta]` with explicit Node/Edge wrappers and source
References whose artifact is the logical node/edge ID and representation is
`graph-node`/`graph-edge`. Supply original support references in the matching
lifecycle target inventory. In this profile, supports must share the manifest's
namespace/source/revision/access identity. Register the actual adapter with Executor
and Cleaner; confirmed publication and exact retained inventory are required for
reads. Do not invent sources to represent host-owned facts.

Configure graph node/edge attribute schemas, optional typed codecs, a required deep
CloneMeta policy and positive source-record limit. CloneMeta must own nested values
and must not mutate its input. Canonical entity metadata belongs on the fact;
source-specific provenance belongs in its support references. Two sources with
different canonical payloads under one ID produce conflict records and do not merge
silently. Identity resolution/aliases must be explicit before storing resolved IDs.

Call Traverse with managed.Request: captured Read binding, explicit traversal,
MaxNodes and MaxEdges. This profile applies NodeFilter to expansion and output;
forbidden nodes never serve as bridges. FindByIDs uses Traversal.Seeds as lookup
IDs under the same admission and returns only allowed requested nodes. Both APIs
return typed snapshots and original support references, with no fabricated scores.
The result owns node labels, metadata via CloneMeta, and reference slices.

The adapter is in-process storage: retain its instance while using its captured
snapshots, or use another adapter with a declared durable retained-read guarantee.
After process/instance inventory loss, reads return unavailable. Cleanup removes
only the exact retired inventory, preserving equal facts supported by other sources.
Use the typed ontology/extractor and `recipe/graphexpand` / `recipe/graphsummary` contracts below. Host-owned graph foundations require an explicit immutable basis ID; ResultSet projection must preserve original supports and rank-only graph semantics.

### Managed graph basis and filesystem lifecycle storage

Pass an explicit byte budget to `filestore.New(root, maxSnapshotBytes)` everywhere,
including restarted instances. Configure it for the retained namespace history;
no implicit unbounded reader or compatibility overload exists.

Register host-owned graph foundations with immutable `SetHostBasis` IDs and select
the desired ID explicitly on graph reads or BackendConfig. Rotate to a new ID when
facts change; release old IDs only after retained reads no longer need them.
Keep HostBases separate from managed source references in consumer projections.
Do not manufacture source locators for facts without an original managed source.
Use the default graph backend only when conflicts should fail; otherwise supply
an explicit projector with the application's conflict policy. Graph reachability
has absent score by default, so consumers must use rank or declared fusion rather
than treating it as a dense similarity score.

### Bounded text recipe consumers

Opt in explicitly to `recipe.New(Config).Run`. Supply typed planner/assessor,
complete leaf admission, pure pricing quote, BYOT clone policies, source support
resolver and dedup identity. Use model-free retrieval/admission/pricing ports in this declared text profile;
model calls run through the reserved planner/assessor ports. A model call hidden
inside backend or pricing is outside the declared contract. This recipe profile uses text retrieval; a host query encoder must have its own declared budget-aware contract before use. Keep baseline as the default until the specified experiment supports a
change. Treat partial/insufficient as retrieval outcomes, not a generated answer.

Stop passing precomputed vectors or graph seeds to text rewrite recipes. Their
meaning belongs to the original query; use a separately budgeted host query encoder
for derived text rather than silently reusing it. Planner can return only text
variants/subquestions; filters, intent and pinned publication remain from the
original request. Preserve Result.Queries and Selected.Contributors when exporting
source/query provenance. Do not serialize raw envelopes into a recorder without
the separately required evidence redaction policy.

### Evidence and recorder consumers

Build `evidence.Input` from observed stages; mark unavailable, unsupported or not-run
instead of inventing earlier hits. Preserve original score semantics and source
revision. For scoped export provide schema/metadata codec and authoritative
SourceAdmission for every support; a public winner does not authorize contributors.
Configure ID/numeric allowlists and separate query/snippet opt-ins. Do not serialize
auth, credentials, metadata or errors as diagnostic strings.

Use explicit recording mode and BYOT CloneResult with `evidence.Run`. Check required
mode errors even when completed retrieval data is retained: overall success was not
achieved. Do not retry retrieval on sink failure. Use Receipt.State to distinguish
disabled/failed recording. Absent labels are ungradable; require source/query/rubric
identity before consuming redacted labels as external ground truth.

### Recipe evidence recording

Consumers opting into recipe recording call `recipe/recording.Run` with the
recipe, mode, sink, retrieval identity, metadata schema/codec/clone and original
source admission port. Set identifier/numeric/snippet/query allowlists explicitly.
Handle `evidence.ErrRecordingFailed` separately from retrieval success: required
sink failure preserves completed evidence but fails overall recording. Do not
restart model or retrieval calls to repair a sink write. Protection errors return
no result or receipt. Disabled mode requires no recording ports. Retain actual typed contributions/locators and apply the location/contribution allowlists before export. The recorder preserves query stages, fused contributors and failed-attempt observations; missing observation remains explicit. Geometry export does not imply retained source resolution or source truth; source admission remains host supplied.

### Document source mapping

Populate `Document.SourceMapping` with the retained original/derived mapping and
`Document.SourceSupports` with original contributors. Use `SourceLocations()` to
obtain their owned union. Never copy winner metadata to reconstruct other source
references. Rewriting Content requires a corresponding new mapping or zero mapping
for unobserved precision; ValidateDocument rejects stale coordinates. Custom group
mergers retain input supports automatically. Renderer defaults to document mapping;
string-only snippet transformations retain supports without exact byte claims.

For persistent dense/tensor ingestion, populate the typed record's SourceMapping;
reindex/backfill old content from retained originals for exact coordinates. Storage
owns the mapping in its checksum-covered payload; it rejects foreign source
revision/access claims. Absence of a supplied mapping remains explicitly
unobserved. Recipe Supports ports must authorize and return all locations attached
to the document, rather than returning only the winning document's reference.
Source authorization/catalog retention remains a host responsibility.

### Managed graph projection clear break

`managed.BackendConfig.Project` now returns typed `managed.Projection` values,
each with a Document and contributing Facts (`NodeFact`/`EdgeFact`, original fact
ID). Declare facts independently of the display/document ID; fabricated identities
or source references fail closed. Backend attaches their original source supports
from an inventory captured before the callback. A content mapping may reference
only these admitted source references; retained resolver validation is still needed
to establish the authenticity of coordinates and text. Projected metadata must
satisfy the backend node metadata schema and mandatory scope. Clear/recompute a
mapping whenever content changes. No legacy document-only projector overload is
kept. Default node projection requires no callback and preserves all original
supports. Host-owned bases remain explicit in the typed traversal result; absence
of managed source refs must remain source-unavailable in document/evidence export.

### BYOT graph identity decisions

Pass typed kinds, relations, attributes, local mention IDs and original locators
into `graphingest/resolution`. Supply ontology validators, explicit namespace/alias
identity decisions, relation keys, attribute clone/equality and source admission.
Do not use display names as global identities. Handle ambiguous mentions and
multi-variant conflict groups explicitly; do not silently choose first/last values.
Store ontology/policy identities with transformation configuration and source
support decisions when materializing lifecycle-managed graph artifacts. Changing
policy identity requires reevaluating decisions even when canonical IDs happen to
remain unchanged. No automatic source extraction or persistent decision-history
materialization is implied by the resolver result.

### Resolver materialization and lifecycle handoff

Use `graphingest/materialization` to prepare an owned managed graph payload and
planned manifest for each source/revision. Supply schema-checked BYOT node/edge
projection, attribute/metadata cloning, support admission, finite bounds and actual
source/config/payload fingerprints. Its transform binds ontology/alias policy
identities; changing either requires reevaluation and a new source plan with the
correct expected publication. Ingestion source admission must authorize the new
retained revision, which may not yet be published. Submit the plan through lifecycle
Prepare/Stage/Publish; do not directly overwrite the active graph.

Handle same-source conflict, unresolved assertions and missing endpoint support
explicitly. Cross-source conflicts remain distinct managed variants; source deletion
uses a tombstone plus retained artifact/support cleanup. Retain full resolver decision
records in durable host storage where history is required; the new transformation
fingerprint and retained manifests do not alone store the entire typed decision log.

### Bounded model graph extraction

Use `graphingest/extraction.Adapter` with admitted source mappings, trusted ontology
namespaces and typed access metadata. Supply metadata attributes/schema, deep cloners,
source/quote admission, kind/attribute validators, shared budget ledger, pricing,
exact input-token counting and a bounded injected client. Construct snippets through
scoped retained-source loading before calling the adapter. The client receives only
ordinal/text pairs and token limits; it must enforce those limits and avoid retries.
Return typed local mentions with snippet ordinals and known actual usage even on
failure when available. Canonical identities, source revisions and permissions are
assigned outside the model. Handle ambiguous mixed namespace, protocol/ontology
errors, budget errors, deadline and revocation explicitly; do not retry a failed
model call implicitly. Actual transport and quality experiment remain required to
qualify a chosen client for production use.

For a supplied HTTP transport, construct the optional `structured.Extractor` and
assign its `Model` and `CountInputTokens` methods to core extraction configuration.
Choose the model, instructions and strict response schema explicitly. Supply a
validator implementing that same schema, exact model tokenizer, positive duration,
byte limits and actual host-price conversion matching the core quote. Core request
token limits are carried into the provider output cap and pre-dispatch input check.
Handle refused/incomplete/protocol/token-overrun errors without implicit retry; known
usage must still settle failed calls. Do not expose binding, access metadata or source
identity to the model. Use a live experiment to qualify the chosen model/tokenizer;
HTTP fixture success alone does not establish quality or production budget accuracy.

### Retained identity decision history

Use `Result.EntityDecisions` and `Result.RelationDecisions` to associate local mentions
with explicit policy outcomes, canonical identities and original source supports.
Capture the full typed input and actual resolver result through `resolution/history`,
with explicit run/extraction configuration identity and predecessor snapshot ID.
The JSON profile requires faithfully serializable BYOT values and bounded byte/support
limits. Append the immutable snapshot to durable storage and retain its reference in
the host source catalog before declaring decision history persisted. Keep previous
references during recomputation; changing policy/ontology identities must not rewrite
old records. Coordinate graph publication through its lifecycle rather than infer it
from a history write. Read history with fresh retained-source admission; an old write
does not authorize later access. On uncertain append completion, inspect the original
reference explicitly and avoid silently creating another resolution run.

### Bounded local expansion

Use `recipe/graphexpand` with the managed scoped graph profile, explicit canonical
seeds/traversal filters, finite depth/node/edge limits and a shared budget ledger.
The reference configuration is depth 2, 50 nodes, 100 edges, duration 5 seconds and
at most 4 graph calls in the ledger. Each recipe run dispatches one BFS traversal,
without model/token usage. Quote a fixed per-attempt graph cost and provide a deep
BYOT metadata cloner. Retain the original binding; handle insufficient budget/price,
target errors and cancellation/revocation explicitly. Use `SourceReferences` for
original evidence supports, and preserve explicit conflicts/host foundations.
Successful expansion is not an automatic final-answer sufficiency decision.

### Community/global summaries

Construct `recipe/graphsummary` with explicit community members, source snippets,
typed access metadata, mandatory schema/cloners and host membership/source admission.
Load originals through scoped `source.Reader` and explicitly pin its original
transformation target alongside graph inventory. Use a shared ledger: reference caps
4096 input / 1024 output tokens, three model calls, duration/deadline 5 seconds and
20 snippets per community. Community is one map; global is two maps plus one reduce.
Configure the optional HTTP `Summarizer` output schema/validator/tokenizer/pricing and
assign its model/counting ports, or provide another conforming bounded client.
Models select input ordinals; core owns original citation/revision association.
Resolve immutable summaries through fresh binding/source admission before using their
derived text, including cached artifacts. Handle missing declared member coverage,
bounded partial budget stops, source loss, cancellation/revocation and invalid output
explicitly. No final answer, implicit retry or fabricated global result is produced.

### Planner/assessor model limits

Change Planner callbacks to `(ctx, request, recipe.ModelLimits)` and Assessor
callbacks to `(ctx, assessmentInput, recipe.ModelLimits)`. Enforce both passed token
maxima on that dispatch; separate static limits must not replace reservations.
Quotes for Plan/Assess require positive input/output token maxima, including unknown
price profiles. Retrieve quotes retain zero model tokens. For advisory unknown cost,
set CostKnown=false and Cost=0; observed token usage must still be reported accurately.
The core rejects observed token overruns independently of quote cost knowledge.

Optional structured HTTP Planner.Plan/Assessor.Assess implement these signatures.
Provide matching executable output schemas, a real model tokenizer, explicit model
instructions and actual price conversion. Set assessor document bounds across all
executed queries. Only admitted query text/snippets reach the model; authoritative
source refs and domain metadata remain in ragy. Handle malformed/foreign selections,
revocation, failed calls with usage and unknown accounting without automatic retry.

### Failure journals and fusion observations

Use Recipe.Run for retrieval with error-payload suppression. Use RunObserved only
when explicitly collecting failed-attempt observations: an error plus Failure /
StageFailure is not a successful retrieval outcome, and Selected remains empty.
Honor protection/cancellation suppression; never rebind a journal to a refreshed
scope or publication. Enabled recipe/recording uses this port automatically and
still requires source admission and metadata ownership before export. Handle the
original execution error even when its failed record was written successfully.

Result.Fusion now explicitly declares whether fusion ran, completed or lacks a
retained observation. When constructing fixtures/consumer results, set it honestly;
an unknown value is rejected by recording. Do not infer observed fusion from empty
Selected or reinterpret failed retrieval as observed-empty. Failed-call available
usage is settled once, and unknown usage keeps the reservation; do not retry based
on missing observation or sink failure.

### Evidence location wire contract

Update strict evidence readers/schema fixtures for mandatory locations_state and
locations. Supply original evidence.Hit.Locations or typed document SourceMapping /
SourceSupports with matching admitted hit Sources. Never substitute a graph/index ID
or latest source for a retained original locator. Unknown mapping remains unavailable.

Enable Policy.AllowLocation only for geometry and printed-page/table/element labels
that may be exported. Keep source identifier and numerical policies explicit; denied
or partly redacted source identity omits the entire location. Access fingerprints are
excluded from wire regardless of permission. Decoder validation proves shape, not
source authorization/retention: resolve with the original host binding/catalog.
Re-export old records from retained authoritative observations when available; do not
invent missing coordinates. Handle strict incompatible shape explicitly.

### Contributor wire association

Update strict evidence schema/readers for contributions_state and contributions.
Enable Policy.AllowContribution only for query-ordinal/document/list-position links
that may be exported; keep document identifier, numeric and location policies explicit.
No callback receives raw query text or domain metadata. Association ranks are positions
in the captured query document list; native stage rank is exported separately when
observed. Unknown upstream rank is not reconstructed from this position.

Supply generic producer Hit.Contributions with truthful executed ordinals and admitted
original locations. Recipe recording supplies and validates the tuples automatically.
Consume each contributor's own locations after dedup, rather than assigning the whole
support union to every query. Source authorization/retention still requires the host
binding/catalog. Migrate old records by re-exporting retained observations; absent
associations stay unavailable and are never fabricated from current rankings.

### Exact inventory during persistent recovery

Persistent dense/tensor Inspect now requires a valid non-tombstone manifest containing
the named target and exactly the artifact reference set stored in its durable catalog.
Missing, duplicate, substituted or extra references cannot acknowledge readiness.
Ordering of the same set is irrelevant. Preserve the full original manifest for
reconciliation; do not construct recovery requests from only recently produced IDs.

Treat a catalog missing a descriptor as an incomplete target even if all remaining
payload checksums match. Stop publication, retain the original inventory and recover
from authoritative source data through an explicit operation. Do not guess deleted
references or silently publish a reduced corpus. No storage-format conversion is
required by this check; malformed or incomplete existing targets must be repaired.
The check proves exact artifact references, not source-support authenticity or fenced
namespace coverage; bootstrap still requires its separate host verification contract.

### Explicit partial publication capture

Use CapturePublication for strict reads. Select CapturePartialPublication explicitly
when a target branch may be excluded. The partial pin retains ExcludedTargets and only
ready target/source revision tuples. A target missing for any active source excludes
that whole branch; ready faq records must not turn an incomplete policy branch into
successful empty retrieval. No previous revision is substituted.

Custom retrieval adapters supporting this profile implement PublicationAdmission and
reject excluded/unrecognized target branches before I/O. Request projection forwards
that admission. Use PartialReadNode for an explicit pre-dispatch branch skip, and retain
result Coverage; runtime failures/revocation do not become permissible skips. Shipped
managed dense/lexical/tensor/graph profiles implement the contract. An owned readonly
BM25 snapshot accepts only its originally captured binding during retrieval.

Binding fingerprints now include exclusions. Discard previously computed external
cache identities and compute fresh keys from the current immutable binding. Evidence
capture combines producer coverage with the pin and preserves partial outcome even
when a producer supplied complete/complete-empty. Failed/insufficient outcomes retain
their own cause; invalid complete-empty input with hits is still rejected. Allow source
identifiers explicitly when exporting available per-target revisions; auth data remains
excluded. No storage-format conversion is introduced by partial capture.

### Verified ingestion reuse

Use Executor.CheckReuse before skipping ingestion of an unchanged source. Supply the
exact desired namespace/source/revision/content/transformation/access identity and
complete target name profile. Only ReuseDecision.CanSkip()==true confirms reuse at
the observation point. Changed ACL, changed transformation/revision/content, partial
publication and missing target data require ingestion or explicit recovery.

Prepare/Stage replay retains a previously confirmed workflow checkpoint; it does not
certify that volatile backend data still exists. CheckReuse inspects each target's
retained inventory once and fences its decision with a second durable generation read.
Handle ErrOutcomeUnknown, context errors and ErrConflict without inferring success or
starting a blind retry. The decision's Publication identifies the observed active
manifest; subsequent changes still require ordinary expected-publication CAS.

An ACL-only replacement must carry new access fingerprints through source references,
manifest and target payloads, even when content and source revision are unchanged.
Host authorization epoch/revocation remains a separate read freshness responsibility.
No storage-format conversion, scheduler, model call or background retry is introduced.

### Volatile inspection and registered source supports

Preserve the full original manifest for lexical/graph recovery. Inspect now compares
full identity, payload fingerprint, artifact references and original-support sets;
a matching operation ID alone cannot acknowledge readiness. Retained staging checkpoints
own their nested slices. Use Manifest.Clone when retaining or adjusting a manifest
without sharing caller artifact/support backing arrays.

All supplied target adapters reject Stage requests whose supports differ from the
durable registered plan, even if artifact references and payload fingerprint match.
A changed support requires a new explicit operation plan and expected-publication CAS.
Handle invalid/protocol/conflict outcomes without publishing or inventing completion.
These are managed in-memory checkpoint changes; no persistent data conversion is added.
Source authenticity and access decisions still require the host/source admission ports.

### Persistent catalogs with original supports

Dense/tensor catalogs now require storage identity ragy.dense-index/inventory or
ragy.tensor-index/inventory and a complete artifacts array with original supports.
The earlier catalog shape is rejected with ErrProtocol, even if its payload checksums
are intact. Missing supports cannot be inferred from current chunks or operation IDs.

Reindex from trusted source revisions and their explicit original support mappings
into a new host-owned storage root, then stage and publish under expected-publication
CAS. Keep existing unknown data unmanaged until the host verifies ownership and
coverage. Neither a format error nor a successful reindex authorizes deleting old
records. Supply the complete retained manifest to recovery/inspection; changed original
supports require a new explicit plan. Handle rejection without blind retry or readiness
acknowledgement. This storage break requires updated consuming deployments as well as
updated Go callers.

### Actual persistent backend bootstrap verification

Register dense/tensor adapters as InventoryObservers in NewFencedInventoryVerifier,
then supply it to NewBootstrapper for the exact target profile. Provide published
manifests with full original supports, a unique host watermark and explicit coverage.
Enumerate opaque immediate storage keys separately as UnmanagedRecord; retain unknown,
old-format, staging and retired keys without inferring source identities. Complete
inventory must account for all entries; delta does not propose removal of omitted
sources. A watermark identifies the supplied observation and is not a storage lease.

Use bounded inventory limits sufficient for the enumerated target entries and verified
payloads: dense MaxScanRecords, tensor MaxRecords. Handle fence conflicts, protocol
errors, missing data, unavailable limits and cancellation without confirming import or
blindly retrying mutation. Observation performs no model calls, writes or deletion.
Bootstrap receipt Missing entries still need an explicit expected-publication tombstone.
The supplied observers currently cover persistent dense/tensor; other target profiles
require an actual observer implementing the same held-fence contract.

### Lexical and graph bootstrap observers

Create an observer with Adapter.InventoryObserver(maxEntries, maxRecords), then register
it by its exact target name in NewFencedInventoryVerifier. Supply explicit positive
limits for retained entry enumeration and verified records. Include opaque retained
versions as manifest:<operation-id> UnmanagedRecord keys when they are not being imported.
Graph host foundations are host:<basis-id> keys and must remain Unmanaged. Complete
inventory must account for both classes; delta may omit other entries.

Recover or reindex a lost in-memory target before importing its published manifest.
A ready durable checkpoint alone is insufficient; a new adapter returns Unavailable
for missing retained data. Handle conflicting writers, bounds, incompatible supports
and cancellation without import acknowledgement. Keep normal read/policy freshness and
expected-publication CAS after this observation; no observer promises a storage lease.

### Registered lexical cleanup and unknown deletion recovery

Use durable Cleaner.Begin/Attempt/Reconcile for retained lexical revisions. Direct
Cleanup requests now require a registered owner/retired/target item in unknown dispatch
state and exact retained artifact/original-support inventory. A pending job alone does
not authorize deletion. InspectCleanup also rejects unregistered or changed inventory;
known completed work remains idempotent through Cleaner.

After a lost deletion response, create a fresh Cleaner against the same durable store
and call Reconcile once. Do not retry physical deletion while the item is unknown.
Keep host-driven deadlines and capped backoff; overdue work requires explicit recovery
and still respects the confirmed NextAt time. Read barriers are independent of physical
cleanup. Expired access tokens require renewed host authorization even if their retained
publication data still exists; cleanup never extends read authority.

### Lost manifest checkpoint acknowledgements

Treat ErrOutcomeUnknown as a request to reload/reconcile durable state, including
Prepare and ready/publication checkpoint writes. A candidate manifest returned with
an error is not a success acknowledgement. Recreate Executor with the same durable
store and admitted ports; replay Prepare/Publish to load confirmed ownership/publication,
and use Reconcile for unknown targets before an explicit Stage continuation.

Unknown dispatch may precede any backend effect. If Inspect confirms pending, the host
may explicitly Stage once. If Inspect confirms ready, do not stage again. A durable
ready checkpoint replays without Inspect or Stage. Preserve old publication bindings
until the common logical publication is confirmed; do not expose staged records based
on candidate progress or infer rollback from a lost response.

### Lost cleanup job/checkpoint acknowledgements

After a Begin acknowledgement error, reload by calling Begin on a fresh Cleaner with
the same store and owner. A committed job is reused; an uncommitted one is created only
by that explicit continuation. After an Attempt checkpoint error, reload the registered
item and call Reconcile for unknown state before any further Attempt.

Unknown dispatch may have no physical backend effect. Inspect waiting allows a later
explicit Attempt after NextAt. If the backend already deleted the exact inventory,
Inspect done confirms completion without deleting again. A durable done checkpoint
replays without Inspect/deletion, including a lost whole-job completion response. Do
not infer success from the candidate CleanupJob accompanying an error, and do not reopen
tombstoned reads while waiting for physical cleanup or manifest acknowledgement.

## Optional persistent query payload reader

Dense/tensor Config accepts an optional lifecycle.PayloadReader. Leave it nil for
bounded local-file materialization. A typed nil implementation is invalid. The port
receives the exact admitted index reference, a managed local path and positive byte
cap. Honor the supplied context, return owned bytes within the cap and perform no
implicit retries. The adapter retains mandatory filtering before this call, validates
returned bytes against the retained digest and payload contract, and rechecks freshness
before delivery. Cancellation/deadline and protocol failures remain distinct; arbitrary
host storage errors become unavailable without exposing paths or credentials.

This port grants no write/delete, original-source ownership or publication authority.
Stage, Inspect and inventory verification continue to inspect actual physical files
independently. Consumer conformance probes may wrap the query reader to count actual
payload materialization, record admitted references and trigger revocation during I/O.

## Planner predicates in adapter admission

Pass the complete planned request through PrepareRead before adapter I/O. It now checks
Plan.Filters as well as query filters, then intersects both with immutable mandatory
scope. Remove separate preliminary plan intersections that return raw schema errors or
skip preflight. Empty plan filters retain the query/mandatory restrictions; contradictory
predicates return no matches. Handle protected unsupported capability during negotiation;
execution-time failures remain non-skippable and must not trigger an unrestricted rescue.

## Stored integer catalog matching

For custom persistent adapters, keep exact JSON numbers until schema normalization,
then retain its owned typed output for matching. Validation alone is insufficient.
The supplied dense/tensor targets now apply this at every catalog read. Existing exact
numeric JSON does not require arithmetic repair; previously rounded metadata requires
trusted-source reindex, as described for BUG-001. The declared mandatory scope profile
is Eq/In/And: an exclusion query operator is not a supported mandatory policy.

## Custom metadata codecs at wire writes

Supplied Qdrant/pgvector Upsert now validate and canonicalize custom codec output before
transport serialization/dispatch. Handle ErrInvalidArgument with zero write effects;
remove assumptions that Encode success alone authorizes storage. Custom clients must
preserve integer JSON numbers before the library schema boundary. If an older path stored
rounded or invalid metadata, reindex from trusted source. Injected port tests do not replace
a consumer's verification of its actual driver/service implementation.


## Standalone locator persistence and host extensions

Persist standalone locators through `source.EncodeLocator` and load them with
`source.DecodeLocator`. Store the schema envelope, not an unvalidated raw locator;
handle unsupported schema and protocol errors before lookup. Reparse/reindex from
trusted sources when older data cannot supply exact revision/representation or
geometry. Do not fill missing coordinates or resolve a missing revision as latest.

`ExtendedLocator[T]` JSON keys are `location` and `extension`. Update host storage
code that relied on default capitalized Go field names; validate/copy extension data
using the host's own contract. Keep canonical locator identity separate from UI or
domain annotations. Codec success alone does not authorize source payload access.
