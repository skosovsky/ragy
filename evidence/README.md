# Explicit evidence export

`Capture(ctx, read, Input[TMeta], Policy)` snapshots observations into an immutable
Record with schema identity `ragy.retrieval-evidence/v2`. Record owns canonical bytes;
MarshalJSON and Snapshot return detached data. Decode rejects unknown/missing fields,
duplicate keys, incompatible schema, malformed scores, ranks, labels and coverage.
Wire shape validation does not certify a producer's authorization or export policy.
Decode/canonical capture have a 4 MiB wire bound, 32-level JSON depth bound, 1024-item collection bounds and 65536 Unicode code-point text bound. Sink durability and retention remain host responsibilities.

Stages are supplied from actual execution observations. Not-run, unsupported,
missing-observation and unavailable are explicit statuses. This package does not
invent earlier stage hits or claim an absent observer saw a stage. Native scores
retain their declared finite scale, including negative values and values above one;
rank-only and unknown scores have no fabricated numeric zero. Input outcome and
coverage are explicit; complete-empty is rejected when an observed input has hits.

## Authorization, redaction and required fields

Capture freshness-gates every policy callback and final delivery. Scoped capture
requires schema/metadata codec plus SourceAdmission. It verifies mandatory metadata
before any document-ID policy callback, pins namespace/source/revision/access, and
admits each source support separately. SourceAdmission must check an authoritative
thin source catalog under the same binding; a public winner's metadata does not
authorize all contributors. A missing scoped export capability fails before export.
The codec and policy callbacks must be pure, concurrency-safe and must not perform
model calls. SourceAdmission I/O receives the supplied context and deadline.

All identifiers need an explicit AllowIdentifier decision, including score semantics,
source revisions and recipe/publication identifiers. IDs denied by policy do not
produce hit rows. Raw query is omitted unless AllowQuery is enabled; snippets require
per-document AllowSnippet. Numbers also require AllowNumbers. Diagnostic names come
from a fixed numerical vocabulary. Arbitrary metadata, raw auth scope, credentials,
error messages and free-text diagnostics have no wire field and are never serialized.
Mutation of source documents, metadata, support slices or a sink's returned buffers
cannot alter the record. Inputs must not be concurrently modified during capture.

Consumer-required fields distinguish unsupported capability, unavailable observation
and privacy denial. An optional absent label is ungradable, never grade zero. Present
labels bind the retrieval ID, exact supporting source reference and rubric before
export. A consumer must require the identifying fields before treating a redacted
label as usable ground truth; this package is not a judge or quality framework.

## Recording wrapper

`Run(ctx, read, RecordingConfig[TResult])` executes retrieval once, captures a BYOT
snapshot via CloneResult, and invokes a host Sink according to explicit mode:

| Mode | Behavior |
|---|---|
| disabled | Neither capture nor sink runs |
| best_effort | Retrieval outcome retained; receipt reports recording failure |
| required | Recording failure returns an error and preserves completed retrieval result and captured record; overall success must not be claimed |

There is no retry or repeat retrieval on sink failure. RecordingError has a stable
message and preserves errors.Is/As classification without adding sink error text
to the record. Protection failure during capture/sink always fails closed in every
mode and clears result/receipt. A completed authorized sink write cannot be undone
if revocation occurs afterward; final delivery is still suppressed.

The current schema lives in `tooling/testdata/contracts/evidence-v2`, with fixtures in `evidence/testdata` and independent validation in `tooling/schemas_test.go`. Earlier wire records remain historical and are rejected as incompatible. The verifier combines JSON Schema with explicit relational checks for sequential ordinals and query references, which standard JSON Schema cannot express. Go validates the same adversarial corpus and additionally exercises ownership, freshness and source authorization. Raw transport byte/depth limits remain separate from the declarative schema.

## Original location export

Hits accept explicit original `Locations` and automatically include typed document
source mappings/supports. Every locator is validated and must belong to the already
admitted hit source inventory before export callbacks. No geometry is inferred from
source IDs or unavailable mappings.

Location export requires explicit `Policy.AllowLocation`, allowed numbers, and all
six source identifiers permitted by the identifier policy. The location callback
permits its geometry and printed-page/table/element labels as a whole. It is a pure
host policy decision, followed by a fresh scope gate. Default location state is
omitted when known but not permitted; unknown location is unavailable. All denied
locations are omitted without exposing their count. Host authorization and retention
remain required independently of export permission.

Wire locations retain canonical document/text/page/region/cell/image geometry and
source representation/revision/transform identity. They exclude access fingerprints
and arbitrary metadata. Decoder validation checks tagged-union geometry and source
association; decoding does not authenticate the source or authorize resolution.
Location records own immutable bytes. The strict wire shape requires locations_state
and locations; incompatible stored evidence requires explicit consumer migration.

## Query contributor export

Hit.Contributions associates an executed query ordinal, original document ID,
one-based position in that query's captured result list and original locations.
Query text, intent, metadata and authorization are excluded. The observation producer
attests executed ordinals; generic decoding cannot authenticate a producer.

Association export requires explicit Policy.AllowContribution, numeric permission
and permitted document identifiers. The callback receives an owned location slice;
callback mutation cannot alter input observations. A fresh gate follows the decision.
Contributor geometry independently requires AllowLocation and its identifier policy.
Unavailable observations and omitted associations remain distinct, with no hidden
counts. No association permission enables raw query or snippets automatically.

Wire validation rejects invalid ordinals/ranks, duplicate query/document/rank tuples,
and contributor locations absent from the hit's admitted original location union.
Recipe recording additionally matches every selected contributor to the actual
captured query document/rank/support tuple before export. Query-index association
remains intact through dedup. The strict shape requires contributions_state and
contributions; update stored-record consumers explicitly.

## Structured decisions

`Policy.AllowDecisions` explicitly permits attempt-local query and selected-input ordinals, contributor ranks, fusion observation, per-query selection/delivery/uncertainty, actual sufficiency signal and fixed stop reason. These associations are omitted by default. They contain no domain document IDs, metadata, hashes or rationale. Retrieved means a captured retrieval completed, including an empty response; it does not claim evidence existed. Selected delivery means the contributor was directly returned when artifact rendering was disabled, or appeared in actual packed output when rendering was requested; uncertainty retains partial/derived source delivery. Recipe Result.ArtifactRequested distinguishes disabled rendering from a requested render that returned no artifact. The latter remains undelivered and uncertain.

Variant text independently requires AllowQuery. Recording Config.Revisions contains separately host-supplied model/prompt/config/recipe revisions, each independently gated by AllowIdentifier; missing values remain unavailable. A nil sufficiency signal means assessment output was not captured, distinct from a captured false signal. Captured decisions support audit of available input, not deterministic model replay.

## Original-source membership, rank and work bounds

Capture checks original-source namespace/source/revision/access membership. It
does not attest published index transformation/target membership: an original PDF
or UTF-8 quote can have a different transform from the retrieval index. Source
Reader's exact retained artifact admission remains required to authenticate quotes.
No source support is manufactured from a document ID or index transformation.

Document.Rank is a collection ordinal: zero unavailable, positive values at most
MaxExactRank (2^53-1), so float64 export preserves it exactly. Larger external
numeric IDs are rejected before hit identity callbacks; earlier attempt/stage
export callbacks may already have run. Hosts should bound rank by their collection.
Contributor ranks remain explicitly typed integer ordinals on their own wire path.

Finite stage/hit/source/location and accepted wire caps do not bound all input
allocation/CPU. Host pre-bounds nested document mappings/supports, contribution
locations, BYOT metadata and pure bounded callback work before Capture. Output
bounds are not an input-size or RSS guarantee. Full batch source/scope admission,
owned immutable record and fail-zero protection remain unchanged; sink side effects
after revocation and retention are host-owned.
