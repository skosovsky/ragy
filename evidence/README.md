# Explicit evidence export

`Capture(ctx, read, Input[TMeta], Policy)` snapshots observations into an immutable
Record with schema identity `ragy.retrieval-evidence`. Record owns canonical bytes;
MarshalJSON and Snapshot return detached data. Decode rejects unknown/missing fields,
duplicate keys, incompatible schema, malformed scores, ranks, labels and coverage.
Wire shape validation does not certify a producer's authorization or export policy.
Transport byte limits, sink durability and retention are host responsibilities.

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

Schema and fixtures live in docs/task12. The independent verifier checks wire
structure; Go tests additionally exercise ownership, freshness, source authorization
and runtime invariants. Automatic execution/recipe stage observation adapters and
complete cross-capability evidence acceptance remain required integration work.

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
