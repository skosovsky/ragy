# Task12 implementation design

Current implementation/profile inventory: [capabilities.md](capabilities.md).

Status: contract design in progress. The full specification remains authoritative; no new-capability conformance is claimed by this document.

## Existing foundations retained

Typed request/planner/binder and execution graph remain the retrieval composition. Existing route switch, aggregate, fallback/rescue, RRF, metadata codecs, source projection and contextual splitters are extended rather than reimplemented. `filter.Intersect` validates built conditions against each target schema; immutable conditions support stable structural fingerprints. Equality conflicts remain a restrictive intersection, not an unrestricted filter.

## Read access and publication

The new request contract will carry explicit read access, separate from mutable query filters. Unrestricted access is an explicit choice. Scoped access is created through an immutable binding constructor with host scope identity/policy snapshot, mandatory condition and required freshness authority. Planner, binder and request projection cannot replace that binding. Publication snapshot is pinned before backend fan-out and includes target/source revision associations.

A host freshness port validates authorization snapshots before cache delivery, before external payload consumers and before final result delivery. It reports expired/revoked/unavailable separately. A changed authority epoch invalidates an otherwise immutable old binding; it does not rewrite the binding to newer privileges.

Each leaf adapter declares schema/operator/access/publication capabilities. Built-in scoped composition refuses unsupported execution before I/O. Explicit partial mode reports the skipped capability without disclosing denied object identity. Public adapter conformance checks enforcement before payload loading and traversal; returned-document post-filtering does not certify a backend.

## Scores and processing

Replace the score contract with an explicit value/state/semantics representation: absent rank-only, finite native, explicitly normalized. Native scores admit negative and greater-than-one values. Comparisons and thresholds must name comparable semantics; rank fusion uses ordering and reports its derived score identity. Remove automatic raw-score clamping from adapters and keep any requested normalization as an explicit policy.

Post-processors receive context and the read binding so cancellation and freshness checks remain effective before downstream I/O. Grouping outputs retain constituent provenance and source mapping. Typed host metadata snapshots use caller-provided projection/ownership, not arbitrary reflection cloning.

## Lifecycle

A dedicated source lifecycle package owns source operations and durable manifest state transitions. Manifest persistence/CAS, target prepare/check/read/delete operations and host scheduling are ports. Runtime recovery is an explicit operation, without hidden retries or background workers.

Required targets are staged before a common logical publication; strict reads pin that publication. A separate explicit partial profile exposes per-target source revisions and coverage. Artifact IDs bind namespace/source/revision/transformation. Cleanup deletes recorded managed IDs and decrements graph source supports. Tombstone barrier and physical cleanup have separate outcomes.

Working adapters must cover dense/lexical/tensor/graph and the three required joint profiles. Persistent dense/tensor and manifest publication require actual storage integration, including reopen/restart, CAS conflicts, uncertain acknowledgements and deletion. Backend selection and concrete storage protocol will be recorded with evidence before declaring any persistence capability complete.

## Recipes, graph and source locators

Recipes reserve retrieval/model/token/cost usage before dispatch through atomic recipe-local accounting; host supplies pricing and external clients. They stop with typed evidence coverage and reasons. Single rewrite, multi-query and depth-one decomposition have independent executable cases and real-backend comparative experiments.

Graph schema/identity/conflict and membership policies remain BYOT. Extraction output is validated before publication. Source revision supports are retained across merging and deletion; derived summaries cannot retain revoked or removed support. Local expansion and community/global summary recipes enforce scope, visited-set and call/token/deadline bounds.

Locators identify the immutable source representation, revision and location. Text spans use UTF-8 byte offsets; page geometry, table cells and image regions use the reference normalization from the specification. A real optional parser adapter provides normalized output, with synthetic PDF/image assets for integration. Mapping loss and partial/OCR coverage stay explicit through chunking, merge, render and resolution.

## Verification and progress

The initial requirement matrix has 141 mandatory rows. The final completeness auditor must independently derive the denominator from the specification, detect missing rows, and count only requirements fully backed by implementation and matching verification. The correctness auditor independently examines the final code and reproduces findings. Both audits occur after full implementation and are repeated after fixes.

Versioned schemas, generic external-module conformance, persistent/parser integration and reproducible experiment artifacts remain required. Synthetic fixtures in `fixtures/` preserve the specified reference data; their presence is not equivalent to executing any acceptance check.

## Implemented tensor computation

`tensor.Space` requires explicit model/revision/configuration/vector-space/dimension
identity. `Embedding.Validate` and `Record.Validate` reject invalid float32 matrices;
unit-norm tolerance is 1e-5 on the squared norm. Computation uses float64 accumulation
without implicit clamping or normalization. MaxSim compares the complete identity.

`Rerank` accepts a bounded supplied candidate universe, validates all candidates
before scoring, rejects duplicates/overflow, uses stable tie order, preserves all
candidate IDs in output evidence and applies TopK only after scoring. Its contract
claims exact ordering within supplied candidates, never global index recall. The
caller owns input matrices and must not mutate them during computation. Output
contains value-only scores/IDs, without retaining mutable tensor slices.

The existing raw tensor embedder is a provider port; the host declares the space
when constructing records/queries. Persistence, scope/publication enforcement,
retrieval ResultSet conversion and experiment acceptance remain unfinished.

## Implemented retrieval scores

The unified score representation uses value/state/semantics fields on Document.
Unspecified state is absent. Numerical consumers check compatible state and scale
before ordering or arithmetic. Threshold is an explicit pointer carrying state,
scale and value, allowing negative/zero minima without overloading disabled state.
All existing adapters retain native scores instead of automatic clamping/logistic.
The obsolete ClampScore helper and MinSimilarity option were removed.

RRF is an explicitly chosen relative-max rank fusion policy with configured k in
its declared semantics. A caller comparator produces rank-only order and preserves
its inputs in score history, which prevents terminal TopK from undoing the order.
Default numeric mergers retain observations from losing contributors. ResultSet
and snippets copy value-only history slices; BYOT metadata ownership is separate.
Full revision-bound evidence export and persistent tensor result conversion remain
pending and are not inferred from these score tests.

## Implemented binding and target admission

The access package holds private immutable binding state, authorization identity,
mandatory reference-profile predicate, host authority/clock and copied publication
inventory. Live publication is a distinct explicit profile; pinned capability is
not falsely advertised by old raw-storage adapters. Request.Read is mandatory,
including for explicit unrestricted calls. Binder and projector cannot replace the
host binding. Each existing dense/sparse/lexical target validates the mandatory
intersection against its own schema before dispatch.

ProtectionError has sanitized text and preserved error classification. Rescue
cannot turn it into success. Entry and delivery wrappers cover success, empty,
partial and error returns; revocation suppresses both result sets and arbitrary
BYOT execution metadata. Graph remains rejected for scoped calls until pre-expansion
node/edge enforcement is implemented. No graph traversal isolation or persistent
publication guarantee is inferred from an output filter.

This increment does not complete scope: global composition preflight, explicit
partial coverage, public external conformance, hydration/resolution and pinned lifecycle adapters
remain required. Downstream and cache increments are described below.

## Implemented downstream processing boundary

Processor and query-aware reranker ports now carry context and immutable binding.
The chain validates before/after every consumer; built-in grouping checks before
host selectors/merge callbacks. Numerical history remains preserved through the
same processing API. Model HTTP dispatch is gated independently of enclosing
composition, and all model result/error return paths are gated again. Observability
forwards the binding rather than constructing an alternative policy.

Protection failures collapse joined sibling errors to the protection classification,
so a typed partial-result sibling cannot retain data through the failure boundary.
This remains contract enforcement with trusted host code, not a sandbox for
arbitrary custom Go callbacks. Export/recipes and full cross-target conformance
remain incomplete; the optional cache increment is described below.

## Implemented optional cache boundary

CachedBackend requires explicit index/revision/recipe/configuration/capability
identity, a BYOT host identity projector and a BYOT request snapshotter. Core query,
options, graph/plan profiles and immutable access/publication inventory are hashed
structurally. The store receives a digest instead of a raw query or policy. These
ports must capture all host-owned values that affect retrieval; core cannot infer
them from arbitrary generic types.

MemoryCache is bounded with an injected clock and LRU eviction. Metadata cloning
is explicit, and returned documents/history have independent ownership. Request
capture copies core slices and nested option/range pointers. Concurrent mutation
during capture is unsupported; the host must synchronize that boundary.

Cache hits still require underlying target capability/schema admission and current
host authorization. Revocation during cache loading clears the result and forbids
refresh. Live index identity is checked after loading and after target I/O; a
changed identity rejects the old hit or prevents storage under the old key.
Partial/error target outputs are never cached, and no retries/workers are started.
The live profile bounds staleness by TTL and host index identity rotation; it does
not replace the pending durable managed-publication protocol.

## Implemented strict composition preflight

Target negotiation now precedes aggregate parallel dispatch and route planning.
Every configured reachable leaf is included, including currently unselected
conditional/default/fallback/rescue branches. This is the strict profile; callback
selection is not an implicit partial capability policy. Pipeline admission is
repeated after binding, before target execution, so planner/binder changes do not
avoid schema checks. Custom execution compositions require RequestReadAdmission;
undeclared scoped/pinned nodes reject before planners run.

Preflight executes no planners/predicates/target payload I/O. It retains required
host freshness checks, and does not replace target enforcement or later delivery
gates. Route planner output is checked before decision recording; child results
are checked before route continuation, and protected failures stop rescue
predicates/targets. Reference tests exercise a real scoped lexical index inside
nested parallel composition. The explicit partial increment is described below; public external conformance
still remains required.

## Implemented explicit partial admission and coverage

The admission port now returns immutable ReadCoverage, including explicit
unobserved/complete/partial/unrestricted states. InspectRead validates the report
and rechecks freshness; undeclared scoped/pinned nodes reject. Public built-in
execution wrappers retain admission coverage independently of which branches are
actually selected. Branch reducers and route continuation preserve the report.

PartialReadNode is an explicit host choice for one configuration branch, without
weakening the immutable scope. A direct typed unsupported-capability admission
failure produces a static-label skip and never dispatches the child. A nested
partial child is still executed so its useful allowed payload is retained. Runtime
unsupported errors, authority failure, cancellation, invalid configuration and
joined admission errors cannot become a successful skip. Typed nil interfaces are
rejected before invoking their receiver methods.

Coverage contains no target payload IDs/counts/content or adapter error messages.
Its private state owns label slices and has a strict JSON envelope with schema
identity; failed decoding does not alter previously accepted state. The published
JSON Schema is checked independently in the host QA environment, without adding a
JSON Schema dependency to core. Full target-specific conformance and automatic
immutable evidence export remain pending.

## Implemented public external conformance boundary

The prior internal helpers moved to the public contracttest package, retaining a
single implementation and updating all shipped consumers. The new scoped suite is
generic over host request and metadata types and observes direct backend calls,
without letting a safe pipeline wrapper hide a defective adapter. Capability
negotiation must perform no payload I/O. Positive fixtures use the actual shipped
lexical engine and a separately instrumented host payload materialization port.

An independent Go module with an import namespace outside ragy executes the suite
with GOWORK disabled. Undeclared adapters reject before Retrieve; adapters that
load forbidden payload and post-filter later fail certification, even when final
IDs are correct. Premature I/O before a leaf gate also fails when denied output is
empty. Scenario factories own isolated mutable authority/clock/I/O state. The
suite is optional testing infrastructure, not an IAM, judge or experiment runtime.
Persistent dense/tensor, graph and parser-specific conformance still remains.

## Implemented source identity and scoped hydration

source.Reference is a value-only, comparable identity with exact revision and
representation, including transformation/access fingerprints. Hydrator validates
requested identities and pinned target inventory, then loads thin permission
metadata and validates the full schema/mandatory intersection before payload I/O.
It captures separate port request copies and rejects unsupported/incomplete source
identity, missing/denied descriptors and protocol substitutions. Output metadata
ownership is supplied explicitly by the host, not inferred through reflection.

Hydration is all-or-nothing; no source payload is delivered after any identity,
authorization, cancellation or freshness failure. Freshness is checked around every
I/O and metadata consumer. Deleted r1 cannot become r2 even when r2 remains retained.
The raw legacy storage contract was renamed RawStore and is not accepted as a
hydration port. Actual durable managed targets and citation locators are separate
remaining requirements; these host-memory retention fixtures establish only the
implemented hydration boundary.

## Retained text locators and rendering increment

`source.Locator` is a comparable, tagged value referencing an exact namespace,
source revision, transformation, access fingerprint, artifact and representation.
Built-in text/page/region/cell/image locations reject irrelevant active fields and
invalid geometry. `Locator.Identity` contains no score or rank; equal coordinates
including signed zero have equal canonical identity. Page coordinates are points
on the unrotated page; clockwise transforms preserve physical and printed identity.
Typed host extensions remain host-owned and do not override canonical identity.

`source.MappedText` is an owned private snapshot created from retained original
text or explicit derived content. All fragments carry support references; derived
context has unavailable precision. Slice uses rendered UTF-8 byte boundaries and
recalculates original byte spans. Join retains every source and marks separator
bytes derived. JSON uses the `ragy.source-mapping` schema identity, distinguishes
unobserved zero state, rejects invalid/unknown/trailing wire data and preserves the
old snapshot on failed decode. Structural decode does not authenticate source text;
resolution under the original binding supplies that guarantee.

`Hydrator.ResolveText` admits the entire batch before payload materialization,
collapses repeated locations and reference loads, validates retained UTF-8 spans,
and suppresses every citation on any error or freshness failure. It deliberately
rejects geometry/cell/image locations before I/O until their respective source
resolution paths are implemented. No latest substitution is accepted.

ArtifactRenderer now requires context, the original binding and explicit metadata
cloning. Every callback/consumer and final delivery has freshness gates. Mapping
and string-only Snippet rewriting are mutually exclusive. Rune budgets are
translated into byte spans; mapped trimming preserves exact source coordinates.
Dedup projects all contributors before output and retains all supporting locations.
Custom formatting receives separate metadata/support/history copies. Mapping
coordinates describe snippet content, not formatted headings or labels. Plain
label provenance remains unobserved precision. Source propagation through all
storage, chunking/grouping/rerank and real parser paths is not yet certified.

## PDF/layout parser contract before adapter implementation

The normalized envelope has schema identity `ragy.layout`, exact source Reference,
physical PageCount, document/page complete|partial|unsupported coverage, typed
non-content diagnostics, retained normalized page text, UTF-8 word spans, logical
table cells with row/column spans and unrotated point regions, and original image
regions. Source identity/revision/transformation/access fields must agree across
all representations. Physical page indices are unique/ordered and do not derive
from printed labels. Complete document coverage requires every page and complete
page coverage; missing pages or OCR-unprocessed regions cannot become complete.

The optional PDF adapter runs an external text/layout parser through a configured
Python executable. Core has no Python/parser dependency. Input is already authorized,
owned PDF bytes from ingestion/host source materialization, rather than a URL/path
lookup; the parser is not a scoped retrieval backend or authorization service.
Caller context bounds external execution. Input/output/page/word/table/image limits
are explicit; parser failure returns no partial payload via errors, and the adapter
never retries. A normalized output is validated before delivery. Unsupported native
geometry is rejected rather than inferred. Native rotated display rectangles are
converted back into top-left unrotated page coordinates, with rotation retained.

Words are joined by single spaces in a separately named normalized text
representation. UTF-8 spans are generated from actual encoded word lengths;
no offset is interpreted within PDF binary. Table cells are identified from the
parser's actual cell rectangles/grid boundaries; merged cells appear once with
explicit spans. Image regions reference original source representations; absence
of an OCR engine yields partial coverage and `ocr_unprocessed`, not fabricated
text or successful OCR. A separate injected/simulated OCR output contract must
preserve its diagnostics/partial state. Parser dependencies and source durability,
retention and metadata ownership remain external/optional.

Acceptance must use the synthetic PDF's actual external parser output, then
project/index/retrieve/resolve under explicit scope. Deterministic layout fixtures
and OCR simulation are additional contract checks, not substitutes for that path.

## Shared typed source materialization contract

Move permission/reference/publication admission into one generic source.Reader
with typed thin Catalog and typed Loader payloads. Reader validates the full batch
of exact identities before any payload consumer, then validates/copies payloads
with explicit host callbacks and freshness gates. It returns owned materialized
records or no payload on any failure. Scope/schema/pinned-publication checks remain
before loading; latest fallback, retries and blob/storage ownership are excluded.

Document hydration specializes this reader with retrieval.Document validation and
metadata/history copying. Text, table, image and layout source hosts may supply
other typed payloads without converting bytes or nested layout into document strings.
Remove replaced document-specific Catalog/Descriptor/PayloadLoader/Hydrated/request
contracts; do not retain aliases or duplicate admission implementations. Host must
supply truthful exact-reference loading and pure/thread-safe payload validation and
copy callbacks. This API does not authenticate raw input or provide a blob service.

## Retained layout resolver contract

The layout resolver specializes source.Reader with Retained payloads: canonical
original document/page/cell/image location, original text or media bytes, page word
geometry where present, observed coverage/diagnostics and separate derived mapping.
Payload reference/geometry is validated after scope/revision admission; no latest
fallback. A derived description must be support-only DerivedContent referencing the
same original artifact, not substituted original bytes/text.

Requested text spans validate against retained UTF-8 text. Page/region locators
must match retained page geometry; a region selects only intersecting word evidence
and reports original byte mappings, rather than assigning the whole page text to a
region. Cell locators match the exact logical cell; merged-cell duplicates collapse.
Image region must be contained in the retained original extent. Resolver returns
original media bytes plus requested/retained geometry, not an implicit image edit.
Document-level binary resolution returns the exact retained representation. All
results retain partial coverage/diagnostics and independent byte/metadata snapshots.
Any malformed/denied/missing/revoked entry makes the batch fail without output.

## Typed OCR observations before implementation

OCR observations are explicit optional host/adapter outputs tied to the exact
original image locator and a host transformation fingerprint. States are
unobserved, recognized, unreadable or unsupported. Only recognized carries text;
text is derived evidence supported by the original image, not original image bytes
or a guessed byte span within PDF. Invalid state/text/reference combinations fail.

Applying observations is a bounded pure transformation over an owned layout
snapshot under caller context. It checks exact image identity, duplicate/unknown
observations and preserves original text/geometry/bytes. Unreadable records
ocr_region_unreadable; unsupported remains explicit; recognized text does not
implicitly promote partial page/document coverage. Core supplies no OCR engine or
accuracy guarantee. Deterministic OCR simulation must accompany actual parser
integration and remain labeled as simulation, with coverage carried through host
chunk metadata/artifacts and later source manifests.

## Layout projection contract before implementation

Projection emits typed evidence for retained normalized page text, each original
logical cell once, and image OCR/description when explicitly available. Source
locator and observed page/document coverage/diagnostics stay attached independently
of retrieval rank or host metadata. Whole page text has exact UTF-8 mapping; cell
quotes retain original support with unavailable byte precision; image OCR/descriptions
are derived support-only evidence. Pixel-only images without text do not become
fake searchable descriptions. Host projects its own types and chooses the target.

Optional image text callback receives context and the original read binding and is
gated before/after. It returns owned derived evidence supported by that exact image.
No model/observer is required for page/cell projection. Source supports and partial
coverage must survive chunking/index/retrieval/rendering through host-owned metadata;
projection must not promote OCR-unreadable pages or discard diagnostic provenance.

## Modality retrieval-to-retention integration acceptance

The actual parser fixture must connect returned artifact supports to the scoped
layout resolver, using exactly the retrieved r1 locators. Retention supplies page
text, original cell text/extent and original PNG bytes independently of indexed
image descriptions. Resolving a derived image snippet returns the original PNG
and separate derived text; resolving a cell returns its original logical cell.
With r2 also retained, deleting or denying r1 must fail before payload loading;
no latest substitution or partially resolved artifact batch is acceptable.

## Durable lifecycle snapshot contract before implementation

Lifecycle persistence uses an exact schema-identified namespace snapshot containing
all retained manifests and active source publication pointers. A store loads an
owned validated snapshot and replaces it atomically only when its generation equals
the supplied expected generation; a successful CAS increments generation once.
Generation zero denotes a previously absent namespace, not permission to overwrite
an existing state. Active pointers reference published manifests in the same source;
tombstones are publications and retain a read barrier independent of cleanup.
Manifest identity includes source revision/content/transform/access fingerprints,
idempotency key/payload fingerprint, target readiness and complete artifact/support
inventory. Failure/unknown/canceled states retain an explicit confirmed checkpoint.
Publication requires every required target ready, unless partial publication was
explicitly selected and its absent targets remain explicit in the manifest.

The optional filesystem reference store must persist across a fresh store instance,
serialize CAS across processes, atomically rename a synced temporary snapshot and
sync the containing directory. Lock contention fails explicitly rather than hiding
workflow retries. Invalid/corrupt snapshots are not replaced or treated as absence.
This storage contract supplies durable state, not target staging, lifecycle workers,
a scheduler or a substitute for persistent dense/tensor integrations.

## Explicit lifecycle executor operations before implementation

An executor registers typed target Stage/Inspect ports and a payload capture and
fingerprint validator. Prepare persists a full planned inventory, validates the
expected active source publication and detects namespace-wide idempotency-key
conflicts. Same-key retries resume the original operation; changing identity,
expected publication, profile, target/artifact/support plan or payload is conflict.

Stage durably marks a target unknown before calling it once. On restart an unknown
target requires explicit Inspect/Reconcile; it must not be blindly staged again.
Only confirmed ready with the exact planned revision advances readiness. Context or
port failure after dispatch retains an unknown checkpoint and returns uncertainty.
No hidden workers/retries, detached cancellation contexts or inferred rollback.

Publish atomically changes the manifest and source pointer under both namespace
generation CAS and expected active source publication. Required targets must be
ready in default mode. Explicit partial publication freezes absent target state;
a late stage result cannot silently change an already published manifest. Tombstone
publication can precede physical cleanup. Reads/target cleanup and bootstrap remain
separate operations to be implemented against this durable publication contract.

## Durable cleanup operations before implementation

Publication records its host-clock acknowledgement time. Cleanup derives a durable
job from the owner's retained expected-publication ancestry, never from new chunk
IDs or all sources absent from a delta. Jobs name exact retired manifest/target
inventories. Current publications and newer planned source artifacts are not retired.
Unknown/bootstrap-unmanaged artifacts are not inferred; their inventory acceptance
remains a separate bootstrap operation.

Begin persists the job and cleanup-pending checkpoint. Attempt dispatches one due
cleanup after durably marking it unknown. Unknown outcomes require explicit Inspect;
confirmed pending applies host-provided capped backoff. Deadline is measured from
publication acknowledgement. Overdue stops further ordinary destructive dispatch;
explicit recovery can continue after backend restoration, preserving the read barrier.
Target cleanup must remove only exact managed supports, preserve shared/host-owned
facts and honor concurrent write fencing/retained snapshot policy. That target
contract must be exercised by real adapters; the coordinator does not invent it.

## Managed lexical target before implementation

The reference managed lexical adapter owns staged revision-bound document snapshots
and uses actual BM25 for retrieval. Its supported storage profile is in-process
memory, not durable lexical persistence. Durable manifests remain external through
lifecycle.Store. A missing retained revision returns snapshot unavailable, never
latest. Cleanup removes exact retired inventories under the same adapter lock as
staging, fences changed active publications and never deletes other sources.

CapturePublication loads one validated namespace inventory before fan-out, requires
all requested target revisions ready, excludes tombstoned sources, and hashes the
active publication inventory into a stable logical reference. An explicitly pinned
empty inventory is a valid complete-empty snapshot, not current/live access. Strict
capture rejects unavailable requested targets in partial publications; broader
explicit partial capture remains separate acceptance scope.

Managed lexical queries require this pinned profile. They check binding/schema
before store access, retain exact selected revision snapshots, apply mandatory AND
query metadata conditions before owned payload projection, and build a single BM25
corpus for the admitted snapshot. Canonical revision/artifact keys prevent document
ID collisions. Freshness is checked before projection and final delivery. Old pins
can finish on captured data; after physical cleanup an unavailable old pin fails.

## Inventory bootstrap before implementation

Inventory import is an explicit non-destructive operation over a validated namespace,
kind (delta/complete), watermark, target coverage, verified published manifests and
opaque unmanaged keys. A host verifier confirms the exact owned inventory fingerprint
and fenced namespace/watermark/coverage; incomplete complete inventories fail before
verification/I/O. Unmanaged records never acquire an invented source manifest.

Import uses namespace generation CAS and each incoming source's expected publication.
It preserves every absent source and all old manifests. A complete verified inventory
can propose missing managed sources with their captured expected publication; the host
must issue explicit tombstones before any cleanup. Delta and incomplete coverage
produce no deletion proposals. Receipts preserve these exact proposals so replay of
an old watermark cannot propose removal of a newer source publication. Receipt/key
reuse with different inventory is conflict. Import never calls target mutation ports.

## Persistent tensor filesystem target before implementation

The optional tensor filesystem profile stores separate thin metadata catalogs and
matrix/content payload files under exact manifest identity. Source namespace and
static target are fixed by configuration; embedding model/config/space/dimension
must match before I/O. Full staging writes a private directory, syncs each file and
directory, checks the still-expected publication, then atomically installs and syncs
the ready directory. Cross-process locks serialize staging/cleanup; corruption is
protocol failure and missing ready state is unavailable/pending, never latest.

Query receives bounded source-reference candidates rather than scanning the corpus.
It validates matrix/space/budgets and original pinned binding before catalog I/O,
applies mandatory AND query metadata conditions before matrix/content payload reads,
and computes native MaxSim only over admitted candidates. Results retain candidate
universe/budget and raw score semantics. Exact cleanup is manifest-bound and fenced
against current publication. Real filesystem restart and faults are integration
acceptance; comparative retrieval quality still requires a separate experiment.


### Portable tensor candidate path

Query contracts are placed in `tensor/query` so candidate composition does not
depend on filesystem storage. Search uses typed retrieval and scoring ports; both
leaves declare and execute scope/publication capabilities. It makes one bounded
candidate call, projects exact source references using owned BYOT metadata, and
makes one candidate-local MaxSim call. No model/parser/runtime is built into this
composition. A sparse BM25 snapshot is the first actual integration path; this does
not replace the separate required persistent dense profile or comparative baseline.

Filesystem target directories are reserved by exact namespace/target/operation
identity. A deterministic private staging path permits reconciling known interrupted
writes without guessing ownership from logical source IDs or arbitrary prefixes.
Unknown bootstrap keys remain outside that managed set. Durability and deletion are
limited to the declared trusted local filesystem profile.


### Persistent dense baseline and joint publication

The first dense managed adapter uses a real local-filesystem index in
`dense/persistent`, on the same explicitly bounded durability profile as the tensor
adapter: flock, atomic directory rename and file/directory fsync. Dense vectors have
separate payload/schema types, `dense.Space` and normalized-dot semantics; no token
matrix union or implicit conversion is used. The adapter performs exact scanning
of admitted records, under an explicit MaxScanRecords limit. It is a persistent
baseline, not an approximate-nearest-neighbor performance claim.

Thin catalog predicates and total admitted-record admission precede payload reads
and metadata Decode/CloneMeta. The payload codec and lifecycle contracts are
parallel to the tensor target, including checksum validation, owned input capture,
registered staging, expected-publication fencing, deterministic interrupted-write
paths, retained reads and exact cleanup. Shared low-level locking/fsync/byte bounds
remain in internal/durablefs. Dense/tensor serialization and scoring are separate;
the duplicated adapter lifecycle code must stay covered by shared profile integration
checks until a justified storage abstraction is introduced.

`lifecycle/integration` executes actual dense+lexical and dense+tensor profiles.
Lexical storage is the declared in-process BM25 profile; dense/tensor are actual
persistent files with durable namespace manifests. Typed batch projection makes the
idempotency fingerprint explicit. Lost-response faults run after an actual target
commit; reconciliation queries actual target state without blindly replaying Stage.
These profiles do not certify the pending dense+graph path or every crash boundary.


### Managed source-supported graph target

`graph/managed` uses real owned in-process graph records plus durable lifecycle
manifests. It preserves separate per-source revisions; losing the in-process index
returns snapshot unavailable rather than substituting current/empty data. Reads
require confirmed published manifests in the durable ledger, not merely staged
records or a manually supplied matching target revision.

Logical node/edge IDs are supplied by the host. Shared equal canonical payloads
combine admitted original source supports; differing canonical payloads produce
conflicts and have no implicit winner. Labels are treated as canonical sets.
Cleanup releases one exact source inventory; another source's support preserves a
shared fact. This adapter does not infer identity from display names or implement
an ontology/alias policy. Those typed extraction/resolution contracts remain a
separate required part of task12.

The first profile permits each managed record's supports only from its own
namespace/source/revision/access identity (original transformation/representation
may differ). Cross-source summaries must obtain separately admitted inputs and
will need their own supported storage/summary contract. This restriction prevents
exporting unadmitted support references by treating a public aggregate's metadata
as authorization for all contributing sources. Host-owned foundations use explicitly selected immutable basis IDs and separate
HostBases support fields; they never fabricate managed source references.

Both node and edge filters admit traversal as well as output. A forbidden node is
not used as a bridge. Thin attributes are tested before payload clone callbacks;
all callbacks and final delivery are freshness-gated. Traversal is bounded by
explicit depth/visited-node/edge limits, cycles use visited sets, overflow fails
without partial payload delivery. FindByIDs shares admission and loads only allowed
requested IDs. Paging is unsupported in this initial profile and rejected explicitly.

### Host basis, retrieval projection and bounded lifecycle files

`graph/managed.SetHostBasis` captures an immutable host-owned graph under an explicit
ID. Reads opt in with `Request.HostBasis`; missing/released IDs fail unavailable,
with no latest substitution. Equal managed and host facts retain both origins;
source cleanup removes only the retired managed inventory. `ReleaseHostBasis` is
an explicit host retention action. Scope and callback freshness gates apply to
both origins. A host basis does not invent source/revision locators.

`graph/managed.Backend` projects admitted traversal into retrieval ResultSet.
Default node projection is score-absent and preserves traversal order. Conflicts
return `ErrConflictingFacts`; an injected host projector must make any resolution
policy explicit. Projection and metadata capture are freshness-gated before and
after callbacks. Source-support export through every retrieval transformation
remains a separate unfinished requirement.

`filestore.New(root, maxSnapshotBytes)` requires a positive finite host budget.
Reads consume at most budget+1 bytes and reject oversized files as protocol errors.
Writes reject oversized captured snapshots before replacing the durable file.
Hosts must choose a sufficient shared budget; a smaller reader does not truncate
or silently interpret an existing publication. The fixture budget is not a core
default. Lexical reads also verify publication confirmation and exact inventory in
the durable ledger before cloning staged payloads, including manually pinned tuples.

### Shared recipe budget ledger

`recipe/budget` is an attempt-local shared ledger with explicit injected limits,
clock/deadline and required/advisory cost policy. Admission atomically reserves one
call and maximum token/cost usage before dispatch. Failed admission spends nothing;
admitted calls are never refunded. Single-settlement leases return unused usage
only when actual usage is known; unknown/error-without-usage retains reservations.
Actual overrun fails and retains accounting rather than reporting compliance.
Unknown advisory pricing is explicitly diagnostic and cannot promise a cost cap.
All counter comparisons avoid integer overflow. This does not implement model
pricing, billing, hidden retries, dispatch, evidence or the recipes themselves.

### Three optional bounded text recipes

`recipe` implements explicit single rewrite, multi-query and depth-one decomposition
with the existing typed backend/request projection and RRF. Planner emits text only;
assessor selects executed indices and signals sufficiency. Original intent/meta,
preplanned constraints and access/publication binding remain captured. Core ownership
is explicit through BYOT clone policies; native score observations and every query's
source locators remain in contributor evidence. Scoped actual BM25 integration covers
all three paths; scripted planning is reserved for deterministic contract fixtures.

Each owned call is reserved before dispatch and settled with observed model usage or
conservative unknown usage. Earlier parent and real attempt deadlines propagate to
ports, while the injected clock independently tests budget deadlines. Final bounded
partial assembly uses already captured identities/metadata, avoiding host callbacks
past the attempt deadline. Parent cancellation and revocation suppress all side
outputs; malformed outputs and overrun remain errors. The declared text profile
rejects precomputed vector/graph options rather than misusing them for rewritten text.

Real model adapters, immutable recording/export, graph recipes and comparative
experiments remain separate unfinished acceptance requirements.

### Immutable evidence and explicit recording

`evidence.Capture` snapshots supplied stage observations into owned canonical bytes.
Scores retain native semantics; absent/unknown does not become zero. Typed wire
fields never include arbitrary metadata, raw auth, error strings or free-form
diagnostics. Identifiers and numbers require explicit export decisions; query and
per-document snippets are separate opt-ins. Scope metadata schema/codec and
SourceAdmission mechanically gate scoped hits and every original support before
hit-ID policy callbacks. Publication membership alone cannot authorize a support.
Required capability, missing observation and privacy errors remain distinct.

The recording wrapper executes retrieval once and snapshots BYOT results with
CloneResult. Disabled skips capture/sink; best-effort retains retrieval with failed
receipt; required failure preserves result/record but returns an error. Scope failure
suppresses all outputs in every mode. Record wire validation does not certify a
producer's export policy or a sink's durability. Automatic execution/recipe stage
observation adapters and complete cross-capability acceptance remain unfinished.

### Recipe recording integration

The optional `recipe/recording` adapter executes exactly one bounded recipe and
projects its observed retrieval stages plus selected fusion stage into immutable
`evidence.Record`. It preserves captured publication/source identities and native
versus RRF score semantics. Scope/source admission and export allowlists run under
the original binding. Unsupported required judged labels fail before dispatch;
the adapter does not infer grades. Sink errors do not retry execution. Revocation
at source admission or sink delivery suppresses the complete returned envelope.
Unknown accounting and integers beyond exact float64 integer range are exported
as unavailable. Failed recipe attempts currently do not retain a stage journal;
that missing observation remains explicit and must be addressed for full evidence
acceptance. Full locator/contributor wire associations also remain pending.

### Document provenance through composition and storage

`retrieval.Document.SourceMapping` is an immutable `source.MappedText` addressing
exactly `Document.Content`; zero means unobserved precision. `SourceSupports`
retains original locators independently of winner metadata. `SourceLocations()`
returns an owned, deduplicated union. ValidateDocument rejects stale mapping text
and malformed supports. These values do not authenticate a source: host support
admission remains required and is distinct from metadata filtering.

Default grouping joins complete mappings, marks separator bytes as derived, and
retains all source supports. If any nonempty fragment lacks a mapping, the merged
mapping is unobserved; known supports remain and complete coordinate coverage is
not claimed. Custom grouping retains input supports automatically and must return
a valid new mapping or explicitly clear it when rewriting content. Dedup/max-score
merge and RRF retain the winner's content mapping and the supports of every losing
contributor. ResultSet/cache/snapshot/candidate/recipe/hydration paths copy support
slices; MappedText remains immutable. Automatic rendering slices the document's
mapping at UTF-8 boundaries. A string-only snippet callback clears precise mapping
while retaining supports; a typed mapping callback can supply explicit transformed
mapping and may add supports.

Persistent dense/tensor records now accept SourceMapping and serialize it inside
the checksum-covered payload. Stage and read validate matching content and source
namespace/source/revision/access identity; original transformation/representation
can differ from the index representation. Query results also retain the indexed
record's document-level locator. Managed lexical capture retains mapping and source
supports and rejects claims about another source revision/access identity. Recipe
support ports must confirm every attached document support before evidence is
captured, assessed or fused; missing confirmation fails closed. Original geometry
is not inferred from content or IDs. Graph projection and complete automatic
locator/evidence wire integration remain outstanding.

### Graph fact-to-document projection contract

Managed graph backend projection must declare each emitted document's contributing
facts by typed node/edge identity. The backend captures the admitted fact support
inventory before invoking host projection, validates all claimed fact identities,
and attaches every original support automatically. Unknown facts and fabricated
source locations fail closed. Projectors may supply a content mapping only when
its locators belong to the declared facts' admitted supports. Display/document IDs
are not assumed to be graph fact IDs. Default projection declares each returned
node explicitly and remains score-absent. Host-owned bases with no source references
remain source-unavailable; no invented source citation is attached.

### Typed extraction identity-resolution contract

`graphingest/resolution` consumes BYOT entity kind/relation kind/attributes with
local mention IDs and original locators. Constructor requires explicit host
ontology validators, namespace/alias identity policy, relation key policy, attribute
cloning/equality, source admission and finite batch/support bounds. Admission
validates the complete structural batch and all original supports before invoking
identity/attribute ports. Unknown namespace is an explicit ambiguous decision in
the reference fixture, never an inferred production namespace. Equal canonical
namespace/key identities group variants; equivalent attributes union source refs;
conflicting attributes/kinds remain separate variants with no automatic winner.
Relations with ambiguous endpoints remain unresolved and still pass schema/ontology
validation. Canonical IDs are deterministic framed tuple hashes; ontology/policy
identities track decision configuration separately from stable canonical IDs.
Result owns attributes/support slices and is suppressed on cancellation/revocation.
Groups are canonically sorted; variants/supports retain encounter order. Consumer
must apply explicit conflict/materialization policy for canonical graph payloads.
Source extraction/model adapter, durable decision history and materialization remain
outstanding; this resolver does not impersonate those deliveries.

### Resolver-to-managed-graph materialization contract

Materialization is an explicit source/revision-bound operation. It selects only
variants whose original supports belong to the configured namespace/source/revision/
access identity, admits those supports before projection, and returns a planned
lifecycle manifest plus owned managed graph payload. It does not write or publish.
Canonical node/edge IDs come from resolver groups. Host projects BYOT kinds and
attributes into schema-validated graph labels/types/metadata. Ontology/policy identity
must match the resolved result; their identities are included with the declared
transformation in its fingerprint. Original supports remain in the artifact inventory.
Two conflicting variants for the same source/revision fail explicitly; conflicts
across sources remain separate managed versions and are reported by managed reads.
Relations require source-supported endpoints; missing closure fails rather than
fabricating endpoint support. Unresolved source mentions prevent claiming a complete
materialization. Each source's plan uses its own expected publication/idempotency key;
only the lifecycle executor performs stage/CAS publication and preserves old manifests.

### Bounded model extraction adapter contract

An optional extraction adapter admits typed source snippets under mandatory metadata
scope and original source/quote admission before any model dispatch. The injected
model client receives only snippet ordinals/text, ontology/config identity and
reserved token limits; it receives no access metadata, binding, credentials or source
reference objects. Its typed output can name local entity/relation mentions, domain
kinds/attributes and supplied snippet ordinals, never canonical identities or source
revisions. Core derives original locators and namespace from admitted snippets;
conflicting/unknown namespaces remain ambiguous. Schema/ontology validation and
endpoint closure precede delivery. A shared atomic budget reserves one model call
before dispatch, settles usage even on error, and forbids hidden retries. Host
provides pricing/input token counting and a client that enforces supplied input/output
limits. Unknown accounting remains conservatively reserved. Earlier parent/attempt
deadline and revocation suppress output. Provider transport and actual quality
experiment remain separate acceptance requirements.

### Structured provider transport

The optional provider module now exposes a generic `structured.Client[T]` plus
`structured.Extractor[TKind, TRel, TAttr]` binding to core extraction. Each call uses
one HTTP POST, explicit host model/instructions/schema, a provider output token cap,
bounded request/response bytes and an attempt timeout. Redirects are disabled.
Duplicate JSON members, excessive nesting, trailing data, invalid UTF-8, missing or
inconsistent usage and excess tokens fail explicitly. Domain JSON is validated by
the host's executable schema before typed `UseNumber`/unknown-field validation.
The host must validate the same immutable schema supplied to the provider and count
the exact model request including message framing, instructions and schema. Core
source/metadata admission and ledger reservation precede transport. Protocol errors,
refusal and incomplete output preserve trustworthy actual token/cost usage; unknown
accounting retains reservations. No raw content or credentials appear in errors.
Real HTTP integration covers core scope denial before dispatch, original locator
propagation and settlement; its scripted response/tokenizer does not prove live
quality or exact provider token counting. Live experiment remains required.

### Durable resolution decisions

Resolver results retain per-input entity/relation traces, before alias grouping:
local mention, explicit resolved/ambiguous decision, canonical identity, endpoint
identities/relation key and original supports. Optional `resolution/history` captures
complete typed input/result, ontology/policy/extraction identity and host-declared
predecessor. Immutable JSON snapshots own BYOT data. The filesystem profile publishes
content-addressed records with synchronized payload and atomic no-overwrite link;
identical concurrent appends are idempotent and previous records remain unchanged.
Support inventory participates in the filename and is reauthorized before payload
reads, then checked against decoded input/decision/result evidence. Snapshot references
belong in the host durable source catalog; graph publication and history append remain
separate explicit operations. No latest pointer, workflow, source retention service or
automatic deletion is introduced. Current source revocation does not become a
historical access grant. Full extraction-to-publication experiment remains pending.

### Model-free local graph expansion

Optional `recipe/graphexpand` binds one managed BFS traversal to a shared budget
ledger, finite depth/node/edge limits and attempt deadline. Host seeds, direction,
filters and flat per-attempt prices remain explicit. Scope/publication/traversal
admission precedes pricing and reservation; node/edge admission precedes traversal
and payload cloning. The reference Team A→Service→DB case uses depth 2 undirected
traversal, with the target visited set cutting cycles. No model or answer generation
is introduced. Results retain independently owned supports/conflicts and export only
observed original source references, never graph IDs as invented citations. Budget
refusal is insufficient with zero dispatch; cancellation/revocation suppress outputs.
Known flat attempt cost settles even if traversal fails. The actual staged/published
source graph integration preserves original source/revision supports. Comparative
hybrid quality/cost/latency acceptance remains pending.

### Community/global summary recipes

Optional `recipe/graphsummary` accepts explicit host community membership and typed
access/source mappings. Complete batch shape/publication/metadata admission precedes
source/model ports. Community performs one map; global performs two maps plus one
non-recursive reduce. Shared atomic ledger reserves calls/input/output/cost before
dispatch; known failed-call usage settles and unknown usage retains reservations.
Model input contains question/stage/ordinal text only. Core rejects invented snippet
indices, verifies declared member coverage, derives original support union and
requires both communities in reduce. Source admission is rechecked between stages,
after pricing/counting and before dispatch/delivery, preventing deleted evidence from
reaching reduction. Summary text is private; fresh Resolve binds scope predicate,
authorization snapshot and publication and rechecks original supports before exposing
support-only derived mapping. Partial budget stops retain only actual completed,
revalidated community artifacts. The optional HTTP Summarizer implements the matching
model/counting ports. Original source Reader targets require explicit original
transformation inventory in the pinned binding. Actual source/model/graph membership
pipeline and comparative quality/cost/latency acceptance remain pending.

### Retrieval model reservation propagation

Planner and assessor ports now accept `recipe.ModelLimits`, copied from the exact
per-call Quote reservation. Both token maxima must be positive before model dispatch.
Known token overruns are checked independently of whether cost is known; advisory
unknown pricing keeps conservative reservations and does not disable token limits.
Optional structured HTTP bindings project only effective query text and admitted
query-index/document-content evidence. They validate strategy cardinality, text byte
bounds, aggregate evidence bounds and executed-index selection. HTTP admission runs
after token counting to suppress model dispatch on host revocation in that callback.
Transport limits, executable schema, tokenizer, instructions and actual prices are
explicit host inputs. HTTP fixtures prove protocol behavior, not live quality.

### Failed recipe observations

Recipe.Run keeps its error-payload suppression contract. Explicit RunObserved
retains only owned already captured queries, actual dispatched stages and the settled
ledger snapshot on ordinary attempt errors. Failure/StageFailure mark that envelope;
Selected is cleared. Protection failures and parent cancellation suppress it entirely.
Enabled recording uses this observational port, re-admits original metadata/supports,
clones owned observations and exports immutable bytes. Pre-attempt errors without a
journal remain explicitly missing; nothing infers earlier hits or zero usage.

Result.Fusion declares not-run/missing-observation/observed independently of whether
Selected is empty. Failed retrieval without a validated captured query retains its
dispatch and missing observation; failed planner/assessor retains a dispatched empty
model hit set and available usage, without fabricated selection/fusion. Default
privacy omits model/backend/error payloads. RRF converts k/rank operands to float64
before denominator addition so a valid MaxInt k cannot silently overflow into zero
scores; very large k can create floating-point ties resolved by stable ordering.

### Immutable original locations

Evidence.Hit carries explicit original locators plus automatic typed document support
mapping. Source/metadata admission precedes locator validation and export policy;
every location must reference the admitted hit source inventory. Policy.AllowLocation
explicitly permits geometry and page/table/element labels, in addition to numerical
and complete source-identifier permission. Callback freshness is revalidated. Default
privacy omits known geometry; unavailable mapping remains unavailable without guesses.
The wire excludes AccessFingerprint and arbitrary metadata, validates canonical
locator union/geometry and source association, and owns immutable serialized bytes.
Recipe recording forwards query supports and the dedup contributor support union.
Exact per-query contributor associations remain a separate pending wire deliverable.

### Query contributor association

Evidence contributions carry executed query ordinal, original document identifier,
one-based position in the captured query result list and original locations. Native
stage document rank/score remains a distinct observed value. Recording verifies the
query/document/list-position/support tuple against captured query observations,
then emits native-query and fused contributor associations. Generic producers attest
execution ordinals; wire decoding validates shape and association, not authenticity.

Explicit AllowContribution, permitted numbers and document IDs gate association
export. Location permission remains independent. Callbacks receive owned support
slices and freshness checks; raw query/domain/auth metadata is excluded. Invalid or
duplicate tuples and contributor locations outside the hit location union are
rejected. Typed contributor and immutable wire association are both retained after
dedup; a location union no longer substitutes for the per-query association.

## Actual partial publication and committed cancellation acceptance

The real dense+lexical, dense+tensor and dense+graph fixtures now verify explicit
Manifest.Partial publication with only dense r2 confirmed. The other target is either
pending (never dispatched) or unknown (actual stage committed but its response was
lost). Published per-target state and empty unconfirmed Revision stay frozen across
replay. Strict joint CapturePublication rejects the missing target; an explicit
dense-only capture selects r2, while an earlier joint pin continues to select r1.
No old secondary revision is silently substituted into the new publication.

This verifies publication behavior, not a complete partial execution/export path.
Explicit partial fan-out capture and automatic association of publication coverage
with retrieval envelopes and immutable evidence remain required by the original scope.

A second real-backend fault fixture cancels the context immediately after durable
publication CompareSwap returns a committed snapshot. Publish reports both context
cancellation and uncertain acknowledgment with its Published checkpoint. Fresh reads
select r2 in both targets; retained authorized reads select r1. Fresh-context Publish
replay performs no further CAS or staging. Cancellation does not roll back publication.

## Explicit partial capture contract

CapturePartialPublication observes one durable namespace snapshot. If any active
source lacks a ready requested target, the complete configured target branch is
excluded; another source's ready records do not disguise that branch's incompleteness.
The result pins only admitted ready branches, retains static excluded target labels,
and never substitutes an older source revision. Default CapturePublication stays strict.

The immutable publication carries excluded labels into binding fingerprints, pre-I/O
adapter admission and retrieval/evidence coverage. Target adapters explicitly admit
that publication; an excluded or unrecognized target cannot produce complete-empty.
PartialReadNode may skip the excluded branch during preflight. Runtime failure and
revocation remain non-skippable. Evidence preserves partial coverage even with no hits.

## Verified ingestion reuse decision

Executor.CheckReuse is a read-only decision for one exact desired source identity and
complete named target profile. Matching content alone is insufficient: revision,
transformation and access identity must also match the active non-tombstone publication.
Partial or incomplete target profiles cannot yield reusable. Every named target is
inspected once against its retained manifest inventory; no Stage, publication or retry
runs. A second durable snapshot read fences concurrent lifecycle changes. Inspection
failure/uncertainty and cancellation are errors, not successful reuse. A confirmed
result records the observed active publication; it does not authorize a later mutation
without the usual expected-publication CAS. Scheduler and rebuild policy remain host-owned.

## Persistent original-support inventory

Dense and tensor catalogs retain complete lifecycle.Artifact entries, including each
original source-support set, alongside payload descriptors. Storage identities are
ragy.dense-index/inventory and ragy.tensor-index/inventory. The catalog validates
artifact identities, support identities/uniqueness, exact descriptor membership,
attributes and payload digests before it can acknowledge readiness. Inspect compares
retained artifact/support sets with the requested manifest; read and cleanup compare
with durable managed ownership. Ordering is immaterial to inventory equality.

Artifact.Validate and SameArtifactInventory centralize these semantic checks. Equality
is not source authentication: callers must validate inputs and retain host/source
admission. Full namespace coverage and a fenced bootstrap observation still require
an actual backend inventory verifier; the persisted supports alone do not attest them.

## Fenced backend inventory observation

InventoryObserver verifies one named target and calls next synchronously exactly once
while holding its mutation fence. It never stages, adopts or deletes inventory data.
FencedInventoryVerifier owns its registration map, validates the exact configured
profile, acquires observers in sorted target order and confirms the original envelope
fingerprint only while every target's fence is held. A skipped/duplicate callback or
nested observation failure produces no confirmation, including a failure swallowed by
an outer observer. Ports are trusted host Go implementations under this contract.

Dense and tensor observers share the cross-process target.lock used by Stage/Cleanup.
They validate retained catalog identity/payload/artifact/support inventory and actual
payload digests. Complete inventory accounts for every immediate storage key except
the lock file. Unlisted directories, staging/retired directories, old formats and
opaque host files must remain explicitly unmanaged; they are never decoded to guess
ownership. Delta may omit other keys. An unmanaged key is an immediate basename,
not a filesystem path, and must actually exist without overlapping a managed manifest.

Enumeration and verified payload counts are bounded by dense MaxScanRecords and
tensor MaxRecords; byte reads retain existing catalog/payload limits. Insufficient
bounds reject observation. These limits may require the host to configure a larger
bounded inventory profile than an individual query profile. Confirmation is a fenced
observation point, not a lease or an atomic transaction with Bootstrapper's later
manifest-store CAS. Subsequent mutation follows expected-publication and normal read
freshness/snapshot checks. Complete receipts only propose removals; they do not delete.

## Retained lexical/graph inventory fences

Managed lexical and graph targets expose InventoryObserver(maxEntries, maxRecords)
with explicit positive bounds independent of query options. Observation uses TryRLock
on the actual Stage/Cleanup mutex, returning Conflict without waiting behind a writer.
The fence remains held through the next observer and is released on error/cancellation.
After the callback, cancellation is checked again by all supplied target observers.

Each retained source version is an opaque manifest:<operation-id> key until the input
supplies its exact published manifest identity, payload fingerprint and original support
sets. The observer verifies actual retained record references without executing host
metadata/model callbacks. Missing memory state returns Unavailable even when the durable
ledger says ready. Complete inventory accounts for every retained version; delta can
omit other versions. Graph host foundations use host:<basis-id> keys in Unmanaged,
never managed manifests. Observation holds the same fence as Set/ReleaseHostBasis and
cannot change their ownership or authorize source cleanup of host facts. Entry and
verified-record caps fail explicitly before confirmation. These targets retain their
in-process durability contract; bootstrap does not turn memory into persistent storage.

## Destructive lexical cleanup admission

Lexical Cleanup validates the registered durable owner and exact retired identity plus
artifact/original-support sets before mutation. Dispatch requires the registered
owner/retired/target item to have an unknown checkpoint, written by Cleaner before
calling the backend. InspectCleanup requires registration and exact inventory without
issuing deletion. Source expected-publication fencing is checked under the same adapter
mutation lock. A changed inventory or raw unregistered request is Protocol, not complete.

Cleaner recovery separates backend effect from acknowledgement. A committed deletion
with a lost response remains unknown in durable progress; a new Cleaner performs one
actual InspectCleanup and cannot repeat deletion through Attempt. Physical cleanup
failure does not reopen tombstoned reads. Fake-clock deadline/backoff is host-driven;
explicit overdue recovery retains NextAt, so restoring the backend does not bypass an
existing confirmed delay. Every read still needs live host authority independent of
publication retention and cleanup progress.

## Durable staging checkpoint uncertainty

An unknown dispatch checkpoint must be durable before target Stage. A lost CAS response
can therefore mean either pending/unwritten dispatch or unknown/no target effect. The
host reloads durable state and reconciles unknown with actual Inspect; a pending result
allows a subsequent explicit Stage. A target commit followed by failed ready CAS remains
unknown and must be inspected rather than blindly staged again. If ready CAS committed
but its response was lost, fresh state is ready and Stage/Reconcile replay executes no
target I/O. Candidate manifests returned alongside an error do not prove durable state.

Prepare and publication acknowledgement use the same rule: replay loads known state.
A committed plan/publication needs no repeated CAS; an uncommitted one may proceed by
an explicit host continuation using expected-publication CAS. Target data remains
staged outside default read visibility until confirmed logical publication. Old pins
retain exact revisions under live host authorization. Fault acceptance distinguishes
checkpoint uncertainty from target/process durability guarantees.

## Cleanup checkpoint CAS uncertainty

Cleanup job creation, unknown destructive dispatch, per-target completion and whole-job
completion each persist through namespace CAS. The coordinator never dispatches backend
cleanup after a failed dispatch checkpoint acknowledgement. If that CAS committed, a
fresh Cleaner sees unknown and performs InspectCleanup before any explicit continuation;
if it did not commit, the item remains waiting. A lost completion checkpoint response
is resolved against durable item state and the actual backend effect, not the candidate
job returned with an error. Confirmed done items replay without deletion or inspection.

Whole-job completion is part of the same snapshot CAS as the last target's completion.
An acknowledgement error does not undo tombstone visibility or authorize another
physical deletion. Job creation replay reuses committed ownership and preserves the
captured retired inventory. Recovery of a waiting item still respects host NextAt;
physical deletion stays explicit and bounded. These contracts apply to the supplied
persistent/in-process targets and do not imply process persistence for volatile data.

## Persistent physical process-failure boundaries

Dense/tensor staging is outside publication visibility until the final installed
catalog is ready. Partial payload sets and complete catalogs in an owned .stage directory
remain pending after process loss; only explicit continuation rebuilds that exact
registered operation. An installed synced catalog with intact original supports and
payload digests can be inspected ready after restart without inferring publication.
Old published source revisions remain separate directories and retain exact provenance.

Cleanup first retires the exact old manifest directory. A retained .retired directory
is waiting, not physically complete, and requires explicit due cleanup continuation.
Absent installed/staging/retired directories confirm done through InspectCleanup without
another deletion. The durable unknown dispatch remains the recovery anchor when the
process dies before acknowledgement. Separate-process acceptance stops at observed
filesystem states using a test-only context and os.Exit, bypassing Go defers. It does
not add production hooks or claim simulation of storage hardware power loss.

## Admitted persistent payload read port

lifecycle.PayloadReader is an optional bounded read port for an already-admitted managed
index artifact. PayloadRead carries its exact reference, local payload path and byte
cap. The reader must honor context, own bytes and avoid retries; it confers no write,
delete, source adoption or publication authority. Dense/tensor default to the bounded
local file reader. Query admission and mandatory filters precede invocation, and target
validation still checks returned digest/reference/schema/vector shape plus final scope
freshness. Byte-cap/context checks also apply to host output. Staging and inventory
inspection retain their independent physical-file verification.

## Unified planner filter admission

PrepareRead validates and intersects query and immutable Plan.Filters against each
provider schema before applying the mandatory binding. A schema incompatibility is a
protected unsupported capability during admission; a leaf's execution error remains
non-skippable. Empty plan predicates are identities and contradictory conditions retain
empty intersections. Direct persistent leaves and bounded tensor candidate search use
this same path, rather than a preliminary intersection with different error semantics.
Graph traversal also classifies unmapped node/plan conditions as capability failures
before payload projection. A planner condition cannot be ignored by preflight or used
to replace mandatory authorization. No host callbacks execute to negotiate predicates.

## Typed persistent numeric catalog attributes

RawAttributes JSON decode preserves numbers without deciding their declared kind.
A decoded dense/tensor catalog is private owned data; its record attributes are then
validated AND replaced with Schema.NormalizeAttributes output before matching. Merely
checking normalization success leaves json.Number values at the matcher boundary and
produces incorrect numeric equality/exclusion. Metadata projection cannot substitute
for candidate admission. No float intermediary or guessed integer identity is allowed.
Invalid wire integer attributes also fail in capture before target lock/index writes.
The mandatory authorization profile remains scalar Eq/In/And; query operators outside
that profile do not gain permission to become mandatory policies.

## Metadata codec output before transport writes

A custom MetadataCodec supplies serialization behavior, not permission to bypass adapter
schema validation. Supplied vector wire adapters normalize and retain owned encoded
attributes before constructing transport payloads/JSON. Invalid integer, overflow or
non-finite output fails as ErrInvalidArgument before a Client/DB write call; no implicit
retry or partial batch write is introduced. JSON number preservation and typed membership
values are verified at injected transport boundaries. This does not certify a custom
external driver's precision or confer managed lifecycle capabilities on these ports.


## Concurrent recipe attempt accounting

Text Recipe.Run/RunObserved constructs a fresh atomic ledger per invocation and
shares it across that attempt's sequential retrieval/planning/assessment stages.
Concurrent invocations on the same immutable recipe instance have independent
limits and owned request/result metadata. They do not implement an organization
quota. All configured host callbacks must be concurrency-safe; input capture does
not authorize concurrent mutation of caller data.

Community/Global summary accepts an explicit host attempt ledger. Concurrent
variants may share it. An executable quote barrier brings two Community executions
to the last available model-call reservation while the admitted model stays in
flight. Only one dispatch occurs; the loser returns typed insufficient/budget-
exhausted, and the winner settles known usage without outstanding leases or retry.
A separate parallel text Run fixture confirms independent 2-model/2-retrieval
attempt limits, exact settled usage and result/host/caller metadata ownership.
These are actual recipe contract tests with injected host ports, not live quality
experiments or a new global quota contract.


## Text comparative experiment scoring boundary

The external recipe_comparison consumer contains fixed corpus/qrels and an offline
evaluator for a complete baseline/three-strategy by five-query observation grid.
It computes per-query Recall@3 and MRR@3 over the four answered queries, separately
counts selected evidence on the no-answer query, retains raw usage/timings and
rejects incomplete/duplicate/foreign captures. Unknown usage cannot pass budget
acceptance; failed executions cannot support recommendation. Both baseline and
recipe budgets/failures matter, so degrading or failing baseline cannot manufacture
a gain. Input/output/cost/call/deadline reference caps are explicit.

Numeric thresholds are calculations, not capture provenance verification or
authority to change the default. The report always retains baseline default and
records small-dataset limitations; no latency percentiles are manufactured.
Contract-fixture scorer tests do not constitute the required real model experiment.
Actual BM25/model capture, configured exact tokenizer and live comparative runs
remain outstanding. The example is consumer tooling, not an experiment framework
or model/tokenizer dependency added to runtime core.


The consumer model_ports binds actual optional structured planner/assessor clients
to a host attempt context with an existing deadline and the hostCounter executable.
Only complete serialized provider requests enter stdin; model/tokenizer identities
and a positive token count must match its bounded receipt. Process environment
excludes inherited credentials. Local computation is bounded by two seconds and
parent deadline, and output by 512 bytes, without shell/retry or raw child errors.
Executable schema validators require all planner/assessor output members, including
an explicitly present sufficiency boolean. Qualification of model framing/tokenizer
remains a host task; scripted HTTP/fixed-token subprocess fixtures are protocol
evidence only. Full BM25 capture orchestration and live experiment remain open.


## Complete text capture orchestration

recipe_comparison now exposes explicit live capture and separate offline scoring.
The grid executes five baseline queries and all three opt-in recipes on the same
owned scoped readonly BM25 snapshot. Corpus publication is the fixed corpus hash
and exact original reference inventory; this is not persistent lexical storage.
An adversarial foreign document is denied by snapshot capture. Known original
content/tenant/ID checks validate supports before assessment/fusion.

Per-sample retrieval/HTTP boundaries observe actual calls, independently retained
model stage usage protects failed observations from being mistaken for measured
zero usage, and parent/recipe contexts bound attempts. Failed executions select no
payload and cannot satisfy recommendation gates. Offline validation rejects source
references detached from selected IDs or the exact fixed source/revision/representation.
The live CLI accepts explicit model/tokenizer settings and environment credentials;
no scripted response fallback exists. HTTP/subprocess fixtures exercise the whole
20-observation path but remain protocol evidence. Live acceptance is outstanding.

## Reference graph capture provenance

The external consumer now carries sourceExtraction.Configuration as an explicit
SHA256 identity. Provider factories bind model, endpoint, request template/schema,
limits and cost policy; the orchestration binds qualified tokenizer identity. Core
extraction receives that same configuration identity. Missing/invalid source
configuration rejects before graph target construction. A claimed model differing
from its bound provider rejects before extraction or summary dispatch.

Materialization consumes model-extraction:<configuration> as its incoming
transformation. Existing core materialization fingerprints this together with
ontology and identity policy into graph-resolution:<digest>. Thus changes to model
configuration partition artifact/publication provenance without changing canonical
entity identities or original support tuples. Summary observations preserve their
separate request/counter configuration identity.

Combined source inputs and typed resolution results pass through history.Capture,
FileStore.Append and a newly constructed FileStore.Read before publication. The
archive metadata fingerprints every per-source configuration; caller can provide
an explicit parent reference on recomputation. The one-shot initial capture has
no predecessor. Core archive persistence and ownership remain host-controlled.
The CLI cleans its temporary backend/archive directory after preserving capture
metadata, so the exported history ID does not promise post-command payload retention.
The persistent profile test verifies both old/new entries in one reopened history
store, with new parent equal to old ID and unchanged original supports.


## Parser coverage across generic lifecycle boundaries

Layout parser coverage/diagnostics describe parsing, whereas lifecycle inventory and
retrieval branch coverage describe independent guarantees. Do not merge these states
or infer complete parsing from a ready target. Hosts retain typed parser metadata in
records/source envelopes; manifest payload fingerprints bind the staged records.
The actual PDF lifecycle integration persists partial coverage alongside original
source mappings in dense catalog/payloads, publishes r1/r2 with CAS, then reopens
adapter/ledger and resolves old r1 with exact host retention. Original source blobs
and their retention authority remain external ports; a persisted index does not
automatically establish persistent blob retention.


## Persisted standalone locator envelopes

`source.EncodeLocator` / `source.DecodeLocator` use schema identity
`ragy.source-locator` and the executable `schemas/locator.schema.json`. All tagged
union fields, including inactive zero fields, are present. Decode rejects unknown,
duplicate, missing and case-aliased fields, trailing JSON, invalid UTF-8 and
incompatible schema identities. Geometry and union invariants are checked in Go;
exact UTF-8 boundaries relative to retained text are still checked at resolution.
Neither decode nor schema validation grants source authenticity or authorization.
The envelope contains an access fingerprint and belongs to an authorized storage
boundary; exporting it as telemetry still requires a separate privacy policy.

Typed host extensions use `ExtendedLocator[T]` alongside the canonical location.
Host code owns extension schema, validation, cloning and persistence. Extensions do
not change canonical citation identity or override the retained reference. JSON
field names are explicit `location` and `extension`; no opaque host domain map is
required. Actual PDF/durable retained-revision integration round-trips projected
locators through the canonical codec before resolution.
