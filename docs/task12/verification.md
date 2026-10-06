# Task12 verification record

This is an incremental evidence record, not a completion report. The authoritative scope remains `.cursor/docs/task12.md`; completeness is tracked in `requirements.md` and must be independently audited after implementation.

## Current audit checkpoint

The fixed inventory contains195 mandatory atoms. Historical mechanical checkpoint was187/195 (95.90%). The actual calibrated CLI experiments now satisfy the five previously pending live requirements under task12§7.1; independent completeness review confirms the actual receipts and recalculated metrics. Current source gates, corrected CLI failure paths and document reconciliation are independently verified. Final independent verdict is **195/195 =100%**, U0, unchanged scope: `audits/completeness-cli-profile.md`. Final correctness audit confirms both defects fixed and no unresolved confirmed finding in its reviewed area.

All previously confirmed core findings remain corrected and independently reproduced: exact JSON integers, atomic BM25 updates, owned metadata/freshness, finite score controls, exact cleanup support inventory and tensor recorder association. Actual external joint/cache, persistent tensor, PDF and five independent schema suites remain part of acceptance; evidence is indexed by the195-atom matrix.

Actual text20-row/30-call and graph6-row/8-call comparative captures, reports, frozen calibrated profiles and external graph summary rubric are saved in `results/*-cli-live-*` and the separately reviewed `results/*-cli-v2-*`. Negative quality and failed community validation are preserved, not silently retried/promoted. CLI tokens/pricing/isolation/provider dispatch limitations are explicit; original reference conformance is unchanged. User has existing CLI authorization; no API key or tokenizer acquisition is required for the clarified live acceptance.

Independent correctness review found two consumer defects: early trace failure discarded later returned evidence/usage, and duplicate JSON fields were ambiguous. Current fixes preserve full stdout/stderr, all settlements and tool observations, reject recursive duplicate/case-alias fields, and keep aggregate usage unavailable after malformed/overflowing settlement. Independent8-case adversarial recheck and76actual-trace final-decoder replays passed, including sticky malformed/overflow accounting and Unicode fold aliases. Separate complete v2 runs preserve raw stdout/stderr without overwriting the first runs. Their pre-final-error-guard compiled snapshot is explicitly documented; valid actual traces pass the final decoder.

Final all-module `make lint` and `make test` after Unicode/trace/accounting corrections both terminated exit0: `results/cli-final-lint.txt` and `results/cli-final-test.txt`. Owner sessions63973/25997 are closed. `git diff --check` passed. First corrected-retention v2text/grid evaluation also terminated0 (20rows/30calls,161085input/1216output); v2graph/grid evaluation terminated0 (4preparationcalls+6rows). Previous joint-recorder gates and audits remain historical evidence only. No release, publication or issue closure has been performed.

The sections below are historical implementation checkpoints, not current pending-contract declarations.

## Initial implementation

Implemented:

- BUG-001: number-preserving JSON attributes, exact schema normalization including below-MinInt64 overflow and long fractional decimal rejection; retrieval, graph and stored JSON regression tests.
- BUG-002: BM25 staging and atomic publication with serialized Upsert; failed prefix/token/codec cases, concurrent readers and concurrent Upsert regression tests.
- Filter foundation: `filter.Intersect(schema, conditions...)` and structural `Condition.Fingerprint()`, tested for contradictory scopes, schema mismatch and concurrent immutability. Full scope enforcement has not been implemented.

Evidence:

- `GOCACHE=/tmp/ragy-implementation-go-cache go test -race ./filter ./retrieval ./lexical ./graph ./adapters/pgvector/... ./adapters/elasticsearch/...` passed.
- `GOCACHE=/tmp/ragy-implementation-go-cache make test` passed all current modules and race/example checks. Go emitted non-fatal module-cache permission warnings when resolving example module metadata. This does not establish persistent backend/parser integration or any of the new capability guarantees.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache make lint` passed all current modules. Repeated-literal issues in existing contract helpers/adapters/examples and formatting/style issues in new files were corrected. A subsequent full `make test` also passed on the final state of this initial stage. These are checkpoint gates, not final acceptance of unimplemented capabilities.
- `git diff --check` passed for the tracked diff at this stage.

Remaining:

All seven new capability deliveries, their schemas/conformance/integration/experiments, complete migration guide, and both final independent audits. No release, publication or issue closure has been performed.

## Next implementation step

Implement explicit immutable read access/publication bindings through planner/binder/projector and every leaf/composition path, together with context-aware processing and semantics-aware scores. Use `design.md`, the full requirement matrix and fixtures as the contract baseline. Docker daemon is available for actual persistent backend integration; parser commands are available, but no persistent/parser integration has yet been executed. Final auditors have not been launched because full implementation is not ready.

## Tensor computation increment

Implemented and executed `tensor.MaxSim` and bounded `tensor.Rerank`. The tests
load the specification fixture directly and verify native scores 2/1/-1, ranking,
score semantics, candidate evidence after TopK, and output ownership. Negative
checks cover empty/ragged/non-finite/non-unit matrices, every space identity field,
missing record identity, duplicate candidates, candidate overflow, invalid limits
and cancelled operations. Shared tensor-index contract checks now reject malformed
writes; the reference test index preserves the supplied space when copying records.

Verification after this increment:

- `GOCACHE=/tmp/ragy-implementation-go-cache go test -race ./tensor ./testutil ./filter ./retrieval ./lexical` — passed.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache make lint` — passed, all modules; `/tmp/ragy-task12-lint.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache make test` — passed, all modules/race/examples; `/tmp/ragy-task12-test.log`.
- `git diff --check` — passed.

The Go tool reported sandbox-denied metadata-cache writes while resolving some
workspace example module metadata; commands still exited successfully and builds
and tests ran. This does not substitute for any pending persistent integration.

TENSOR-03 is backed by executable oracle evidence. Other tensor rows remain partial
or pending where they require persistent adapters, generated candidates,
scope/publication integration, ResultSet conversion or comparative experiments.
No final completeness percentage or absence-of-defects claim is made. Both final
independent auditors remain required after full implementation.

## Retrieval score contract increment

Native scores now require declared semantics and allow all finite values. The
zero-value state is rank-only. The old MinSimilarity field and automatic
ClampScore/logistic conversions were removed. An explicit ScoreThreshold names
its value/state/scale, including meaningful zero and negative minima. Numeric
ordering/merge/grouping reject incompatible scales; explicitly chosen RRF uses
its declared rank-fusion scale. Caller-comparator Rerank returns rank-only order,
so terminal TopK no longer reverses the caller ordering.

ScoreHistory retains input numeric observations through grouping, deduplication,
RRF, model rerank and rendering. Its slices are copied; arbitrary BYOT metadata
is not claimed to be deeply immutable. Tests prove native 2/1/-1 validation,
negative/zero thresholds, incompatible/absent rejection, explicit heterogeneous
fusion, raw-score preservation, defensive copies, losing-contributor observations,
and caller ordering surviving terminal TopK. Old invalid-native-score probes were
replaced with explicitly invalid normalized scores. Shared resolver-parity probes
now use the actual returned scale, including genuinely scoreless output.

SCORE-01/02/03 have matching executable evidence. SCORE-04 remains partial until
persistent tensor result integration; SCORE-05 remains partial until immutable
revision-bound/privacy-governed evidence export. Adapter raw-score changes have
unit/module evidence, not newly claimed live storage integration.

Final checks for the score increment, after the last code/fixture changes:

- `GOCACHE=/tmp/ragy-implementation-go-cache make test` — exit 0, all modules with race and all example checks; `/tmp/ragy-task12-test.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache make lint` — exit 0, all modules; `/tmp/ragy-task12-lint.log`.
- `git diff --check` — exit 0.

These are incremental checks. Persistent/parser integrations, comparative experiments,
mandatory access/publication enforcement, remaining issue cards and the two final
independent audits still prevent declaring the full goal complete.

## Access binding and initial target enforcement increment

Implemented `access.Binding`, value-only host snapshots, required freshness
Authority/clock, immutable publication inventory and the scalar Eq/In/And mandatory
profile. `Request.Read` is mandatory; explicit unrestricted live reads replaced
implicit unrestricted zero-value requests throughout adapters/examples/tests.
Planner/binder/projector cannot replace the captured host binding. Existing
lexical/dense/sparse target paths intersect mandatory predicates against their own
schema before dispatch. Unsupported graph traversal and pinned publication are
rejected instead of claiming guarantees through old raw-storage APIs.

Entry and delivery wrappers cover success, empty, partial and error returns.
Host revocation suppresses both documents and BYOT execution outputs. A typed
ProtectionError has sanitized diagnostic text and cannot be rescued into success.
Tests exercise policy revocation at t10 before expiry, expiry at t31, publication
inventory mutation, missing binding before planner, unknown scoped adapter before
dispatch, lexical a-public/a-private/b-public isolation, contradictory optional
filters, replacement attempts by planner/binder/projector and revocation during I/O
with zero secondary rescue calls.

The initial scope increment remains incomplete. Global composition preflight,
explicit partial profile, all operator/target spies, graph pre-expansion checks,
scoped hydration/resolution/caches, per-processor/export gates, public external
conformance and actual pinned lifecycle adapters remain required. CurrentPublication
explicitly makes no managed publication consistency claim. A stored inventory by
itself does not establish persistent backend support.

Final checks for the access increment after all code/test changes:

- `GOCACHE=/tmp/ragy-implementation-go-cache make test` — exit 0, all modules/race/examples; `/tmp/ragy-task12-test.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache make lint` — exit 0, all modules; `/tmp/ragy-task12-lint.log`.
- `git diff --check` — passed after README whitespace cleanup.

This is incremental evidence. The full goal remains unachieved; no final audit,
release, publication or issue closure has been performed.

## Context and freshness across downstream processors/model reranking

Updated the single public processing contracts to `Process(ctx, read, rs)` and
`Rerank(ctx, read, query, rs)`. The chain receives the binding explicitly and checks
before/after every processor. Built-in processors check before host selectors and
merge callbacks. All shipped wrappers and conformance fixtures use the new
signatures. Model reranking independently checks before HTTP dispatch and every
return, including ordinary error/partial paths. Observability forwards the same
binding/context and validates its output boundary.

Executable evidence:

- `TestRevocationBetweenProcessorsStopsNextConsumer` covers both successful and
  error/partial first-processor returns; subsequent consumers see zero calls and
  returned payload is empty.
- `TestProcessorReceivesDeadlineAndCancellationStopsChain` verifies exact deadline
  propagation, cancellation classification and zero subsequent consumers.
- `TestModelRerankerGateBeforeDispatchAndAfterIO` performs actual local HTTP adapter
  calls: already revoked policy causes zero requests; revocation inside HTTP causes
  one request and zero delivered payload. This is adapter-contract evidence, not a
  real-provider quality experiment.
- `TestProtectionClassificationDropsJoinedSideErrors` prevents joined sibling
  errors from carrying unrelated data through a protection failure boundary.

Final checks after the last code/test changes:

- `GOCACHE=/tmp/ragy-implementation-go-cache make test` — exit 0, all modules/race/examples; `/tmp/ragy-task12-test.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache make lint` — exit 0, all modules; `/tmp/ragy-task12-lint.log`.
- `git diff --check` — passed.

SCOPE-09 and SCOPE-17 remain partial for cache/export/recipe and remaining target
paths. No new completeness percentage is claimed. The full objective and both
final independent audits remain outstanding.

## Optional cache implementation checkpoint

The supplied cache is bounded in-process storage and a generic target decorator;
it requires injected clocks, host identity/revision/configuration ports and explicit
BYOT metadata/request ownership. No cache server or background worker is required.

Evidence in `retrieval/cache_test.go`:

- Tenant A/B isolation holds even with the same host snapshot ID. Epoch v8 at t10
  rejects v7 before the 30-second TTL; t31 rejects expired authorization until a new
  binding is supplied by the host.
- Revocation inside cache Load suppresses the hit and causes zero target refreshes
  and zero Store calls. A target without scope capabilities is rejected before
  Load even when cached payload is available.
- Index identity changes during Load reject the old hit; changes during target I/O
  prevent storage under the old key. The live profile still requires host identity
  rotation and does not claim persistent publication isolation.
- Ordinary partial target failure preserves authorized partial output but is never
  cached and does not trigger a retry within the same request.
- Metadata mutations in source/returned values cannot change stored snapshots.
  Concurrent Load callers mutate independent maps under `-race`. Capacity, LRU
  eviction and exact expiry are covered with the fake clock.
- Structural keys partition tested core query/top-k/fetch/vector/threshold/planned
  text, host identity, index/recipe/configuration/capabilities, tenant/policy and
  pinned publication inventory. Complete graph/filter/plan partition fixtures and
  durable pinned-adapter integration remain outstanding.
- Core request snapshots own vector/threshold/graph/page/range pointers. BYOT
  nested ownership remains an explicit host port contract, not inferred cloning.

The root package suite and strict root lint passed after the cache load identity
race fix. All-module checkpoint results are recorded below after the last changes.
SCOPE-14 is verified for the reference fake-clock profile; SCOPE-09/13/16/17 remain
partial. No final audit or full completeness percentage is claimed.

Checks after the final cache target-admission spy and implementation changes:

- `GOCACHE=/tmp/ragy-implementation-go-cache make test` — exit 0, all modules,
  race and example checks; `/tmp/ragy-task12-test.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache make lint`
  — exit 0, all modules; `/tmp/ragy-task12-lint.log`.
- `git diff --check` — passed.

Example module resolution emitted the same non-fatal sandbox-denied module
metadata-cache write warnings. No live persistent/parser integration, quality
experiment or final independent audit was performed in this cache checkpoint.

## Strict composition admission checkpoint

`retrieval/admission_test.go` adds independently exercised public composition:

- `TestScopedCompositionPreflightRejectsBeforeAnyDispatch`: a supported sibling
  and unsupported leaf in parallel aggregate, fallback, rescue, conditional and
  route cases cause zero backend calls and zero planner/predicate callbacks. The
  test invokes the composition directly, without relying on a pipeline wrapper.
- `TestScopedPipelineRejectsOpaqueCustomNodeBeforePlanner`: an undeclared custom
  execution node is rejected before its Execute or the pipeline planner is called.
- `TestScopedNestedCompositionExecutesOnlyAllowedPayload`: actual BM25 scope
  enforcement within nested parallel aggregate/fallback returns only a-public;
  both primary targets run once and the admitted fallback remains unexecuted.
- `TestRouteRevocationBlocksRescueAndDecisionConsumers`: policy changes inside
  route planning prevent decision recording/target dispatch; changes inside target
  I/O suppress payload and prevent rescue predicates/secondary target calls.

All-module checks after the final route regression tests and implementation:

- `GOCACHE=/tmp/ragy-implementation-go-cache make test` — exit 0, all modules,
  race and example checks; `/tmp/ragy-task12-test.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache make lint`
  — exit 0, all modules; `/tmp/ragy-task12-lint.log`.
- `git diff --check` — passed.

The strict profile negotiates all configured reachable branches, including branches
not currently selected. Explicit partial admission and public external-module
conformance remain pending. These checks do not establish graph traversal isolation,
durable publication, live persistent/parser integrations or quality experiments.
The objective and both final audits remain outstanding.

## Explicit partial admission checkpoint

The public admission port returns immutable ReadCoverage. PartialReadNode makes
capability omission explicit in host configuration, retaining the original scope.
All public built-in execution wrappers and pipeline envelopes retain the report;
aggregate and branch/route merging preserve it. No compatibility admission API is
retained. Root still has no mandatory dependencies beyond the standard library.

Evidence in `retrieval/partial_read_test.go`:

- `TestPartialScopeSkipsUnsupportedBeforeIOAndPreservesNestedCoverage` runs real
  scoped BM25 retrieval in aggregate, nested optional composition and unused
  fallback. Only a-public survives, unsupported backend calls are zero, useful
  nested partial output is retained and static skip coverage survives the pipeline.
  Returned label slices are independent; serialized coverage has no target payload
  IDs, policy predicates or document counts.
- `TestPartialCannotSkipAuthorityDenialOrCancellation` proves that even a host
  authority error wrapping UnsupportedCapability is non-skippable. Cancellation
  similarly produces protected failure with zero target calls.
- `TestPartialDoesNotSkipJoinedAdmissionFailures` distinguishes pure capability
  mismatch from a join carrying a fatal sibling.
- `TestPartialNeverConvertsRuntimeFailureToCapabilitySkip` dispatches one admitted
  target that returns unsupported after retrieval; the partial wrapper propagates
  failure and suppresses protected payload instead of reporting a successful skip.
- `TestPartialRejectsTypedNilTargetsBeforeDispatch` rejects typed nil target/node
  interfaces without invoking their nil receivers or reporting a false skip.
- `TestCoverageWireRoundtripRejectsUnknownSchemaAndMalformedReports` reads the
  committed golden fixture, preserves its wire representation and rejects unknown
  schema/fields, missing required fields, inconsistent state and duplicate labels.
  Every failed decode retains the previously accepted coverage value.

Independent schema evidence:

- `PYTHONPATH=/tmp/ragy-schema-validator python3 docs/task12/verify_coverage_schema.py`
  — passed: 7 valid and 12 negative reports against the committed JSON Schema using
  an independent JSON Schema validator. The temporary host QA dependency is outside
  the repository and is not a core runtime dependency.

All-module gates after final typed-nil guard, golden roundtrip and lint fixes:

- `GOCACHE=/tmp/ragy-implementation-go-cache make test` — exit 0, all modules,
  race and examples; `/tmp/ragy-task12-test.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache make lint`
  — exit 0, all modules; `/tmp/ragy-task12-lint.log`.
- `git diff --check` — passed.

SCOPE-06/12/17 and ARCH-08 remain partial for full target-specific conformance,
immutable evidence export and remaining persisted envelopes. Public external-module
conformance, graph traversal/hydration, lifecycle, persistent integrations, parser,
recipes and quality experiments remain outstanding. Neither final auditor has
reviewed the complete implementation; no overall completeness percentage is claimed.

## Public external-module conformance checkpoint

The complete former internal contracttest implementation moved to the public
`github.com/skosovsky/ragy/contracttest` package. All shipped imports were updated;
the internal directory and compatibility import path were removed. The new scoped
suite is generic over host intent/request/source metadata and observes raw adapter
enforcement before relying on pipeline wrappers.

The independent consumer module uses `example.com/ragyconsumer`, outside ragy's
import namespace, and provides its own metadata types/schema field names. Its
reference adapter uses the shipped BM25 engine for candidate selection plus an
instrumented host payload materialization port. Evidence:

- `TestExternalScopedAdapterConformance` passes nine scenarios: allowed query with
  optional filters removed, contradiction, unsupported schema field, missing
  binding, revocation/expiry/cancellation before I/O, revocation during I/O and
  incoming deadline propagation.
- `TestExternalSuiteRejectsUndeclaredAdapterBeforeIO` proves zero Retrieve calls
  and zero payload I/O when capabilities are undeclared.
- `TestExternalSuiteRejectsPostfilterOnlyAdapter` rejects an adapter that loads
  forbidden payload then returns only allowed IDs. The spy observes before output
  filtering; final output cannot conceal that contract violation.
- `TestExternalSuiteDetectsMissingLeafGateDespiteEmptyDeniedOutput` rejects payload
  I/O before the direct adapter gate even though final denied output is empty.
- `TestExternalPartialAdmissionPreservesScope` executes public generic partial
  composition, delivers only a-public and retains a static skipped-branch report.
- `TestExternalSuitePreservesEarlierParentDeadline` ensures the suite itself cannot
  lengthen the host's existing deadline.

Verification commands on the final code/test state:

- From `examples/conformance`: `GOWORK=off GOCACHE=/tmp/ragy-implementation-go-cache go test -race -v ./...`
  — exit 0, independent of workspace resolution.
- `GOCACHE=/tmp/ragy-implementation-go-cache make test` — exit 0, all modules/race,
  including the new external consumer and all existing examples;
  `/tmp/ragy-task12-test.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache make lint`
  — exit 0, all modules including external consumer; `/tmp/ragy-task12-lint.log`.
- `git diff --check` — passed; no Go import references to the removed internal
  contracttest package remain.

SCOPE-15 is verified for the public generic suite and independent module. This is
fixture-specific contract evidence; it does not certify live persistent dense/tensor,
graph traversal, parser or remaining target integrations. Those requirements,
lifecycle/evidence/recipes/locators/quality experiments and both final independent
audits remain outstanding. The full objective is not complete.

## Scoped hydration, retained text locator and artifact mapping checkpoint

Implemented source.Reference, documents.Hydrator and ResolveText. Thin permission
metadata, exact publication/reference validation and complete batch admission precede
payload materialization. Every return has freshness gates. Missing/denied/deleted or
mismatched entries yield no payload; r1 is never replaced by existing r2. Host ports
must enforce exact representation retrieval and must not embed payloads in error
causes. Old Store is removed; its explicitly raw operations are named RawStore.

Implemented value-only tagged source.Locator and geometry validation/rotation,
canonical revision/location IDs, UTF-8 ByteSpan validation and private owned
MappedText snapshots. Original text, derived enrichment and unavailable/unobserved
precision remain distinct. Slice/Join recalculate byte coordinates or retain
support-only provenance. Explicit JSON schema identity/state round-trips the
snapshot and rejects unknown/invalid/trailing data without replacing existing state.
Wire validation is structural and does not claim to authenticate retained text.

Artifact rendering now requires context, the original binding and metadata cloning.
All callbacks and delivery are gated; failures return no artifact. Rune-budget
trimming recalculates exact mapping; dedup retains supports of every contributor.
Mapping cannot be combined with a string-only rewrite. Formatter receives independent
metadata/support/history slices so it cannot mutate the output snapshot/source.

Executable evidence:

- documents/hydration_test.go: retained r1/r2, deleted/denied/foreign/wrong descriptor
  before loading, revoked/cancelled I/O, mismatched publication, request capture,
  owned mutable metadata and all-or-nothing batches.
- documents/resolve_test.go: exact beta/Gamma spans in r1 while r2 exists, one load
  for repeated references, location dedup, deleted/denied/revoked/latest substitution,
  invalid UTF-8 boundaries with no partial result, unsupported kind before I/O.
- source/locator_test.go: ASCII/Cyrillic boundaries, unsupported/invalid geometry,
  clockwise 90-degree and inverse transforms, merged-cell identity, distinct
  spans/revisions/representations and canonical signed-zero geometry identity.
- source/mapping_test.go and mapping_json_test.go: beta->be [6,8), prefix enrichment
  without source offset shift, multi-source join, support ownership, absent precision,
  structural JSON round-trip/negative cases and failed-decode preservation.
- retrieval/artifact_mapping_test.go: dedup support retention, exact truncation,
  Unicode rune budget distinct from source bytes, revocation before/within every
  consumer stage, formatting ownership and rejection of ambiguous rewrite/mapping.

Final checkpoint gates after code fixes:

- GOCACHE=/tmp/ragy-implementation-go-cache go test -race ./source ./documents ./retrieval
  passed.
- GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache
  make lint — exit 0 across all modules, /tmp/ragy-task12-lint.log.
- GOCACHE=/tmp/ragy-implementation-go-cache make test — exit 0 across all modules,
  race and examples, /tmp/ragy-task12-test.log.

These confirm the checkpoint, not full acceptance. Geometry/cell/image resolution,
real parser fixtures and integration, persistent targets, source support propagation
through all chunking/grouping/rerank/storage paths, lifecycle/evidence/recipes and
comparative experiments remain partial or pending in requirements.md. No final
completeness percentage is claimed. Both final independent auditors remain required
after complete implementation; no release, publication or issue closure occurred.

## Synthetic PDF fixture preparation

Created adapters/pdf/testdata/manual.pdf and diagram.png with a deterministic
create_fixture.py using the bundled PDF dependencies. The PDF has physical pages
0/1, printed labels i/1, text Alpha beta. Gamma., a merged Revenue cell spanning
two columns, an embedded raster image at unrotated [100,200,300,400] points, and
a second image-only page rotated clockwise 90 degrees. Table is above the image
([60,110,260,180]) so there is no visual overlap. Table contents are original text;
no description/OCR capability is falsely claimed for raster pixels.

Ran verify_fixture.py against actual parser output: text, merged cell placeholders,
cell rectangles, page labels, rotation and native image rectangles all passed.
Both pages were rasterized with pdftoppm and visually inspected; the initial table
and image overlap was fixed and the latest pages checked. Fontconfig emitted
cache-directory permission messages but rasterization exited successfully.
The second native display image rectangle is [400,100,600,300], whose inverse
90-degree transform is the original [100,200,300,400].

This is preparatory real-asset/parser evidence only. The Go parser adapter,
normalized typed parser envelope, chunk/index/retrieve/resolve integration,
partial/OCR simulation and artifact lifecycle propagation have not yet been
implemented. LOC-07/08/11 remain pending; asset creation is not their acceptance.

## Actual optional PDF/layout adapter checkpoint

Implemented core layout envelope/validators and adapters/pdf as a separate module.
The adapter uses an embedded external parser script with an explicitly configured
Python interpreter and bounded input/output/page/element/process profile. It parses
already authorized bytes, snapshots input, validates normalized output and preserves
caller cancellation/earlier deadlines. No retry/background worker is started; engine
failure returns no partial source payload or exception text. Core dependencies
remain standard-library-only. Source authorization/materialization and truthful
revision/configuration identity remain host responsibilities.

Actual native output is normalized into retained page text, UTF-8 word spans,
unrotated point geometry, physical indices/printed labels, logical table cells and
original image regions. Differing CropBox, unsupported rotation and rotated table
grid fail explicitly. Missing OCR remains partial with ocr_unprocessed; page limits
retain actual PageCount and partial coverage. Core validators reject coverage
promotion, invalid spans/geometry/source identity and duplicate logical merged cell
IDs even if a duplicate uses a different artifact reference.

Executed actual external-parser tests with:
RAGY_PDF_PYTHON=/Users/skosovsky/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
GOCACHE=/tmp/ragy-implementation-go-cache go test -race -v ./adapters/pdf/...

Passed TestActualPDFLayoutParser, page-limit/invalid-PDF/word-limit/output-limit
cases, cancellation/earlier deadline, and actual native crop/rotation/rotated-table
rejection. TestActualParserProjectionIndexRetrieveAndScopedResolve uses the real
PDF parser, existing Recursive splitter and ProjectDocuments, actual BM25 indexing
and scoped retrieval, then exact retained text hydration/resolution and mapped
artifact rendering. It proves r1 remains beta while r2 exists, deletion/denial
cause no materialization/latest fallback, and host metadata retains partial source
coverage. The declared integration chunk profile is one complete normalized page;
it does not infer offsets for arbitrary string splitter transformations.

Final checkpoint gates:

- make lint with the configured Go/lint caches — exit 0, all modules including pdf;
  /tmp/ragy-task12-lint.log.
- make test with RAGY_PDF_PYTHON set to the actual bundled dependency interpreter
  and the configured Go cache — exit 0, all modules/race/examples;
  /tmp/ragy-task12-test.log. The log confirms actual parser tests PASS rather than
  SKIP. Without the explicit interpreter, integration tests intentionally skip;
  that state must never count as accepted parser capability.
- git diff --check — exit 0.

LOC-07/09/13 have evidence for the declared actual parser profile. LOC-08/10/11/12
remain partial where geometry/cell/image resolution, typed OCR observations,
managed lifecycle/manifest propagation and full source support propagation are
required. Persistent target integrations, recipes, graph/evidence/lifecycle and
comparative experiments are still outstanding. Both final independent auditors
remain required after full implementation. No release/publication/issue closure.

## Shared typed source materialization checkpoint

Moved scope/schema/reference/publication admission into source.Reader[TAccess,
TPayload]. Catalog is thin owned metadata; Loader returns exact-reference typed
Materialized payloads. Entire returned identity/count batch is admitted before any
payload callback. Validation, cloning, I/O and final delivery are gated; failures
return no payload. Clone and post-clone validation preserve owned output semantics.
No storage/retention/authorization engine, retry or worker was introduced.

Document hydration now specializes the reader with document ID/shape validation
and explicit metadata/history copying. Removed document-specific LookupRequest,
Descriptor/Catalog/PayloadLoader/Hydrated contracts, with no aliases. Existing
hydration/text resolution and actual PDF parser pipeline were migrated to source
ports and Payload access. The original permission implementation is not duplicated.

New source/read_test.go executes BYOT binary payload materialization. It proves
output bytes cannot mutate retained originals, one loading call without retry,
denied permission metadata causes zero loading, and wrong revision/mid-load
revocation cause zero payload validation/clone consumers. Existing document
hydration/resolve regression cases and actual PDF parser/chunk/index/retrieve/
resolve integration pass on the new shared implementation.

Final checkpoint gates:

- go test -race ./source ./documents — passed.
- go test -race ./adapters/pdf/... with actual RAGY_PDF_PYTHON — passed.
- make lint with configured caches — exit 0 across all modules;
  /tmp/ragy-task12-lint.log.
- make test with configured Go cache and actual RAGY_PDF_PYTHON — exit 0 across
  all modules/race/examples; /tmp/ragy-task12-test.log.
- git diff --check — exit 0.

Typed materialization does not itself implement geometry/cell/image resolution,
OCR observations or lifecycle. Those requirements and the other pending matrix
rows remain open; no full acceptance/completeness percentage or absence-of-defects
claim is made. Both final independent audits are still required after all scope is
implemented. No release/publication/issue closure performed.

## Retained document/page/cell/image resolver checkpoint

Implemented layout.Retained and layout.Resolver over the single source.Reader
admission path. It resolves exact original document media, normalized text spans,
whole pages, word evidence in page regions, logical cells and original image
regions. Requested geometry/cell identity is checked against retained canonical
location; image extent containment is explicit. OriginalRegion retains cell/image
extent. Bytes are original media, not an implicit crop or generated replacement.
Text mapping addresses original normalized page bytes; empty region evidence does
not claim a text mapping. Derived descriptions are separate support-only mappings
and must reference exactly the same canonical original artifact.

All-or-nothing resolver batches deduplicate location/reference loads and gate
freshness before materialization/projection/delivery. Returned bytes/diagnostics
are owned copies. Retained coverage/diagnostics are never promoted to complete.
Storage, UI, source authenticity/retention and model/description generation remain
host concerns; no blob service, hidden retry or worker was added.

Executed layout/resolve_test.go: mixed text/region/cell/image/document, merged cell
dedup, original/derived separation, output ownership, deleted/denied r1 while r2
exists, mid-load revocation, mismatched page/cell/outside image geometry and rejection
of original text substituted for an image description. Actual parser integration
TestActualParserRetainedPageCellImageDocumentResolution uses the real PDF report,
parsed Revenue cell and image region, original PDF/PNG fixture bytes supplied by
host retention, scoped source materialization and resolution. It verifies original
media, cell extent, description separation and partial coverage. This proves the
host retained source path; it does not claim that the parser extracts PNG bytes
from PDF or that cell/image documents have passed a complete indexing path.

Final checkpoint gates:

- go test -race ./layout and actual optional PDF module tests — passed.
- make lint with configured caches — exit 0, all modules;
  /tmp/ragy-task12-lint.log.
- make test with configured Go cache and actual RAGY_PDF_PYTHON — exit 0,
  all modules/race/examples; /tmp/ragy-task12-test.log.
- git diff --check — exit 0.

OCR observations, complete cell/image projection/index/retrieve integration,
source support propagation, managed lifecycle/persistent targets/evidence/recipes,
graph capabilities and comparative experiments remain pending. The matrix retains
partial status where those requirements are not covered. No full completeness or
absence-of-defects claim is made. Both final independent audits remain required;
no release/publication/issue closure performed.

## Typed OCR and combined modality retrieval/resolution checkpoint

Implemented typed OCR observations (unobserved/recognized/unreadable/unsupported),
exact image identity/fingerprint validation and an owned layout transformation.
Recognized OCR is derived support-only text; partial coverage is preserved rather
than promoted. Optional image text callbacks are freshness-gated. Layout projection
emits exact page mappings, original logical cell support and available derived image
text; pixel-only/unreadable images do not acquire invented searchable text.

TestActualParserOCRSimulationAndModalityIndexRender now runs the actual external
parser, separately labeled unreadable OCR simulation, layout projection, real BM25
indexing/retrieval, artifact rendering and scoped resolution of the artifact's
returned supports. Revenue is one logical merged cell; fixture diagram is a derived
image description. The resolver returns original normalized text, cell text/extent
or PNG bytes plus a separate description. For every retrieved source, r2 is retained
while r1 is deleted and then denied: both operations return unavailable with zero
payload materializations, without substituting latest or exposing a partial batch.
The source retention port in this test is host memory, not persistent storage.
Page text's separate actual pipeline includes existing chunking; this modality test
does not establish source propagation through arbitrary splitters/grouping/storage.

Focused checks: actual optional PDF module go test -race passed; module lint passed
with 0 issues. Global make lint/test results are recorded below after completion.
Managed lifecycle, persistent targets, graph/recipes/evidence and comparative
experiments remain mandatory unfinished scope. Final auditors are not launched yet.

Final checkpoint gates for the combined path:

- make lint with configured caches — exit 0 across all modules;
  /tmp/ragy-task12-lint.log.
- make test with configured Go cache and actual RAGY_PDF_PYTHON — exit 0 across
  all modules/race/examples; /tmp/ragy-task12-test.log. Actual parser/modality test
  executed successfully, rather than being skipped.
- git diff --check — exit 0.

These gates cover the current implementation only; they do not prove the full
specification complete. No release/publication/issue closure performed.

## Durable lifecycle manifest/store checkpoint

Implemented typed source fingerprints, manifests, targets, exact artifacts/supports,
confirmed failure checkpoints and namespace publication inventory. Validators reject
foreign namespace/revision/access, duplicate targets/artifacts/supports, missing
fingerprints, unknown schema/state and unready default required publication.
Explicit partial manifests retain missing target state; tombstone publication can
remain acknowledged while physical cleanup is pending. No target executor or read
barrier integration is claimed by these data invariants alone.

Optional lifecycle/filestore provides durable local-filesystem generation CAS.
Each writer takes nonblocking flock, validates existing state, writes/syncs a private
temporary file, atomically renames and syncs the directory. Corruption is protocol
failure, not absence; stale/contending writers fail explicitly. Returned data owns
nested inventory slices. Cancellation after rename can be an uncertain committed
outcome and must be reconciled by Load.

Executed race tests for round trips/negative manifests, independent store-object
CAS concurrency (one winner), stale writer, mutation isolation, corrupt inventory,
canceled CAS and restart in an actual fresh process. The subprocess loaded durable
publication, advanced the cleanup checkpoint, exited, and the parent loaded both.
This is real filesystem persistence; it does not replace required dense/tensor
backend integrations or lifecycle transition fault injection.

Independent wire validation:
PYTHONPATH=/tmp/ragy-schema-validator python3 docs/task12/verify_lifecycle_schema.py
passed 2 positive and 12 negative fixtures. Go also decodes that fixture and checks
semantic cross-field source/publication/support identities. JSON Schema validates
wire shape; semantic reference integrity remains the executable Go contract.

Global make lint/test passed before the final fixture-only test addition. Final
focused lifecycle race/lint and diff checks are recorded below. Full target executor,
bootstrap/cleanup/reference accounting and remaining matrix scope stay unfinished.

Final focused lifecycle go test -race ./lifecycle/... passed; focused lint reports
0 issues; git diff --check passed. No full acceptance percentage is claimed.
No release/publication/issue closure performed; final independent audits remain
required after all mandatory implementation and integration/experiments.

## Explicit lifecycle executor prepare/stage/reconcile/publication checkpoint

Implemented Executor with typed payload capture/validation and registered Stage/Inspect
ports. Prepare persists the planned artifact/support inventory and validates expected
active source publication. Replay of the complete unchanged operation preserves
progress without writing again; namespace key reuse with changed plan conflicts.
Stage records unknown with confirmed staging checkpoint before one target dispatch.
A lost response retains unknown; another Stage does not retry it. Reconcile invokes
Inspect once and advances only the exact planned ready revision. Published manifests
reject late target mutations. Default readiness and explicit partial/tombstone
manifest rules remain enforced by validation and publication CAS.

Publish atomically changes manifest state and active source pointer using durable
namespace generation CAS plus expected active source check. A lost CAS response
returns ErrOutcomeUnknown while preserving the context cause. Retrying observes the
already committed manifest and does not republish or increment generation.

Race tests use actual filestore and injected target fault ports, clearly distinct
from real persistent target adapter integration. Tests cover unavailable required
tensor after dense staging, timeout after target commit, no blind stage retry,
inspect recovery, immutable old snapshot/owned target request, unchanged-plan replay,
changed-payload conflict, payload validation even on ready target, competing source
writers and unknown-but-committed publication response. These do not claim target
read visibility/scoped snapshot conformance, cleanup/recovery deadlines/bootstrap,
actual backend fault injection or complete lifecycle acceptance.

Checkpoint gates: make test passed across all modules/race/examples with actual
RAGY_PDF_PYTHON; make lint passed across all modules after a formatting-only test
helper fix; git diff --check passed. Logs: /tmp/ragy-task12-test.log and
/tmp/ragy-task12-lint.log. Full mandatory target/lifecycle/recipe/evidence/experiment
scope remains incomplete; final auditors are deferred until that scope is implemented.
No release/publication/issue closure performed.

## Durable cleanup coordinator checkpoint

Publication now records acknowledgement time using injected ExecutorConfig.Now.
Cleaner captures the complete retired publication ancestry plus known abandoned
staging plans, excluding newer plans. Durable jobs contain per-target pending,
unknown/complete, attempts, next due time, deadline and overdue. Validation rejects
future/current inventory retirement, omitted targets and false complete reports.
Cleanup requests carry exact retired inventories and other managed supports; already
cleaned inventory is excluded from retained support hints. Ports remain responsible
for actual backend reference accounting, idempotence and concurrent write fencing.
Prepare rejects reuse of occupied exact target/artifact identity; Stage rejects a
source whose expected publication has already changed.

Actual durable filestore plus explicit fault ports verify one-attempt dispatch,
1/2/4/4 second capped backoff, no calls before due time, 60-second deadline measured
from tombstone acknowledgement, overdue persisted across Cleaner restart and explicit
recovery to complete. Active publication remains the tombstone throughout. A committed
cleanup with lost response persists unknown; restart requires InspectCleanup and
performs no second deletion. Known abandoned plans enter the queue; newer work is
excluded and cannot be forged as retired inventory. This coordinator evidence is not
actual persistent target cleanup or shared graph deletion acceptance.

Lifecycle wire schema/fixtures now include publication timestamps and cleanup jobs.
Independent validation passed 3 positive/13 negative fixtures, including malformed
cleanup timestamps; Go validates semantic relationships and missing-inventory failure.
The filesystem adapter and its dependent integration tests are explicitly Linux/macOS;
portable core contracts retain no OS-specific dependency.

Checkpoint gates: final make lint and make test (all modules/race/examples with
actual RAGY_PDF_PYTHON) exited 0; git diff --check passed. Logs remain
/tmp/ragy-task12-lint.log and /tmp/ragy-task12-test.log. Full target/read/bootstrap/
graph/recipe/evidence/experiment scope remains unfinished; no complete-goal or
absence-of-defects claim. No release/publication/issue closure performed.

## Actual managed lexical lifecycle/scoped snapshot checkpoint

Implemented lexical/managed using actual BM25, exact revision-bound in-process record
snapshots, durable lifecycle manifests and typed Stage/Inspect/Cleanup ports. Capture
fixes one namespace publication before fan-out, excludes tombstones, validates every
requested target ready and uses a stable logical hash independent of staging generation.
Pinned empty inventory is explicit complete-empty. Partial missing requested targets
are rejected strictly; broader partial capture is still unfinished acceptance.

RequirePinnedPublication enables fail-closed admission of pinned-only targets. Raw
BM25 remains live/scoped without managed snapshot claims. Readonly BM25Snapshot owns
metadata, gates mandatory scope before cloning, freezes the original binding fingerprint
and exposes no Index/Upsert API. Managed scoring keeps the original binding rather
than downshifting to unrestricted access. Canonical artifact IDs prevent source-local
ID collisions. Exact cleanup shares the staging lock and fences changed publication.

Actual BM25 race tests prove r2 staging invisibility, r1/r2 publication selection,
old pinned reads, tenant metadata admission before payload cloning, source-local ID
collision handling, owned mutable metadata, revocation during projection, tombstone
barrier before physical cleanup, unrelated faq preservation and unavailable old snapshot
without partial sibling leakage/latest substitution. Fresh volatile adapter state is
explicitly unavailable, not falsely durable. Separate readonly snapshot test rejects
binding replacement and confirms denied payload is not cloned.

These are actual lexical lifecycle tests with a durable filestore, not fake lexical
retrieval. The adapter itself is in-process memory and does not claim durable lexical
restart. Dense/tensor persistent integrations, graph target/reference accounting,
bootstrap, full provenance, explicit partial/joint profiles, recipes/evidence/quality
experiments and final audits remain required unfinished scope.

Final checkpoint gates: make lint and make test exited 0 across all modules,
race and examples with actual RAGY_PDF_PYTHON; git diff --check passed. Logs:
/tmp/ragy-task12-lint.log and /tmp/ragy-task12-test.log. No full completeness or
absence-of-defects claim. Final independent audits remain after complete scope.
No release/publication/issue closure performed.

## Durable inventory bootstrap coordinator checkpoint

Implemented typed delta/complete inventory, full/partial coverage, namespace/watermark,
required target inventory, opaque unmanaged keys and a fenced host verifier port.
Inputs are deeply captured before port calls. Confirmation must match namespace,
watermark, coverage and the exact envelope fingerprint. Complete+partial and malformed
namespace/source/target/identity cases fail before verification. Import preserves all
old manifests and absent source pointers, checks expected publications and generation
CAS, and calls no destructive target port.

Complete coverage records removal proposals against captured expected publications;
only an explicit later tombstone can acknowledge deletion. Durable receipts preserve
these proposals on replay rather than recomputing against newer sources. Different
inventory under the same kind/watermark conflicts; replay performs no new verification
or write. Unmanaged keys never become guessed managed manifests or cleanup jobs.
Existing published manifest adoption compares frozen target states and full inventory,
not just content fingerprint or display identity.

Executed coordinator race tests with actual filestore and an explicit verifier fixture:
delta keeps faq and unknown legacy key; caller/port/output mutation does not alias
receipt; idempotent replay and changed envelope conflict; complete missing faq proposal
remains faq1 after faq2 publication; incomplete coverage fails before verifier and
unconfirmed digest never changes durable state. This is coordinator evidence, not
an actual persistent backend fenced inventory verifier or legacy reindex acceptance.

Wire schema includes inventory receipts. Independent validation passed 4 positive and
13 negative fixtures. Real target inventory/backfill, dense/tensor persistence, graph,
remaining scope/provenance/recipes/evidence and comparative experiments remain required.

Checkpoint gates: make lint and make test exited 0 across all modules/race/examples
with actual RAGY_PDF_PYTHON; git diff --check passed. Logs remain
/tmp/ragy-task12-lint.log and /tmp/ragy-task12-test.log. Final audits are deferred
until complete mandatory implementation/integration/experiments. No full completeness
or absence-of-defects claim; no release/publication/issue closure performed.


## Persistent tensor target: staging, scoped query and exact cleanup

`tensor/persistent` now executes actual local-filesystem storage and query on the
local filesystem profile (Darwin/Linux, flock, atomic rename and directory fsync).
This is not an injected fake target. Records are captured into bounded catalog and
checksum-addressed payload files, written into a private staging directory, and
installed by a durable directory rename. Stage checks the registered durable
inventory and expected source publication before installation. Catalog identity,
artifact inventory, embedding space, shapes, normalization and byte limits are
validated; Inspect verifies payload checksums before reporting ready.

Query accepts an explicit bounded candidate set, requires pinned publication, and
selects only confirmed published manifest revisions. Mandatory AND query/planned
metadata predicates run against the thin catalog before tensor/content loading or
host metadata Decode/CloneMeta. The result is a normal ResultSet with native MaxSim
scores and model/config/space-specific score semantics. QueryResult additionally
owns candidate IDs, budget and candidate-local ranks; QueryCapabilities explicitly
states that this is not exhaustive index search. No unrelated payload is read to
score missing candidates.

Cleanup requires a registered durable cleanup item and the expected active source
publication. It renames only the retired manifest directory into a private retired
path, fsyncs, removes it and fsyncs again. Inspect distinguishes pending physical
retirement from complete deletion. Tombstone visibility is independent of cleanup.

Evidence (race-enabled actual filesystem integration):

- TestPersistentTensorStageSurvivesFreshAdapterAndDetectsCorruption: fresh instance,
  input ownership, checksum failure; no in-memory target reuse.
- TestPersistentQuerySeparateProcessRestart: an actual separate executable process
  reads the parent's durable manifests/catalog/payloads without restaging.
- TestPersistentQueryNativeOracleAndNegativeCandidates: native 2/1/-1, then supplied
  t2/t3 excludes t1 without a global ranking claim; budget 100, TopK 10.
- TestPersistentQueryFiltersBeforePrivatePayloadRead: private t3 payload deliberately
  corrupt, successful tenant-a read proves it is not loaded; contradictory planned
  tenant-b filter produces empty intersection.
- TestPersistentTensorInvalidMatrixFailsBeforeTargetWrites and
  TestPersistentQueryRejectsInvalidMatrixBeforeFilesystemIO: invalid normalized
  matrix creates no target state/lock respectively.
- TestPersistentTombstoneCleanupRetainedReadUnavailable: published tensors disappear
  from new reads before cleanup, remain on old pinned reads until cleanup, then old
  reads fail closed after a fresh adapter opens the physical storage.

This checkpoint does not complete RAG-003. Public tensor conformance, additional concurrency/cancellation/crash
faults, backend inventory verifier, storage JSON Schema, additional staging-crash fault profiles,
joint lifecycle profiles and comparative quality/latency experiment remain pending.
The direct candidate fixture is an oracle/integration check, not an experiment.

Checkpoint commands passed:

- `GOCACHE=/tmp/ragy-implementation-go-cache go test -race ./...`
  (root module, including tensor/persistent; log `/tmp/ragy-task12-root-test.log`).
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache golangci-lint run ./tensor/... ./internal/durablefs/...`
  (changed tensor/filesystem packages; log `/tmp/ragy-tensor-lint.log`).

These commands do not certify the other workspace modules or replace the final
`make lint`/`make test` gates and independent audits required by the full task.


## Portable sparse candidates and interrupted staging

The query contract now lives in `tensor/query`, independently of the selected
filesystem adapter. Search composes one admitted retrieval backend and one typed
bounded scoring port. Both leaves are admitted before candidate dispatch. The
original binding, scope and planned filters survive projection; candidate requests
own separate options/token matrices. Candidate overflow fails rather than silently
truncating; no hidden retry or cross-scale numeric fusion is performed.

`TestActualSparseCandidatesToPersistentTensor` executes actual readonly BM25
candidate retrieval and actual persistent tensor query, with budget 100 and TopK
10. All-candidate and missing-t1 cases demonstrate candidate-local ranking and
visible lost candidate recall. This is still an integration fixture, not a quality
experiment or a persistent dense/hybrid baseline.

`tensor/query` unit fault tests additionally prove: unsupported target rejects
before candidate dispatch; faulty backend overflow does not reach tensor scoring;
candidate request mutation cannot change the captured scoring embedding; cancellation
inside metadata projection prevents scoring; malformed embedding fails before
candidate backend I/O. The complete malformed matrix/space table now runs through
both actual persistent Stage and Query, checking no target state/lock is created.

Persistent staging now uses a deterministic reserved operation path. Retry validates
the exact registered plan and expected publication before recreating its interrupted
path. Cleanup/InspectCleanup include it. The tombstone/cleanup integration fixture
leaves a partial reserved staging directory and an opaque unknown directory: cleanup
removes only the registered operation's path and preserves the unknown inventory.
The interrupted-directory fixture is explicitly a simulated crash residue; the
separate-process restart read test is a real process boundary. A killed staging
process/fault-at-every-fsync suite remains pending and is not claimed by this check.


Latest portable-composition checkpoint:

- Root `go test -race ./...` passed after adding portable contracts, actual sparse
  composition and malformed-profile table; `/tmp/ragy-task12-root-test.log`.
- Focused `go test -race ./tensor/query ./tensor/persistent` additionally passed after
  zero-value/typed-nil fail-closed guards; `/tmp/ragy-tensor-test.log`.
- Focused lint covers all tensor packages and internal/durablefs. These are
  incremental checks, not the full final multi-module gates or independent audits.

Remaining tensor acceptance includes public external conformance, persistent dense
baseline, comparative quality/latency report, full fsync/crash/CAS/cancellation fault
matrix, storage JSON schemas, inventory verifier and joint lifecycle profiles.


## Persistent dense and actual joint profiles

`dense/persistent` implements actual on-disk normalized vectors, thin metadata
catalog, byte/checksum/identity validation, registered Stage/Inspect, pinned exact
query, admitted scan limits and exact Cleanup/InspectCleanup. The declared profile
is local filesystem persistence, not ANN search or another backend's guarantees.

Race-enabled dense integration proves native 1/0/-1 ranking on a fresh adapter and
an actual separate executable process; checksum failure; input ownership; private
corrupt payload exclusion before read; malformed vector/model profiles before
writes/query lock; total admitted scan limit before payload/metadata callbacks; and
tombstone visibility followed by physical cleanup/unavailable retained revision.
See dense/persistent storage_unix_test.go and query_unix_test.go.

`lifecycle/integration` runs both dense+lexical and dense+tensor over actual targets
and durable filestore manifests, with namespace fixture-a, policy@r1:p1/p2,
faq@r1:f1 and policy@r2:p3:

- TestActualJointPublicationUnknownStageAndRetainedSnapshot: dense is physically
  ready; the secondary really commits then loses its response. Required-unknown
  prevents publication, ordinary reads remain r1 in both leaves, actual Inspect
  confirms without another Stage, publication selects r2 in both leaves, old pinned
  reads still select r1 and unrelated faq remains.
- TestActualJointStaleSourceWriterCannotPublish: r2/r3 both stage against r1; r3
  publishes, stale r2 source CAS fails; both leaves return r3 and unchanged faq.
- TestActualJointTombstoneAndCleanupPreserveFAQ: tombstone closes policy in both
  new reads before physical deletion, old reads retain r1 until cleanup, both exact
  retired target inventories are removed, faq continues to retrieve, old dense
  snapshot becomes unavailable without partial/latest substitution. Cleaner uses
  injected clock and 60-second deadline with 1/2/4-second backoff profile.

These tests add real target integration to prior lifecycle state-machine fault
suites; they do not replace the still-pending dense+graph/shared-support path,
partial joint publication, backend bootstrap verifier, full crash/fsync/CAS fault
matrix, public conformance, comparative experiments and final independent audits.


Dense/joint checkpoint commands passed:

- Root `GOCACHE=/tmp/ragy-implementation-go-cache go test -race ./...`, including
  the final wire fingerprint projection and all joint cases; log
  `/tmp/ragy-task12-root-test.log`.
- Focused `golangci-lint run ./dense/... ./lifecycle/integration/...` with the
  implementation lint cache; zero issues, log `/tmp/ragy-joint-lint.log`.

The full multi-module make gates and two final audits remain required after the
rest of task12 is implemented. No release/publication/issue closure was performed.


## Actual managed graph storage, traversal and support release

The graph target now executes actual owned in-process Stage/Inspect, ledger-confirmed
pinned Traverse/FindByIDs and registered Cleanup/InspectCleanup. This is an explicit
volatile profile over durable filestore manifests, not a fake-only graph port or a
claim of persistence after restart.

Race integration evidence in graph/managed/managed_test.go:

- Shared-support fixture: policy@r1:p1 and faq@r1:f1 support one e1; union contains
  both references. Deleting policy preserves e1 with faq/f1 only; deleting the last
  managed source removes it. A cleaned captured snapshot becomes unavailable.
- Private bridge fixture: svc -> secret -> db, with forbidden secret. Neither
  Traverse nor FindByIDs clones the bridge/edges; unreachable db payload is not
  loaded during traversal. Only svc payload callback runs.
- Conflicting canonical nodes produce a conflict with both allowed source refs and
  no automatic winner. Revocation inside a payload clone suppresses the entire
  snapshot and support output.
- Staged but unpublished graph records remain unreadable even through a manually
  constructed matching pinned target. A fresh empty volatile adapter cannot serve
  a durable published revision and fails closed.
- Revision swap selects r2/p3 supports; a retained captured read keeps r1/p1.
- Cycles stop with visited sets; node/edge budget overflow rejects before payload
  callbacks and delivers no partial result.

Root `go test -race ./...` passed on this checkpoint, with log
`/tmp/ragy-task12-root-test.log`. Focused graph lint passed with zero issues,
`/tmp/ragy-graph-lint.log`. These checks do not complete graph acceptance:
at that checkpoint host-owned basis and dense+graph joint profile were pending;
the subsequent checkpoint below implements them. Public conformance, complete
provenance integration, ontology/extractor/model adapter, summary recipes and
experiments remain unfinished.

## Explicit host graph basis and all three actual joint target profiles

`TestManagedCleanupPreservesExplicitHostBasis` verifies equal managed/host graph
supports, source cleanup preserving the explicitly selected foundation, complete-empty
managed inventory, no fabricated source refs, no implicit basis selection, and
unavailable after explicit release. `TestHostBasisIsOwnedImmutableAndScoped` verifies
owned labels, changed immutable-ID conflict, and no private payload callbacks.

`TestGraphBackendRankOnlyAndRevokedProjection` verifies absent score/ranks and
revocation during custom projection suppressing the entire retrieval output.
`TestGraphBackendConflictRequiresExplicitPolicy` verifies default projection rejects
conflicting canonical facts instead of choosing a winner.

`lifecycle/integration/joint_unix_test.go` executes dense+lexical, dense+tensor and
dense+graph against actual target implementations, with durable ledger and persistent
dense/tensor files. Each profile verifies unknown stage after an actual commit,
Inspect recovery without another Stage, unchanged ordinary reads before publication,
retained captured reads, stale source writer CAS, tombstone before physical cleanup,
and exact cleanup preserving FAQ. This does not yet prove every fault stage or the
explicit partial publication profile. The graph joint read helper projects actual
FindByIDs output; standalone graph Backend is verified separately.

## Bounded durable lifecycle snapshots and lexical publication confirmation

`TestSnapshotBudgetRejectsOversizedWriteWithoutReplacingPublication` verifies exact
budget-boundary acceptance, oversized update rejection preserving generation/payload,
and oversized durable read rejection. `TestSnapshotBudgetInvalidBeforeFilesystemIO`
verifies nonpositive/overflow budgets are rejected without creating the root.

`TestManuallyPinnedStagedRevisionCannotReadBeforePublication` verifies manually pinned
lexical staged records remain unavailable with zero payload clone calls, and the same
exact inventory becomes readable only after ledger-confirmed publication. Existing
retained read and all three joint target suites remain passing.

Commands and results at this checkpoint:

- `GOCACHE=/tmp/ragy-implementation-go-cache go test -race ./...`: exit 0,
  `/tmp/ragy-task12-root-test.log` (before the equivalent inventory-helper extraction).
- Focused storage/lifecycle race suite: exit 0, `/tmp/ragy-task12-storage-test.log`.
- Final lexical plus all joint profiles race suite after helper extraction: exit 0,
  `/tmp/ragy-task12-lexical-publish-test.log`.
- Final focused lifecycle, graph, lexical, dense and tensor lint: exit 0, zero issues,
  `/tmp/ragy-task12-storage-lint.log`.

Full module-wide make gates, all remaining acceptance experiments and the two final
independent audits are still mandatory. No release, publication or issue closure
was performed.

## Atomic budget foundation

`recipe/budget` race tests verify four simultaneous model reservations at cost 30
under cap 100 admit exactly three; input/output/call exhaustion; cancellation and
fake-clock deadline rejection before admission; required unknown-price rejection
and explicit advisory unknown-cost diagnostic; unknown usage retaining the full
reservation; known usage refunds without refunding calls; actual overrun failure;
copied leases settling concurrently at most once; and integer-overflow rejection.

Final package race check passed (`/tmp/ragy-task12-budget-test.log`); focused lint
passed with zero issues (`/tmp/ragy-task12-budget-lint.log`). The three recipe
implementations, host pricing ports, dispatch integration, stage evidence and actual
adapter experiments are still pending. No recipe-completeness claim follows from
the ledger's standalone tests.

## Three bounded text recipes and actual scoped lexical integration

`TestReferenceRecipeCases` covers helpful rewrite, worsening rewrite retained
original, no-answer, two-variant dedup preserving both query contributions, complete
decomposition and missing-part partial decomposition. All model calls and native
source observations remain captured; strategies use the existing RRF merger.

Additional race fixtures cover:

- zero model budget rejecting dispatch while retaining typed partial original;
- required unknown pricing with zero model calls and advisory unknown pricing with
  conservative token reservations and explicit unknown-cost/usage diagnostics;
- revocation during retrieval/planner/assessor suppressing every payload/side output;
- parent cancellation before dispatch and propagation of an earlier parent deadline;
- injected attempt-clock deadline during planning, no further retrieval/assessment
  or metadata callback, final assembly from previously captured identities;
- detached intent/request/meta/support snapshots under mutating planner/assessor,
  independently owned query/selection/backend metadata;
- malformed/duplicate/excessive plans, invalid selection, observed usage overrun,
  foreign pinned source support, oversized backend result;
- rejecting precomputed original vectors before any text-variant backend work.

`TestThreeRecipesUseActualScopedBM25` runs every strategy through real scoped BM25
and existing typed request projection over the specified four-document corpus plus
a private tenant document. The private document cannot reach assessor or selected
source evidence. Rewrite retrieves d1, multi-query d2 and decomposition d1+d4.
Planning/assessment are scripted in this integration fixture and are not presented
as actual-model quality results.

Final root `go test -race ./...` passed at this checkpoint, with
`/tmp/ragy-task12-root-test.log`; focused recipe lint passed with zero issues,
`/tmp/ragy-task12-recipe-lint.log`. The module-wide make gates, real model adapter,
comparative experiments, immutable recording/export and two final independent audits
remain mandatory. The text recipe profile uses model-free backend/admission/pricing
ports; it does not yet provide a budget-aware text-to-vector encoder integration.
No release, publication or issue closure was performed.

## Immutable evidence and recording foundation

The evidence package implements private owned canonical records, strict transactional
wire decoding and explicit privacy policies. Race tests mutate original hits/meta,
source slices, decoded snapshots and returned JSON buffers without altering records.
Default policy omits query/text/auth; ID policy denies hit rows; allowlisted snippets
and grade 2 exact-source/query/rubric labels work. Required missing/unsupported/private
fields return distinct errors, including unsupported score capability with no hits.

Scope tests enforce mandatory metadata before support/ID callbacks, exact published
source identity and separate support admission. Revocation during policy/sink suppresses
all result/receipt delivery. Best-effort cannot rescue protection failure. Spy recording
modes prove one retrieval call, disabled skipping capture/sink, separate best-effort
failure receipt and required failure retaining retrieval result/record with an error.
Raw sink error text is not added to the record or stable RecordingError message.

The actual canonical fixture and executable schema are in fixtures/evidence_record.json
and schemas/evidence.schema.json. Independent `verify_evidence_schema.py` passed 8
positive and 16 negative cases. Go additionally rejects duplicate/missing/unknown
fields and incompatible schema, and preserves the old record after failed decoding.
Schema covers wire shape; source admission and label binding are runtime contracts.

Root race suite passed (`/tmp/ragy-task12-root-test.log`) before the final equivalent
validator helper extraction; final focused evidence race tests passed
(`/tmp/ragy-task12-evidence-test.log`). Focused lint passed with zero issues
(`/tmp/ragy-task12-evidence-lint.log`). Full make gates and two final independent audits
are still required. Automatic execution/recipe stage adapters, full cross-capability
recording acceptance and experiments remain pending. No release or issue closure.

## Automatic recipe evidence recording increment

Added optional `recipe/recording.Run` and owned `recipe.SnapshotResult`. The
adapter exports actual executed query stages with native scores and selected RRF
fusion separately, preserves publication/source revisions, and never retries a
recipe to repair recording. Required sink failure retains completed result and
immutable record while failing recording; best-effort records a failed receipt.
Protection failures suppress all outputs. Required unsupported judgments and
scoped typed-nil codecs fail before model/retrieval dispatch.

Executable actual-BM25 integration covers disabled/best-effort/required modes,
exactly two retrieval and two scripted model calls despite sink failure, source
revision r1/publication pub1 association, native versus fusion score states,
private-tenant exclusion, default query/snippet/metadata/error omission, record
ownership, source/sink revocation, and a baseline-only model-budget stop with
explicit `not_run` model stages. Model ports are scripted contract fixtures;
these tests do not constitute the required real-model quality experiment.
Additional regressions cover final-gate envelope suppression, dispatched but
unobserved retrieval stage identities, and unavailable accounting beyond exact
JSON integer range or with unknown usage.

Verification on this increment:

- `GOCACHE=/tmp/ragy-implementation-go-cache go test -race ./recipe/... ./evidence/...` — exit 0; `/tmp/ragy-task12-recording-test.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache GOLANGCI_LINT_CACHE=/tmp/ragy-implementation-lint-cache golangci-lint run ./recipe/... ./evidence/...` — exit 0, zero issues; `/tmp/ragy-task12-recording-lint.log`.
- `GOCACHE=/tmp/ragy-implementation-go-cache go test -race ./...` — exit 0 for the root module; `/tmp/ragy-task12-recording-root-test.log`. No additional parser-runtime environment was supplied for this checkpoint; prior explicit parser integration evidence remains separate.
- `git diff --check` — passed.

Remaining evidence work includes automatic baseline execution recording, full
locator/contributor wire association, and retained failed-attempt journals.
The authoritative requirement matrix remains partial for those contracts and
for real model adapters/experiments. Final independent audits and full acceptance
remain pending. No release or issue closure was performed.

## Document provenance composition and persistent mapping increment

Document now carries immutable content mapping and independently owned original
source supports. Default grouping joins mappings only with complete coverage and
retains every known support; missing mapping cannot be attributed to the winning
metadata. Custom grouping retains original supports and validates returned mapping.
Max-score dedup and RRF retain loser supports while leaving exact coordinates on
the winning content. Renderer automatically uses/slices document mapping; string
snippet transformations retain supports and explicitly lose precise mapping.
Cache/result snapshots, lexical capture, graph projection copying, hydration and
recipe/candidate copying own supports. Recipe source admission must confirm every
attached support, and all query/fusion/assessment snapshots remain independent.

Persistent dense/tensor payloads serialize original mapping under the payload
checksum. Their staging/read checks reject content mismatch and foreign source,
namespace, revision or access identity, allowing the original transformation and
representation to differ from indexed content. Actual storage restart tests confirm
original byte coordinates and index document-level support survive. These tests
restart adapter instances against real files; existing separate-process tests
continue to cover storage process loss. New tests do not claim a new independent
process specifically for the mapping regression.

Verification:

- Focused race tests across retrieval, recipes, dense/tensor persistent, lexical,
  managed graph and documents passed; `/tmp/ragy-task12-provenance-test.log`.
- `make test` with explicit PDF parser runtime passed all modules/race/examples;
  `/tmp/ragy-task12-provenance-all-test.log`. Original-mapping restart and foreign
  revision rejection execute for both persistent targets.
- `make lint` passed all modules with zero issues;
  `/tmp/ragy-task12-provenance-all-lint.log`.
- Final added recipe source-confirmation/assessor/fusion ownership regressions
  passed with race; `/tmp/ragy-task12-provenance-recipe-test.log`.
- Managed lexical's final source identity validation passed with race;
  `/tmp/ragy-task12-provenance-lexical-test.log`. The subsequent full test also
  includes that validation.

Graph source projection, automatic baseline recording/full locator wire evidence,
failed journals, real model adapters/experiments and the remaining matrix rows
remain outstanding. No final completeness percentage or final audits are claimed.

## Managed graph fact-to-document provenance increment

Replaced the document-only managed graph projection callback with typed projections
that declare contributing node/edge facts separately from document identity. The
backend captures admitted original references before calling host code and attaches
those references automatically. It rejects unknown facts, absent fact associations,
foreign source refs/mapping supports and mandatory-scope-incompatible projected
metadata before post-projection metadata cloning. Projection output count is bounded
by declared graph budgets; combined budgets reject integer overflow. Default node
projection remains rank-only and does not fabricate citations for host-only bases.
The backend cannot prove that a host summary is factually correct; declared evidence
and source authenticity via retained resolver remain separate guarantees.

Race regressions against actual managed traversal and durable lifecycle cover shared
policy/FAQ source supports, mutation of the projector's view without mutation of the
captured inventory, unknown/private fact identities, foreign revision citations,
missing fact claims, private projected metadata, host basis without invented source
refs, and reprojected FAQ-only support after policy tombstone/cleanup. Existing
conflict-policy and revoking projector cases continue to pass.

Checks:

- Focused managed graph/lifecycle integration race tests passed before the final
  additional cleanup case; `/tmp/ragy-task12-graph-projection-test.log` was then
  refreshed by the final managed graph race run including cleanup.
- Root `go test -race ./...` passed; `/tmp/ragy-task12-graph-projection-root-test.log`.
- Focused managed graph lint passed with zero issues before the additional cleanup
  test; final focused lint is recorded below when completed.

The full goal still requires graph extraction/resolution/recipes and experiments,
automatic complete evidence recording, remaining lifecycle/conformance acceptance,
and both independent final audits. No release or issue closure was performed.

Final graph projection checkpoint: managed graph race run including the cleanup
regression exited 0; focused managed graph lint exited 0 with zero issues;
`git diff --check` exited 0. These incremental checks do not establish completion
of the remaining requirement matrix or substitute for final independent audits.

## BYOT graph identity resolver increment

Added `graphingest/resolution` with typed entity/relation kinds and attributes,
explicit namespace/alias decisions, schema/ontology validators, relation keys,
source admission, finite batch/support limits and independent metadata snapshots.
Admission validates the complete structural batch and authorizes every original
support before identity callbacks. Canonical namespace/key IDs remain independent
of separately captured ontology/policy identity. Equivalent assertions union their
sources; conflicting attributes remain multiple variants without selecting a winner.
Ambiguous entities and dependent relations remain explicit unresolved records.
Even ambiguous endpoint relations cannot bypass ontology validation.

The reference fixture verifies production Billing/Pay merge by explicit policy,
independent staging Billing, absent namespace ambiguity, conflicting Team A/Team B
attributes with all source supports, and two-source depends_on support union.
Negative cases cover bounds, malformed locators, denied admission, cancellation
before work and during identity resolution with no partial envelope returned.
Attributes remain independent of input storage; policy configuration changes are
recorded without inventing new entity IDs for unchanged canonical decisions.
This is a resolver over typed source-supported fixture input, not a model/source
extractor or completed canonical graph materialization/history implementation.

Checks:

- `go test -race ./graphingest/...` passed; `/tmp/ragy-task12-resolution-test.log`.
- `golangci-lint run ./graphingest/...` passed, zero issues;
  `/tmp/ragy-task12-resolution-lint.log`.
- Root `go test -race ./...` passed on final code;
  `/tmp/ragy-task12-resolution-root-test.log`.
- `git diff --check` passed.

GRAPH-01/03/05 remain partial pending actual extraction, materialization and durable
decision history/recompute integration. Graph recipes, real adapters/experiments,
remaining matrix requirements and the two final independent audits remain required.
No release or issue closure was performed.

## Resolver-to-managed-graph materialization increment

Added explicit `graphingest/materialization` Build, returning an owned managed graph
payload and planned lifecycle manifest without storage side effects. Source tuple
selects variants/supports; complete structural/support admission precedes projection.
Host projects BYOT kinds/attributes into schema-validated canonical graph payloads.
Transformation fingerprint includes extraction/ontology/alias policy identities.
Same-source conflicting variants, own-source unresolved assertions and unsupported
relation endpoint closure fail before projection. Metadata/labels/source inventory
are independently owned. Cancellation/access gates suppress partial plans.

Actual integration runs resolver alias grouping, materialization, durable filestore
manifest execution and managed graph traversal for s1/s2. Shared canonical Billing,
LedgerDB and depends_on IDs combine original chunk supports. Tombstone/physical
cleanup of s1 removes source-only owned_by/Team A while depends_on and its s2 chunk
remain. Removing s2 removes the last derived facts. No support is synthesized from
the canonical graph document ID. Policy identity changes bind a new transformation
while unchanged canonical IDs remain stable; output metadata mutation cannot change
resolver attributes or another plan.

Verification:

- Initial focused `go test -race ./graphingest/...` passed;
  `/tmp/ragy-task12-materialization-test.log`.
- Final root `go test -race ./...` passed, including the added policy/ownership
  regression; `/tmp/ragy-task12-materialization-root-test.log`.
- Final `golangci-lint run ./graphingest/...` passed with zero issues;
  `/tmp/ragy-task12-materialization-lint.log`.
- `git diff --check` passed.

Full typed durable decision-history storage/recompute remains pending; retained
manifests and a transformation digest are not claimed to be a complete decision log.
Actual source/model extraction, graph recipes/experiments and all other pending
matrix requirements still prevent final acceptance. Final independent auditors,
release and issue closure have not been performed.

## Bounded injected model extraction adapter increment

Added optional `graphingest/extraction` with typed BYOT access/domain attributes,
mandatory metadata/schema admission, retained source/quote admission and immutable
source mappings. Model input contains ordinal/text pairs plus ontology/config and
reserved token limits; no binding, access metadata, source ref objects or credentials
are supplied. Output can name only local mentions and input snippet indices.
Original locators/revisions and namespace are derived from admitted inputs. Mixed
namespace evidence stays ambiguous; model output cannot choose canonical identity.
Ontology/endpoint/evidence bounds validate before delivery; attribute snapshots remain
independent of client output and validation callbacks.

One model call is reserved against a shared atomic ledger after host pricing/exact
input-token counting and before dispatch. Usage settles even on failure. Required
unknown price rejects before dispatch; advisory unknown accounting retains full
reservations. Known token overrun is rejected even if cost is unknown. Token counter
and model input slices are independent. Parent/attempt deadline, injected clock and
revocation suppress all returned payloads; there is no model retry. Client contract
requires enforcing supplied token limits; this adapter does not supply a tokenizer
or constrain arbitrary injected Go code as a sandbox.

Regression evidence includes private metadata rejection before model, source
admission revocation, input-byte/token/unknown-price refusal, model error with no retry,
known usage settlement/overrun, foreign snippet index, missing relation endpoint,
unsupported ontology kind, source revision/namespace preservation, client/output
ownership, counter input mutation isolation, mixed namespace ambiguity, post-model
revocation/deadline and advisory-cost token overrun. All model responses in this
checkpoint are scripted client contract fixtures, not external-model quality proof.

Verification:

- Final `go test -race ./graphingest/...` passed;
  `/tmp/ragy-task12-extraction-test.log`.
- Root `go test -race ./...` passed before the equivalent final test variable rename;
  `/tmp/ragy-task12-extraction-root-test.log`.
- Final `golangci-lint run ./graphingest/...` passed, zero issues;
  `/tmp/ragy-task12-extraction-lint.log`.
- `git diff --check` passed.

Actual provider transport, strict provider/schema integration and real model
experiments remain required. GRAPH-02 stays partial. No independent final audits,
release or issue closure were performed.

## Structured HTTP model transport increment

Added optional `adapters/openai/structured` generic typed client and extraction
binding. Protocol matches the official
[structured output guide](https://developers.openai.com/api/docs/guides/structured-outputs):
strict JSON schema response format, explicit completion token cap and refusal/length
handling. Host selects model/schema/instructions/tokenizer and supplies executable
schema validation and host-defined actual price conversion. Baseline/core gains no
provider dependency. One bounded POST, no adapter retry, disabled redirects, positive
attempt duration, bounded response, and no raw payload/credential diagnostics.

Tests execute actual HTTP against local regression servers and verify request shape,
schema snapshot ownership, counter mutation isolation, exact metadata integer
9007199254740993, usage preservation on invalid output/refusal/truncation, missing or
inconsistent usage, duplicate JSON members, unknown/missing fields, trailing data,
token overrun, pre-dispatch input/byte/cancellation admission, response bounds, HTTP
failure, redirects, and cancellation during domain validation. Constructor failures
cover schema syntax/shape/size, schema name, missing callbacks, duration, endpoint
and header-injection credentials. Typed extraction integration verifies actual price
conversion. Core extraction through the HTTP client verifies denied tenant causes
zero dispatch/reservations, admitted source remains the original locator/revision,
private source/access identity is absent from model input, and actual usage settles
the shared ledger after exactly one call.

The final core/HTTP regression also verifies truncated provider output is suppressed
while its actual usage/cost settles the ledger and the single model call remains
spent; no retry or refund is introduced on this failure path.

Final module checks passed:

- `go test -race ./...` in the optional provider module;
  `/tmp/ragy-task12-structured-module-test.log`.
- `golangci-lint run ./...` in the same module, zero issues;
  `/tmp/ragy-task12-structured-module-lint.log`.

HTTP responses and token counts here are deterministic fixtures. These checks prove
transport/core integration contracts, not live model quality or provider tokenizer
accuracy. No provider key or model configuration was present when inspected; live
configuration has been requested. GRAPH-02 remains partial. Other partial/pending
matrix requirements remain mandatory. Final two independent audits, release and
issue closure have not been performed.

## Durable typed resolution history increment

Resolver now retains one source-bound decision trace for each entity/relation
mention, including explicit alias identity, ambiguous outcome, canonical identity,
resolved endpoints and relation key. Trace supports are independently owned and
retain the exact original locator/revision. Group conflict variants still choose
no winner. Resolution wire contracts use explicit field names for typed JSON history.

Optional `graphingest/resolution/history` captures complete typed input/result,
ontology/policy/extraction identities, host run and predecessor digest. Snapshot
bytes are private; reference/record exports are independently owned. Admission
checks every original locator and pinned source/publication membership before
serialization or payload reads; byte/fact/support limits and trace associations
are checked. Result supports must come from captured input. JSON decoding uses
exact numbers and rejects unknown fields/trailing data. This persistence profile
requires faithfully JSON-serializable BYOT data and bounded/pure custom JSON methods.

The filesystem implementation synchronizes staged payload, atomically links an
immutable content-addressed record without overwrite and synchronizes its directory.
Only its own temporary path is cleaned. The host durably prepares root/ancestors
and retains snapshot references in its catalog. Reference supports participate in
the storage filename; a forged allowed support inventory cannot address an existing
private payload by content ID alone. Fresh reads reauthorize retained source evidence
before payload I/O and verify checksum plus exact decoded support inventory.
Predecessor metadata is host-declared lineage, not an ordered CAS/latest log or
proof of graph publication. Uncertain append errors require explicit inspection.

Tests use the actual resolver and filesystem to verify alias grouping, ambiguous
entity/relation decisions, conflict owners with both source refs, revision/policy
recomputation retaining old records after storage reopen, stable canonical identity,
source revision history, input/output ownership, concurrent idempotent append,
corruption refusal, denial before filesystem I/O, forged reference isolation,
missing traces/foreign supports/bounds and pre-canceled capture. Storage reopen is
a fresh instance reading filesystem records; this checkpoint does not claim a
combined live extraction/history/publication experiment or full decision-domain
attestation beyond host policy contracts.

Verification:

- `go test -race ./graphingest/...` passed;
  `/tmp/ragy-task12-history-test.log`.
- Root `go test -race ./...` passed;
  `/tmp/ragy-task12-history-root-test.log`.
- `golangci-lint run ./graphingest/...` passed, zero issues;
  `/tmp/ragy-task12-history-lint.log`.
- Full `make lint` passed for root, adapters and examples;
  `/tmp/ragy-task12-history-all-lint.log`.
- Full `make test` with configured real PDF runtime passed after correcting a
  prohibited production README phrase;
  `/tmp/ragy-task12-history-all-test.log`.

GRAPH-03/04/05 retain partial status until combined actual source/model extraction,
history and publication acceptance. Other pending matrix requirements remain
mandatory. Final independent audits, release and issue closure were not performed.

## Model-free bounded local expansion increment

Added optional `recipe/graphexpand` using the actual managed scoped traversal
profile. Host provides canonical seeds, direction/filters, finite depth/node/edge
limits, duration/clock, metadata ownership and fixed per-dispatched-attempt price.
Public `managed.Adapter.AdmitTraversal` validates scope/pinned-publication and the
complete traversal before pricing/reservation/target I/O. One BFS dispatch uses the
target visited set; no model call, planner, answer generation, fallback or retry is
introduced. The shared ledger reserves retrieval/cost before dispatch, settles flat
attempt cost even on failure, and rejects model token reservations in graph quotes.

Tests execute the actual managed traversal, verifying Team A→Service→Database via
depth 2 undirected expansion, cycles bounded by visited set, inaccessible private
bridge/edge excluded before payload callbacks, one graph call and zero model calls.
Explicit host foundations preserve their own immutable support identities and do
not create original source citations. Budget exhaustion/unknown required price
causes insufficient with zero target I/O; invalid depth fails before pricing.
Cancellation, pricing/snapshot revocation, node/edge limits and fake-clock attempt
expiry suppress all evidence without retry. The published-source integration
physically stages/publishes graph facts through the durable lifecycle executor,
captures a pinned publication, and verifies unique original source/revision support
export distinct from graph fact IDs.

Successful complete expansion describes the declared traversal, not natural-language
answer sufficiency; no-edge output is insufficient. Metadata/label/support/conflict
slices are independently owned. Comparative hybrid quality/cost/latency experiment
and full recipe/evidence acceptance remain mandatory, so GRAPH-07 remains partial.

Final checks passed:

- `go test -race ./recipe/graphexpand`;
  `/tmp/ragy-task12-graphexpand-test.log`.
- `golangci-lint run ./recipe/graphexpand ./graph/managed`, zero issues;
  `/tmp/ragy-task12-graphexpand-lint.log`.
- Full `make lint` for root/adapters/examples;
  `/tmp/ragy-task12-graphexpand-all-lint.log`.
- Full `make test` with configured real PDF runtime;
  `/tmp/ragy-task12-graphexpand-all-test.log`.
- `git diff --check` passed.

Community/global summary recipes, live model experiments and other pending matrix
requirements are not replaced by this local expansion increment. Final independent
audits, release and issue closure were not performed.

## Community/global summary increment

Added optional `recipe/graphsummary` with explicit host community membership,
BYOT access metadata/schema, retained-source admission, prices/tokenizer and typed
one-dispatch model client. Community uses one map; global uses exactly two maps and
one non-recursive reduce, at most 20 snippets per community / 40 total. Shared ledger
reserves calls/input/output/cost before dispatch; known actual usage settles even
on failure, unknown accounting retains reservations, and known token overruns fail
even with advisory unknown price. Counter/client input slices are independent.

Complete batch shape/revision/metadata admission precedes source/model callbacks.
Source checks follow pricing/counting, occur around stages and before dispatch,
reduce and delivery. Models receive ordinal/text/question/stage only, without scope,
source refs, canonical membership or credentials. Output text/ordinal bounds,
duplicates/foreign indices, declared member coverage and both-community global
selection are enforced. Missing member coverage is explicit insufficient; bounded
budget stops retain only actual, revalidated community artifacts, with no fabricated
global output or retry. Private immutable summaries bind authorization/predicate/
publication and require fresh original-source Resolve before exposing support-only
derived text. Semantic correctness of prose is an external evaluation responsibility.

Added optional structured HTTP `Summarizer` exposing matching model/counting ports;
host supplies schema/validator/tokenizer/model/instructions and actual price units.
Its actual HTTP fixtures verify map/reduce request binding and known usage on
incomplete output without retries. Real provider credentials/quality are not claimed.

Core regressions cover C1/C2 supports, exactly one/three model calls, ownership,
model privacy, zero/one/two-call partial stops, whole-batch private/foreign/member/
byte/snippet refusal before callbacks, unsupported selections, omitted global
community, text/usage errors, source deletion/revocation during models or pricing/
counting/partial stops, fake deadline, cached artifact deletion/revocation/TTL and
publication invalidation. Actual scoped source Reader integration admits original
representations before hydration/model input and denies private final descriptors
before any payload load; catalog deletion invalidates summary without loading payload.
The original reader has an explicitly pinned original transformation target alongside
graph inventory. Its transformation check was preserved; graph-index identity was
not substituted for original identity.

Final checks passed:

- `go test -race ./recipe/graphsummary` and focused lint, zero issues;
  `/tmp/ragy-task12-graphsummary-test.log`, `/tmp/ragy-task12-graphsummary-lint.log`.
- Optional provider module `go test -race ./...` and lint, zero issues;
  `/tmp/ragy-task12-graphsummary-http-test.log`, `/tmp/ragy-task12-graphsummary-http-lint.log`.
- Full `make lint`;
  `/tmp/ragy-task12-graphsummary-all-lint.log`.
- Full `make test` with configured real PDF runtime;
  `/tmp/ragy-task12-graphsummary-all-test.log`.
- `git diff --check` passed.

Prose, token counts and model responses here are deterministic protocol fixtures.
Complete actual graph-membership/source/model integration, recipe/evidence export and
comparative hybrid quality/cost/latency experiment remain mandatory. GRAPH-06/08/09
remain partial. Final independent audits, release and issue closure were not performed.

### Retrieval HTTP planner/assessor and exact model reservations

Implemented the clear-break Planner/Assessor `recipe.ModelLimits` argument using
actual per-dispatch reserved Quote token maxima. Zero input/output model maxima fail
before model dispatch; retrieval remains token-free. Added regressions for different
planner/assessor reservations and known token overruns with advisory unknown pricing.
The unknown-price fixture now declares real token maxima, preserving its intended
pre-dispatch price-unavailable assertion.

Optional structured HTTP Planner/Assessor bindings now implement all three recipe
strategies. Typed wire projection excludes domain intent/request/document metadata,
document IDs, source supports, filters and auth/publication identities. Query/text,
strategy cardinality, aggregate evidence byte/document bounds, duplicate/foreign
selections and actual reserved token caps are enforced. A fresh scope admission
runs after the host token counter and immediately before transport; revocation in
that callback prevents HTTP dispatch. Known failed-call usage is retained without
redispatch. Host price failure keeps accounting unknown and reservations conservative.

Actual HTTP regression servers verify each strategy, exact completion cap, privacy,
invalid output usage, no retry, low token cap before I/O, missing binding, foreign
selection and scope revocation inside counting. Responses/tokenization remain
protocol fixtures. Combined real retrieval+HTTP model integration and live comparative
quality acceptance are still required; these fixtures do not claim that acceptance.

Checks passed on this checkpoint:

- Root `go test -race ./recipe/...` and focused lint (zero issues).
- Optional provider module `go test -race ./...` and lint (zero issues).
- Full `make lint` and full `make test` with actual PDF Python runtime.
- `git diff --check`.

Logs are retained in `docs/task12/results/recipe-model-*.txt`. Requirements
RECIPE-01/02/03/08/10 remain partial pending complete combined/experimental acceptance.
No final audits, release or issue closure were performed.

### Combined scoped BM25 → HTTP models → recipe → evidence recording

Added external-consumer integration tests in the optional provider module. Each of
the three strategies executes the actual scoped BM25 snapshot over the fixed
four-document corpus with a fifth private-tenant document, actual HTTP transport
for planner/assessor, per-dispatch token/cost reservations, and core RRF/support
selection. HTTP responses and token counts are deterministic protocol fixtures.

Verified d1 for rewrite, d2 for both multi-query variants with two retained dedup
contributors, and d1+d4 for decomposition. Original source revision r1 survives
selection. Private payload, domain intent/request/document metadata and IDs do not
reach the model. Accounting records exactly two model dispatches, 80 input tokens
and 60 fixture cost units. Host revocation during planner response suppresses all
already retrieved original evidence and prevents subsequent retrieval/assessment.

The combined single-rewrite path also runs through disabled/best-effort/required
recording with an actual failing spy sink. No recording policy redispatches models:
disabled never writes; best-effort retains completed result and failure receipt;
required returns ErrRecordingFailed while retaining the completed retrieval fact.
The immutable record associates retrieve/0, plan, retrieve/1, assess and fusion;
raw query/content, private metadata and sink diagnostics remain absent. Mutating
returned evidence cannot alter the exported record bytes.

This proves combined transport/retrieval/recording mechanics, not live model
quality or exact tokenizer accuracy. The comparative experiment and complete
failed-attempt journal/locator export remain mandatory and unverified.

Checks for this integration checkpoint passed: optional module `go test -race ./...`,
optional module lint (zero issues), and `git diff --check`. Logs:
`docs/task12/results/recipe-combined-test.txt` and
`docs/task12/results/recipe-combined-lint.txt`. Production implementation did not
change in this checkpoint; its preceding full make lint/test evidence is retained.
Final independent audits and release/issue closure were not performed.

### Failed-attempt journals and RRF overflow regression

Added explicit Recipe.RunObserved for recording prior captured queries, actual
stages and the settled ledger on ordinary attempt errors. Failed envelopes declare
Failure/StageFailure and never contain selected hits. Recipe.Run retains its
error-payload suppression contract. Protection failure and parent cancellation
suppress the full journal, including previously retrieved observations.

Enabled recording now uses RunObserved, validates publication identity and exports
actual prior retrieval/model observations on failure. Failed retrieval without
validated retained documents is missing_observation, never observed-empty. Result
Fusion distinguishes unstarted, dispatched-without-retained-result and completed
fusion; unknown values are rejected by the recorder. Pre-attempt failures without a
journal still produce explicit missing observation with unavailable diagnostics.

Regressions cover failing planner/assessor with prior hits and actual usage, no
selected evidence, immutable record/metadata ownership, missing retrieval observation,
unstarted/missing fusion classification, revocation and parent cancellation. Actual
scoped BM25 + HTTP planner/assessor + required recording also verifies a foreign
assessor selection returns the original protocol error, settles both model calls
once and records prior native retrieval stages. No retry or successful fusion is
fabricated. Error strings remain absent from export.

During the fusion audit, reproduced a separate accepted-input defect: RRF k=MaxInt
made integer k+rank+1 overflow, silently producing zero-score outputs. The failing
TestRRFLargePositiveKDoesNotOverflowRankDenominator confirmed this before the fix.
RRF now converts operands before addition. The regression passes with positive
normalized scores and stable ordering; floating-point ties at enormous k remain a
numerical precision limit, not integer overflow. DEFECT-RRF-01 was added to the matrix.

Checks passed:

- Race tests for retrieval and all recipe packages; focused lint, zero issues.
- Optional provider module race tests and lint, zero issues.
- Full make lint and make test with the configured actual PDF runtime.
- git diff --check.

Logs: docs/task12/results/failed-journal-*.txt. Full locator/contributor wire export,
remaining cross-capability/fault acceptance and comparative experiments remain
mandatory. This checkpoint does not establish final completeness. Independent final
audits, release and issue closure were not performed.

### Original locator wire export

Added value-only original Location wire records for document, UTF-8 span, page,
region, merged table cell and image region, with explicit locations_state. Hits
accept explicit original locators and typed document mappings/supports automatically.
Every location is validated and must belong to the already admitted hit source
inventory before export policy. Foreign association cannot reach the location callback.

Location export is explicit opt-in: AllowLocation permits geometry and page/table/
element labels, while numeric permission and all source identifiers must also be
allowed. Default/denied/partly redacted source identity omits the complete location;
unknown mapping remains unavailable. No coordinates are inferred. Access fingerprints
and arbitrary metadata are excluded from wire. A callback revocation suppresses the
record. Wire decoding validates tagged-union geometry and source association, not
source authenticity/retention. Immutable records retain owned serialized locations.

Recipe recording passes query original supports and the union of selected dedup
contributors. An actual scoped BM25 multi-query integration verifies retained d2@r1
location after RRF dedup and both typed query contributors. Per-query contributor wire
association is still pending; a union location is not claimed to replace it.

Go regressions cover all six kinds, exact span/page/cell values, roundtrip/ownership,
default/numeric/identifier/location denial, foreign source before callback, revoked
policy and invalid geometry/source association during decode. The independent evidence
schema now requires locations fields and validates the union shape: 14 positive and
25 negative cases passed. Cross-field geometry/source relationships are additionally
validated by Go; JSON schema does not authenticate producer observations.

Checks passed: focused race tests and lint, independent schema validation, full make
lint and make test with actual PDF runtime, git diff --check. Logs retained under
`docs/task12/results/locator-export-*.txt`. Migration docs describe the strict shape
break and re-export from retained observations without fabricated geometry.

Final independent audits, release and issue closure were not performed. Comparative
experiments, contributor wire associations and remaining full-scope acceptance are
still required.

### Query-to-contributor immutable wire association

Added explicit Hit.Contributions and immutable wire query/document/list-position/
location associations. Native query hits receive their executed ordinal; fused hits
retain each original dedup contributor. Recording verifies selected tuples against
captured query documents and exact supports before export. Association rank is the
one-based position in the captured result list; separately observed native rank/score
is preserved and never reconstructed from this position.

Export requires explicit AllowContribution, permitted numeric diagnostics and document
identifiers. Locations independently require the location policy. Callback input owns
its support slice; mutations cannot change producer observations. Freshness follows
callbacks, and revocation suppresses the whole record. Raw queries, domain metadata
and auth fingerprints are excluded. Unknown association remains unavailable; denied
association is omitted without exposing hidden counts.

Validation rejects invalid ordinals/ranks, duplicate query/document/rank tuples and
contributor locations outside the hit's admitted original union. Generic producer
ordinals remain producer attestations; decoding does not authenticate execution.
Recipe recording additionally checks actual captured query/document/support tuples.
Actual scoped BM25 multi-query dedup verifies both query ordinals and retained original
r1 locations in wire, alongside both typed contributors.

Regressions cover precise association, roundtrip/ownership, callback mutation isolation,
default/numeric/document/association redaction, malformed inputs before callback,
revocation, invalid/duplicate wire tuples and foreign geometry, and recording's
query/document/rank/support substitutions. Independent schema validation passed
16 positive and 32 negative cases. The strict new contribution fields require explicit
consumer migration; unavailable associations must not be invented from current index.

Focused race tests/lint, full make lint, full make test with actual PDF runtime and
git diff --check passed. Logs: docs/task12/results/contributor-export-*.txt.

This completes the tested recipe contributor wire path, not final full-task acceptance.
Comparative experiments and remaining cross-capability/fault/conformance acceptance
remain mandatory. Final independent audits, release and issue closure were not performed.

### Actual persistent dense / tensor reference comparison

The external consumer in examples/conformance/tensor_comparison publishes real
filesystem dense and tensor payloads through separate durable manifests, reconstructs
fresh adapters and performs scoped pinned reads. The saved synthetic corpus contains
three documents and one query with explicit normalized float32 vectors, token matrices
and graded qrels. Dense TopK=10 is timed separately from dense candidates=100 followed
by bounded MaxSim TopK=10; five paired raw timing observations are retained.

The observed dense ranking starts with t2; MaxSim ranks t1/t2/t3 with native scores
2/1/-1. Baseline Recall@10 is 1 and nDCG@10 is 0.7098097413968654. Tensor Recall@10 is
1 and nDCG@10 is 1, yielding a measured gain of 0.29019025860313463. Candidate recall
is 1. Removing t1 from retrieved candidates yields Recall and candidate recall 0.5,
with nDCG@10 0.1310456303875653. Missing qrels are rejected, and absent judgments do
not become zero relevance in metric checks.

Embedding payload sizes are 24 dense bytes and 32 tensor bytes. Actual logical index
file sizes count target catalog/payload files and exclude separate manifest storage.
Runtime model calls and token usage are zero because embeddings are supplied by the
saved fixture. This is an actual persistent reference experiment, not live embedding
quality evidence. The single query and tiny corpus cannot establish meaningful p50/p95:
both remain null, the latency gate is unavailable-small-fixture and recommend_default
is false. The quality gate alone does not justify changing default behavior.

Report: docs/task12/results/tensor-comparison.json. Reproduction instructions and
measurement boundaries are in examples/conformance/tensor_comparison/README.md.
Race regressions verify actual rankings, analytical qrels, candidate loss, raw timings,
sizes, unavailable percentiles and withheld recommendation. Targeted consumer race
tests/lint and full make lint / make test with the actual PDF runtime passed; logs
are retained as docs/task12/results/tensor-comparison-*.txt.

This verifies the specified synthetic tensor reference profile. Live model experiments,
remaining lifecycle/cross-capability/conformance acceptance and both final independent
audits remain required. No release or issue closure was performed.

### Persistent recovery exact artifact inventory

DEFECT-INSPECT-01 was reproduced against actual durable dense and tensor adapters:
a fresh adapter returned ready for a manifest with an omitted artifact, substituted
artifact ID, missing named target or duplicate reference. The saved pre-fix regression
log records all eight false acknowledgements (four per adapter).

Inspect now validates the non-tombstone manifest and named target before filesystem
I/O, compares the exact catalog/request artifact-reference set before ready, verifies
payloads and checks cancellation before confirming the result. The same set in another
order remains valid. An additional actual filesystem corruption test removes one
catalog descriptor while retaining payload files; a fresh adapter refuses readiness.
This fixes completeness attestation, not fenced bootstrap namespace coverage or source
support authenticity. Those separate acceptance requirements remain open.

Evidence: dense/persistent/inspect_inventory_unix_test.go and
 tensor/persistent/inspect_inventory_unix_test.go; reproduction and post-fix logs in
results/inspect-inventory-*.txt. The original TЗ and requirement matrix now include
the newly confirmed defect. Consumer recovery instructions preserve full inventory
and stop publication on incompleteness; no new storage format is introduced.

Focused dense/tensor/lifecycle race tests, full make lint and full make test (including
actual PDF parser runtime and examples) passed after the final code change. The full
logs are retained above; git diff --check is clean. This checkpoint does not establish
final-task completeness. Remaining acceptance and both final independent audits are
still required; no release or issue closure was performed.

### Actual joint partial publication and cancellation after commit

TestActualJointExplicitPartialPublicationFreezesMissingTarget runs all three actual
joint profiles (dense+lexical/tensor/graph), each with pending and lost-response
secondary scenarios. Partial manifests keep dense r2 and the secondary's pending or
unknown checkpoint without an invented revision. Staging preserves r1's active logical
publication. After explicit partial publication, strict joint capture refuses access;
explicit dense-only scoped reads select r2. Retained joint reads still select r1,
including unchanged faq. Replay preserves durable partial state and repeats no stage.

TestActualJointCancellationAfterPublicationNeverRollsBack cancels immediately after
real filestore publication CAS in all three joint profiles. The returned error retains
context.Canceled and ErrOutcomeUnknown with the Published checkpoint. Fresh actual
reads select r2 in both targets, retained authorized reads select r1, and a fresh
context reconciles publication without a second CAS or repeated staging.

These are actual backend acceptance cases. Partial fan-out capture and automatic
publication coverage association with envelope/evidence are still pending; passing
these tests does not establish full LIFE-08 or whole-task completeness. No release,
issue closure or final independent audits were performed.

Focused lifecycle race tests, full make lint and full make test (actual PDF runtime
and examples included) passed. Logs: results/partial-joint-*.txt. git diff --check
is clean. LIFE-15 now has actual supported-profile evidence; LIFE-08 remains partial
until the execution/export integration above is implemented and verified.

### Explicit partial capture, fan-out admission and immutable export

Implemented CapturePartialPublication over one durable namespace snapshot. A missing
ready checkpoint for any active source excludes the entire configured target branch.
Available branches retain exact source/revision/transform/access tuples; no older
revision is substituted. Immutable excluded target labels partition binding fingerprints
and are checked before adapter I/O. Unknown custom admission is rejected explicitly.
Shipped dense/lexical/tensor adapters, graph backend/traversal and request projection
honor exclusions. Owned readonly BM25 retains its original full binding fingerprint.

Inspection/execution envelopes and recipe admission retain pin coverage. Evidence
capture cannot promote a partial pin to complete, including empty/redacted records;
failed/insufficient outcomes retain their original cause. Original complete-empty
validation still rejects actual hits. The existing coverage wire schema is reused,
without adding raw source/policy diagnostics.

Actual joint tests now run pending and lost-response partial profiles through parallel
composition: dense r2 plus faq is read, missing secondary is never invoked, and actual
hit export remains partial despite complete producer coverage. Reverse profiles prove
available lexical, persistent tensor and managed graph remain readable when dense is
excluded. Default strict capture still refuses all incomplete joint profiles. Real
empty namespaces stay complete-empty; all-excluded requested targets stay unavailable
partial. Ownership, exclusion/cache identity, invalid labels and contradictory inventory
regressions cover the immutable constructor. Available source revisions remain subject
to existing identifier allowlist and source admission; no hidden source counts are
introduced in exclusions.

LIFE-08 now has evidence for the explicit whole-target exclusion profile. Other
lifecycle fault/bootstrap requirements, comparative experiments and final independent
audits remain mandatory. No release or issue closure was performed.

Focused race tests, full make lint and full make test (actual PDF runtime and examples)
passed. Independent evidence schema validation passed 16 positive and 32 negative
fixtures. Logs are retained under results/partial-capture-*.txt; git diff --check
is clean. This verifies the explicit branch-exclusion increment, not whole-task
completion; remaining mandatory acceptance and final audits are unchanged.

### Verified ingestion reuse and actual ACL-only replacement

Executor.CheckReuse compares the full desired source identity with the active
non-tombstone publication and requires the complete ready target profile. It inspects
each retained target inventory once, checks cancellation, then reloads the durable
namespace to reject any intervening generation change. No Stage, publication write,
implicit retry or model call occurs. Confirmation is a point-in-time observation,
not a lease or permission to bypass later expected-publication CAS.

Root tests verify exact fingerprints/profile, changed content/transformation/revision/
access refusal without target I/O, missing/uncertain/foreign-revision observations,
cancellation and a concurrent durable generation change. Actual joint dense+lexical,
dense+tensor and dense+graph profiles confirm reuse without write/stage repetition.
They then stage/publish ACL-only replacements while retaining identical content and
source revision; all target reference access fingerprints change and fresh inspection
confirms only the new active identity. Host IAM revocation is not inferred from that
fingerprint update.

Fresh volatile lexical/graph adapters retain the durable ready manifest but lose their
physical inventory. CheckReuse returns incomplete, proving that matching ledger hashes
cannot authorize skipping a necessary rebuild. Persistent dense/tensor inspection
continues to verify catalog inventory and payload checksums through existing adapters.

Focused lifecycle race tests, full make lint and full make test (actual PDF runtime and
examples included) passed. Logs retained under results/reuse-*.txt. git diff --check
is clean. LIFE-16 now has implementation and actual supported-profile evidence.
Remaining bootstrap/fault/cross-capability and experiment acceptance plus both final
independent audits are still required. No release or issue closure was performed.

### Actual volatile inspection and registered original-support integrity

DEFECT-INSPECT-02 was reproduced on actual lexical and graph target data. Before the
fix both returned ready for missing/substituted/duplicate artifact inventory or absent
named target; lexical also acknowledged another payload fingerprint. The retained
pre-fix log records nine false acknowledgements.

Lexical now retains an owned full staging manifest. Both volatile inspectors validate
manifest/target, compare full identity/payload and exact artifact/support sets, and
check cancellation before readiness. Graph staging also captures an owned manifest.
A shared exact set comparison preserves order independence while rejecting missing,
duplicate or substituted supports; Manifest.Clone owns nested target/artifact/support
slices. Graph read/cleanup inventory checks use the same full-support comparison.

All four supplied targets now compare original supports against the durable registered
stage plan. Actual integration tests register unknown dispatch and try a forged support
with unchanged artifact refs/payload: Stage refuses and Inspect confirms no target data
was installed. Additional actual volatile tests cover content fingerprint, payload,
support substitution and unchanged inventory permutation. These checks prove checkpoint
agreement, not original source authenticity or complete namespace bootstrap coverage.

Focused lifecycle/target race suites and full make lint / make test (actual PDF runtime
and examples included) passed. Logs: results/volatile-inspect-*.txt. git diff --check
is clean. The TЗ and matrix include the newly confirmed defect. Remaining mandatory
bootstrap/fault/cross-capability/experiment acceptance and both final independent audits
remain required; no release or issue closure was performed.

## Persistent original supports retained and validated

Dense/tensor catalog storage now contains owned artifact and original-support inventory,
not only payload descriptors. Artifact.Validate and SameArtifactInventory centralize
semantic validation/comparison; descriptors must cover exactly the retained artifacts.
Inspect compares complete support inventory, and read/cleanup use the same durable
ownership checks. The incompatible previous catalog shape is rejected rather than
backfilled from guessed source IDs. Migration requires trusted source reindex into a
new owned root and explicit publication; unknown old records remain unmanaged.

Actual published-files tests first confirm intact catalog readiness, then modify an
original support while keeping payload/reference checksums unchanged, remove the
inventory field, or substitute the old storage identity. Fresh adapters return Protocol
and never acknowledge TargetReady. Focused race suites, full make lint and make test
(including actual PDF runtime and external examples) passed. Independent coverage,
lifecycle and evidence schema fixture validation passed; git diff --check is clean.
Logs are results/persistent-support-{lint,test,full-test}.txt.

The actual tensor_comparison consumer was rerun against the changed storage format.
Its refreshed report retains synthetic quality conclusions and raw timings, with actual
logical index sizes 3445 dense / 3462 tensor bytes, including original supports. This
is not a live model quality or production latency result. Fenced complete-namespace
bootstrap verifiers and their real backend integration remain pending; retaining
supports alone does not establish namespace coverage. No final audits, release or
issue closure were performed at this checkpoint.

## Actual fenced dense/tensor bootstrap inventory

Added InventoryObserver and FencedInventoryVerifier contracts and concrete persistent
dense/tensor observers. The verifier owns registration/envelope snapshots, admits the
exact configured profile, holds every target fence simultaneously in deterministic
order and returns the original digest only after all observations succeed. Missing or
duplicate callbacks and nested failure swallowed by a port cannot produce confirmation.
The host observer contract requires synchronous exactly-once callback behavior; arbitrary
custom Go implementations are not sandboxed.

Actual targets enumerate bounded immediate keys under the same cross-process lock as
Stage/Cleanup, validate exact manifest/catalog supports and actual payload digests, and
account for all entries in complete inventory. Unmanaged basenames remain opaque and
must exist without aliasing a managed entry. Delta omission does not invent removal.
Dense MaxScanRecords / tensor MaxRecords bound entries and total verified payloads;
existing byte limits remain enforced. No unknown payload is opened and no mutation is
performed by the observers. The receipt confirms an observation point, not a lease or
an atomic backend-plus-ledger commit.

Real single-target tests publish on disk, create unknown legacy files, restart adapters,
reject incomplete complete coverage, import verified ownership into a fresh filestore,
preserve the unknown bytes, allow delta omission and reject cancellation. An actual
Stage attempt receives Conflict while the observation callback retains its file fence.
The joint dense+tensor test imports both actual catalogs, captures both publication
pins and rejects forged original supports without changing destination generation.
Core tests independently verify simultaneous nesting, owned input/configuration,
unsupported profiles and callback failure propagation; filesystem tests cover reserved
locks, bounded enumeration and path rejection.

Focused lifecycle/persistent race suites and final full make lint / make test passed,
including external examples and actual PDF parser runtime. Logs:
results/fenced-inventory-{lint,test,full-test}.txt. LIFE-21/LIFE-22 remain partial:
lexical/graph observers, those actual joint bootstrap profiles, broader fault/backfill
acceptance and other mandatory experiments still remain. Both final independent audits
are deferred until the complete implementation. No release or issue closure occurred.

## Lexical/graph held-fence bootstrap and complete target profiles

Added actual retained lexical/graph InventoryObservers with explicit maxEntries and
maxRecords bounds. Observation holds the actual mutation mutex via nonwaiting TryRLock
through all nested target callbacks, checks exact retained manifest identity/payload/
original-support sets and actual record references, and distinguishes missing volatile
data from ready durable checkpoints. Every supplied observer now checks cancellation
after its callback. No metadata projection/model callback occurs under these fences.

Opaque manifest:<operation-id> keys account for other retained revisions. Graph
host:<basis-id> keys remain explicitly unmanaged. Real adapter tests cover omitted
complete sources, accepted delta omission, successful opaque retained revision coverage,
entry and record caps, cancellation, fresh-adapter memory loss despite a ready ledger,
and unchanged host basis after observation. Internal tests use the actual mutation mutex
to prove it is held during the callback, released afterwards and returns immediate
Conflict with a writer; callback cancellation and invalid bounds are checked.

Actual filestore import/publication tests now cover all three joint profiles
(dense+lexical, dense+tensor, dense+graph), with independent forged original-support
rejection for each named target and unchanged destination generation. Additional
actual single-target lexical/graph imports capture only the selected target. Existing
persistent suites supply the dense/tensor single-target coverage. The verifier confirms
an observation point; it does not promise a lease or a distributed transaction across
backend fences and later ledger CAS.

Final full make lint / make test passed, including root race suites, external examples
and actual PDF runtime. Independent coverage/lifecycle/evidence schema fixtures passed;
git diff --check is clean. Logs: results/volatile-bootstrap-{lint,test,full-test}.txt.
LIFE-21/LIFE-22 now record this supported-target verification. Broader lifecycle fault
acceptance, migration consumers, evidence/quality experiments and other matrix requirements
remain open; no total completion percentage or final-audit result is inferred from
these tests. No release or issue closure was performed.

## Actual cleanup lost-response restart and deadline recovery

Actual cleanup fault tests now cover dense, lexical, tensor and graph targets across
all three joint profiles. The physical target deletes retained data, then the wrapper
loses its response. A new Cleaner loads durable unknown progress, blocks another
Attempt, inspects the actual target once and confirms completion without repeating
deletion. The retired pinned snapshot fails Unavailable, the other target completes
normally, completed reconciliation performs no I/O and FAQ remains readable.

The fake-clock actual-target outage profile returns waiting until restoration. A new
Cleaner loads the real durable queue on each host-driven attempt at 1/2/4/4 capped
intervals all the way to the 60-second acknowledgment-based deadline. Same-clock calls
never dispatch; overdue ordinary calls persist pending status without I/O. New scoped
reads still honor the tombstone barrier while physical old data remains. The old access
token expires during the clock advance; the host explicitly renews authorization for
that retained publication before its test read. Explicit recovery still honors NextAt
and then performs actual target deletion, preserving FAQ and completing the joint job.

The tests exposed DEFECT-CLEANUP-01/P1: unregistered lexical deletion and changed
original supports both removed actual retained records and returned complete. The
saved pre-fix log contains both failures. Lexical Cleanup now requires durable exact
owner/retired inventory and registered unknown dispatch under its mutation fence;
InspectCleanup also validates registration/inventory without deletion. Three negative
cases (unregistered, pending-only, forged-support) now preserve old ready records.
The TЗ includes the confirmed defect and consumers' migration instructions.

Focused race tests and final full make lint / make test passed, including adapters,
external examples and actual PDF runtime. Logs: results/cleanup-restart-{lint,test,
full-test,lexical-repro}.txt. LIFE-18/LIFE-23 and the defect row record the supported
scope. Wider transition faults, actual newer-writer cleanup protection, graph shared
support acceptance and remaining matrix/experiment requirements are still open. Both
final independent audits remain required; no release or issue closure was performed.

## Actual newer-writer cleanup protection and shared graph support accounting

All actual targets across three joint profiles now have a deterministic publication
interleaving test: a cleanup request captures the tombstone publication, then a real
fully staged r3 writer publishes before the backend cleanup fence check. Cleanup returns
Conflict/OutcomeUnknown and leaves old data intact. A new Cleaner inspects once,
confirms waiting, respects backoff, and removes only r1 using the current source fence.
The job remains restricted to captured r1 IDs; r3 reuses artifact ID p1 safely and
both targets retain r3 plus FAQ. Two dispatches are explicit host calls separated by
inspection/backoff; no implicit retry occurs. This reproduces the relevant concurrency
ordering without relying on scheduler timing or claiming a broader process fault matrix.

The dense+graph fixture now executes actual edges and fingerprints their complete
identity/from/to/relation/metadata shape. Its declared reference graph uses service-a,
db-x and depends_on/e1 with policy@r1:p1 and faq@r1:f1 original UTF8 supports. Joint
publication and actual both-target cleanup verify two-to-one-to-zero source support
accounting. The shared edge survives policy cleanup, disappears after the last managed
support, and remains under an explicit immutable host foundation with zero invented
source citations. Without explicitly selecting that host basis, the cleaned graph is
empty. Canonical graph metadata remains independent of source support identity.

Focused actual integration race tests and final full make lint / make test passed,
including all adapters, external examples and actual PDF runtime. Logs:
results/cleanup-newer-{lint,test,full-test}.txt. Matrix LIFE-19/LIFE-20 now records the
proved scope. Previous cleanup restart logs and DEFECT-CLEANUP-01 evidence have been
saved and its matrix entries synchronized. Broader transition faults and remaining
retrieval/evidence/experiment/migration requirements remain open; both final independent
audits are still required. No release or issue closure was performed.

## Actual prepare, stage and publication CAS acknowledgement faults

Added actual durable-store injection immediately before and after selected CAS effects.
No fake target is substituted: dense/tensor use physical catalogs/payloads and
lexical/graph retain actual source inventories. Target call counters only wrap those
real ports. Each failed operation is checked against a fresh durable Load, not a
candidate manifest returned alongside the error.

The staging matrix has 24 cases: all four targets across three joint profiles, dispatch
unknown vs target-ready checkpoint, and before/after commit. A dispatch checkpoint
failure never calls the target. An unknown committed dispatch with no physical effect
is inspected pending before an explicit Stage. Actual target data committed before a
failed ready CAS is inspected once without repeat staging. A committed ready CAS with
lost response replays without Inspect/Stage. All continuations perform exactly one
actual target Stage, preserve old logical publication during uncertainty, then publish
r2 consistently while retained authorized r1 snapshots and FAQ remain intact.

A further 12 actual joint cases cover Prepare and publication CAS before/after commit.
Fresh Executor Prepare/Publish recovery loads known state without target calls; committed
operations do not repeat CAS, while uncommitted ones resume by explicit host continuation.
Read checks prove the correct durable old/new publication rather than inferred rollback
or success. Combined with actual lost backend responses and post-publication cancellation
fixtures, this verifies LIFE-14 for declared uncertainty contracts. LIFE-13 remains
partial: cleanup checkpoint CAS and physical backend operation phase faults are still
required. In-process target recovery does not claim persistent data across process loss.

Focused 36-case integration race tests and final full make lint / make test passed,
including all modules, external examples and actual PDF runtime. Independent lifecycle
schema fixtures and git diff --check passed. Logs:
results/stage-checkpoint-{lint,test,full-test}.txt. Current experiment credentials remain
absent in the task environment; live quality gates remain unverified and are not counted
as success. Other mandatory matrix work remains open and both final independent audits
are still required. No release or issue closure was performed.

## Actual cleanup job/dispatch/completion acknowledgement faults

Added 48 actual cleanup checkpoint cases across dense/lexical/tensor/graph targets in
all three joint profiles: job Begin, unknown dispatch, per-target completion and final
whole-job completion, each before/after durable CAS. Store faults surround actual
filestore writes; target wrappers delegate real physical or retained-record operations.
Durable Load distinguishes absent/waiting/unknown/done state from a returned candidate
job accompanying an error. Tombstone visibility remains independent of those failures.

Failed dispatch acknowledgements never invoke the target. Committed unknown dispatch
without physical effect is inspected waiting before a later explicit due Attempt.
Failed completion CAS after actual deletion is inspected done without another deletion.
A committed done acknowledgement loss replays without target I/O. Fresh Begin reuses
committed jobs and only explicitly creates an uncommitted one. Every selected target is
actually deleted once; the job finishes across both targets, unavailable old snapshots
are not substituted, and FAQ remains intact. The final-target cases separately cover
the CAS that marks the entire job complete.

Final full make lint / make test passed, including the 48 cases, all adapters/external
examples and actual PDF runtime. The standalone lint initially encountered a shared
runner lock, then the final make lint completed successfully; no lock bypass or check
suppression was added. Independent lifecycle schema fixtures and git diff --check pass.
Logs: results/cleanup-checkpoint-{lint,full-test,test}.txt; test.txt is an extraction of
all 48 passing cases from the authoritative full run rather than an older focused log.
LIFE-13 remains partial: physical backend install/retirement phase faults still need
acceptance. Other matrix and live experiment requirements remain open, and both final
independent audits remain required. No release or issue closure was performed.

## Dense/tensor actual physical process-crash recovery

Added separate-process tests for each persistent target at five observed physical
states: first synced payload in the owned staging directory, synced staging catalog,
installed catalog after rename/root sync, retained cleanup trash, and removed cleanup
files. A test-only context checks actual paths and exits the child process with the
required crash status, bypassing Go cleanup defers. The parent checks that exact exit
rather than accepting a failed helper as a crash. Child temporary fixtures stay under
a parent-owned test root. No production hooks or adapter behavior were changed.

Fresh filestore/adapter instances reconcile durable unknown state against real files.
Incomplete and catalog-only staging inspect pending and never alter current r0 reads;
installed data inspect ready. Explicit continuation publishes r1, and returned source
locations are checked for exact r0/r1 provenance rather than merely equal result counts.
Cleanup restart preserves the tombstone barrier; old retired snapshots are unavailable.
Retained trash inspects waiting and is removed only by an explicit due Attempt. Already
removed files inspect done and complete without a second destructive call. Actual port
counters distinguish the one Inspect from optional continuation.

Focused process-crash race suites and final full make lint / make test passed, including
all modules, external examples and actual PDF runtime. Logs:
results/physical-crash-{lint,test,full-test}.txt. LIFE-13 now records supported process
failure acceptance alongside 84 coordinator CAS cases. This verifies observed process
termination boundaries, not hardware power loss or unspecified filesystem guarantees.
Mandatory remaining conformance/evidence/experiment/migration matrix work and both final
independent audits are still open. No release or issue closure was performed.

## Admitted persistent query materialization boundary

Added optional lifecycle.PayloadReader for dense/tensor query payload reads. The default
continues actual bounded local-file reading. Host readers receive only the admitted exact
reference/path/byte cap; lifecycle Stage/Inspect/bootstrap retain independent physical-file
verification. Invalid reference/path/cap, MaxInt64 and typed nil are rejected before host
I/O. Constructors reject typed nil ports. Returned bytes remain subject to retained digest,
payload schema/reference/shape, byte cap and final freshness validation.

Actual durable Stage/Publish/query regressions cover successful original bytes, substituted
bytes, oversized output, cancellation/deadline and arbitrary storage failure. Failure
returns no hits, dispatches once, and never leaks host storage error text. Context/protocol
semantics remain distinct; ordinary physical/host read errors remain unavailable. Internal
boundary regressions also cover canceled-before-I/O, invalid-before-I/O, cancellation during
a call and actual default-file size enforcement.

Final make lint and make test passed on this code, including all modules/race, external
examples and actual PDF runtime. Focused race logs and full command logs are retained in
results/payload-read-{test,lint,full-test}.txt. Makefile lint now uses the tool's standard
--allow-serial-runners flag to wait for a concurrent global lint lock; all checks remain
enabled. The new port enables direct external consumer materialization instrumentation;
it does not itself complete the still-pending external persistent conformance fixtures.
SCOPE-17/TENSOR-07 retain partial statuses. Remaining scope/experiments/migration and both
final independent audits remain required. No release, publication or issue closure.

## External persistent scope conformance and planner admission

The separate example.com/ragyconsumer module now stages/publishes actual dense/tensor
files with durable manifests, constructs fresh adapter instances and runs the public
nine-case direct-backend suite against each. Its host PayloadReader observes bounded
physical-file materialization, admitted artifact IDs and actual context deadlines; the
request projection adds no protective gate. Forbidden private/foreign payloads never
materialize. Missing/expired/revoked/canceled requests perform no query payload I/O;
revocation during I/O suppresses delivery. This is the declared local-filesystem scope
profile and does not certify unspecified external backends.

Three additional planner cases per adapter cover empty, contradictory and unmapped plan
filters. Initial unmapped cases failed the required error classification, exposing
DEFECT-PLAN-01: preliminary leaf intersections returned schema errors while common
PrepareRead omitted planned predicates. PrepareRead now validates query AND plan before
mandatory intersection; persistent leaves remove their duplicate path. Tensor candidate
search uses the same admission, and graph node/plan negotiation preserves protected
unsupported semantics. Unsupported execution errors remain non-skippable.

Regression evidence includes zero backend calls in aggregate preflight and zero metadata
projection for an actual published graph with unsupported plan. The focused affected
root suites passed under race. All 24 external persistent scope/planner scenarios passed
using GOWORK=off go test -race -v ./..., independently of workspace module resolution.
Final make lint and make test passed on the changed code, including all modules, race,
external examples and actual PDF integration. Retained logs:
results/external-persistent-{consumer-test,consumer-lint,plan-test,full-test,full-lint}.txt.

TENSOR-07 records verified local-filesystem integration/public conformance; SCOPE-17
retains partial status for other recipe/export/target combinations. The matrix still
contains broader unverified requirements and stale rows requiring evidence review.
Remaining experiments/migration/cross-capability acceptance and both final independent
whole-scope audits remain required. No release, publication or issue closure.

## Actual persistent integer metadata and retained normalization

External consumer fixtures stage/publish actual dense/tensor JSON files and reopen adapters
with typed int64 metadata at adjacent values 9007199254740992/9007199254740993 and the
MinInt64/MaxInt64 bounds. Nine cases per target check unrestricted full roundtrip,
Eq/adjacent/bounds, In, query NotEq and mandatory Eq/conflict. Physical payload reader
observations must match the exact allowed integer artifact identities, not merely final
metadata projection. Five invalid host-codec Stage cases per target cover fraction,
positive/negative overflow, excessive exponent and a fraction beyond finite precision.
All reject before any target file/lock write and cannot become published.

Initial tests reproduced DEFECT-INTEGER-CATALOG-01: normalized decoded attributes were
validated then discarded, leaving json.Number at the typed matcher boundary. Stored
numbers themselves were exact. The fix retains owned schema-normalized attributes in
each decoded catalog record before admission; it does not change artifact/support/payload
identity or roundtrip through float. The original failing suite log is retained in
results/integer-storage-before-fix.txt. A separate copied-source consumer run with only
the normalization assignment removed reproduced erroneous query NotEq and empty mandatory
Eq again: results/integer-storage-isolated-repro.txt. The main worktree was not reverted.

An attempted mandatory NotEq fixture clarified a limit rather than demonstrating leakage:
access.Scoped rejects it before creating a binding. The supported mandatory profile is
Eq/In/And. The external explicit rejection fixture passed, and this defect is P2 retrieval
correctness/availability, not a demonstrated authorization leak. Do not expand the
mandatory profile or infer unsupported guarantees to make an audit claim.

All 29 new external integer cases passed under race with GOWORK=off; the earlier persistent
scope/planner suites and comparison example also passed. Final make lint and make test
passed on the changed code, including all modules/race, external examples and actual PDF
runtime. Logs: results/integer-storage-{consumer-test,full-lint,full-test}.txt.
BUG1-04 now verifies the declared local persistent JSON profile; pgvector's decoder is
still a unit wire boundary, not a claimed live database run. BUG1-05 remains partial for
all-wire-adapter acceptance. Task §8.8, design/migration, consumer docs and the atomic
matrix reflect the actual defect and declared limits. Whole-scope acceptance/experiments
and both final independent audits remain outstanding. No release or issue closure.

## Library-owned integer wire metadata boundaries

Inventoried the production MetadataCodec/RawAttributes storage adapter boundaries.
Elasticsearch raw Hit.Source, Qdrant Point.Attributes and pgvector stored attributes JSON
now have actual Store projection/write tests for adjacent IDs and int64 bounds. Exact
integer membership arrays/query arguments are observed at the injected Client/DB boundary.
Neo4j Runner is typed BYOT rather than a library-owned JSON decoder; existing graph integer
normalization and its full module suite remain the relevant contract evidence. Other
adapters do not introduce a further MetadataCodec/RawAttributes storage boundary.

49 new cases passed under race: nine Elasticsearch, twenty Qdrant and twenty pgvector.
The tests use real adapter Encode/render/serialize/project/Decode logic and injected ports;
no external service/driver/database engine precision is asserted. Inventory and limits:
wire-metadata.md. Focused logs: results/{elasticsearch,qdrant,pgvector}-integer-wire-test.txt.

Negative custom codec tests reproduced DEFECT-CODEC-WRITE-01 before correction. Qdrant
sent every malformed integer/non-finite case into Client.Upsert and returned success.
pgvector already suppressed DB writes but reported NaN/Inf serialization errors outside
ErrInvalidArgument. Before-fix outputs are separately retained in
results/{qdrant,pgvector}-integer-wire-before-fix.txt. Both write paths now retain owned
schema-normalized codec output before constructing/serializing transport payloads.
Invalid output fails as ErrInvalidArgument with zero Client/DB writes. Final negative
batches include a valid prefix before the malformed record; no partial dispatch occurs.

Final make lint and make test passed on this code across all modules/race, examples and
actual PDF integration. Logs: results/integer-wire-full-{lint,test}.txt. BUG1-05 records
verified library-owned boundary profiles; custom external driver/service semantics remain
consumer responsibilities. New atomic BUG1-09 and task §8.9 record the confirmed input
validation/error contract defect. GATE-08/09 are verified at this checkpoint and must be
rerun after later source changes. Whole-scope requirements/experiments, final migration
review and both independent final audits remain outstanding. No release or issue closure.


## Concurrent recipe budget and attempt isolation

Two new external-package tests execute actual recipes through concurrency-safe
injected host ports. TestConcurrentCommunityVariantsCannotOverrunLastSharedModelCall
uses a quote barrier and holds the winning model in flight. Two variants share a
one-model-call/30-cost attempt ledger: one returns insufficient/budget-exhausted
without dispatch while the other completes, settling 32 input/16 output tokens and
30 cost. Exactly one model call executes, with no outstanding/unknown leases or
retry. TestParallelRunsOfOneRecipeOwnIndependentAttempts runs the same text recipe
instance concurrently. Each attempt completes with its own 2-model/2-retrieval
budget and 200-input/40-output/60-cost usage; results do not alias one another, host
storage or request metadata. No shared organization quota is introduced.

Both tests passed twenty repeats under race on the final source state:
results/recipe-concurrency-test.txt. Config/README and design document callback
concurrency, stable caller inputs, per-Run ledgers and explicitly shared summary
ledgers. RECIPE-11 records this declared execution contract; these tests do not
substitute for live model quality or comparative recipe experiments.

Initial full checks stopped on Go dependency lock writes outside the sandbox.
A private writable copy of the module cache at /tmp/ragy-implementation-mod-cache
allows the same unchanged Makefile checks to run, including all configured lint
checks. The shared module cache is not modified.

Final make lint and make test both exited successfully with that private module
cache. Checks included all modules/race, external consumer and examples, actual
PDF runtime and comparison fixtures. Logs: results/recipe-concurrency-full-lint.txt
and results/recipe-concurrency-full-test.txt. GATE-08/09 are verified at this
checkpoint and must be rerun after subsequent source changes. Whole-scope
experiments, final migration review and both independent final audits remain
outstanding. No release or issue closure.


## Capability matrix and managed lexical documentation

Added capabilities.md after reading the actual access/leaf/query declarations,
filter mandatory profile, transport walkers, managed read/staging code, durable
store requirements and public consumer fixtures. It distinguishes scalar Eq/In/And
mandatory authorization from optional schema-valid query operators; current-only
transport adapters from pinned managed targets; durable local dense/tensor payloads
from volatile lexical/graph records; raw storage from scoped hydration/traversal.
Each supplied target, joint profile and optional composition path names its actual
verification and limitations. Public custom-graph conformance and comparative
model experiments remain explicitly unverified, so GATE-06 remains partial.

The managed BM25 README incorrectly described provenance propagation as unfinished
and treated all partial pins alike. Staging already retains supplied locations
and adds the canonical artifact locator; explicit partial pins reject the excluded
target, while complete capture still requires all requested targets ready. The
documentation now reflects those existing contracts and avoids claiming original
quotation precision from an artifact reference alone. No runtime API or behavior
changed in this increment.

TestProductionDocsAvoidMigrationLanguage passed; result saved in
results/capability-doc-check.txt. Local matrix evidence links were checked against
the actual workspace. Previous full module/race lint/test checkpoint remains
results/recipe-concurrency-full-{lint,test}.txt; later runtime changes require new
full checks. Whole-scope acceptance and both independent final audits remain open.


## External managed graph read conformance

Added graph_read_unix_test.go in the separate example.com/ragyconsumer module.
The fixture publishes actual managed graph records through durable Executor
prepare/stage/publish and CapturePublication. BYOT node/edge metadata uses an
explicit schema including observation identity. Initial fixture compilation/schema
errors were corrected; they were fixture setup failures, not runtime defects.

The public nine-case direct read suite passes under GOWORK=off/race. A public seed
links through a private bridge to another public node, and a foreign seed is also
requested. CloneMeta is instrumented before downstream projection: private/foreign
facts, both bridge edges and the public fact behind the forbidden bridge cannot
materialize. Revocation during actual cloning suppresses output. The wrapper only
forwards admission/Retrieve and records deadline; it adds no authorization checks,
filtering or final freshness gate. Counts are reset only after complete real setup.
Three direct planner cases pass: empty plan retains allowed public output, conflict
is empty and unsupported schema rejects during admission/execution before cloning.

Focused race log: results/external-graph-read-test.txt. Full make lint and make test
both exited successfully after assertion helpers resolved test-only cognitive
complexity. Logs: results/external-graph-full-{lint,test}.txt. No lint check was
disabled. Public leaf suite evidence is specific to the actual managed reference
profile, not an arbitrary external graph engine or persistent graph payload.
Requirements/capabilities/consumer docs retain mixed-path and comparative-model
acceptance as outstanding. Both final independent audits remain pending.


## External managed graph exact scoped lookup

Added eight direct FindByIDs cases to the external consumer fixture: allowed,
contradictory/unsupported node predicates, missing binding, revoked/expired/
canceled reads and revocation during actual metadata cloning. This invokes the
managed target directly without Backend projection or additional host gates.
Private/foreign facts never clone; nonexistent IDs yield no payload; duplicate
requested IDs stay deduplicated. Revocation after cloning begins suppresses the
complete snapshot/support/conflict envelope. Ordinary successful lookup retains
exact original graph-source@r1 support references and no traversed edges.

The public node behind the private bridge is denied by traversal reachability,
but is allowed by explicit ID lookup under the same mandatory predicate. Tests
assert this distinction: forbidden traversal does not turn an otherwise public
fact into a globally forbidden lookup. Focused race results are saved in
results/external-graph-lookup-test.txt; final make lint and make test passed after
flattening a test-only nested assertion. Final logs:
results/external-graph-lookup-full-{lint,test}.txt. No runtime contract changed.
Matrix/capability/consumer docs include the observed lookup profile and retain
mixed-path/model experiment acceptance as outstanding.

Provider environment was rechecked without reading/logging secret values:
OPENAI_API_KEY, ANTHROPIC_API_KEY and RAGY_EXPERIMENT_MODEL remain absent. Live
model/tokenizer/comparative acceptance is still unperformed, not a negative
quality result. Other implementation/acceptance work remains available; the goal
is active and neither release nor issue closure is authorized.


## Text recipe experiment evaluator and fixed qrels

Added an executable offline consumer example in recipe_comparison with embedded
fixed four-document corpus and all five task cases. It requires a complete grid
of twenty baseline/single/multi/decomposition observations and retains original
ordered IDs, scope/publication, configured identities, actual call/token/cost
usage, known/unknown accounting and raw nanoseconds. It neither calls a model
nor attests capture provenance. No live or scripted quality report is saved.

Executable evaluator tests cover per-query document Recall@3 versus first-relevant
MRR@3, top-K truncation, the separate no-answer denominator/error count, complete
grid/duplicate/unknown/foreign rejection, failed executions, unknown accounting,
calls/deadline overruns, no-answer and isolated MRR regression, strict wire input
and exact uint64 JSON usage roundtrip. A contract-only positive metric fixture
cannot change baseline default. Initial MRR negative fixture was corrected so it
actually isolates MRR regression while retaining sufficient Recall gain; it is
not claimed as a library defect. Configured lint rules remain enabled; profile
names are immutable constants and the list is returned fresh.

Final make lint and make test passed including evaluator race tests and all prior
modules/examples/PDF/conformance. Logs:
results/recipe-comparison-evaluator-full-{lint,test}.txt. Matrix RECIPE-12/13
records partial delivery of scoring; real BM25/live model captures, exact model
tokenizer qualification and the actual comparative experiment remain unperformed.
Model credentials/configuration are absent as recorded in the prior checkpoint.
This example is external consumer tooling, not a runtime-core experiment framework.
Whole-scope completion and both independent final audits remain outstanding.


## Consumer model bindings and bounded executable tokenizer port

Added tokenizer.go and model_ports.go to the external text experiment consumer.
The counter executes a trusted absolute host program directly, with complete wire
request on stdin, no shell/retry/inherited credentials, 32 KiB input/512-byte output
bounds and a two-second local computation timeout capped by the host deadline.
Only matching model/tokenizer identity and positive uint64 token receipts succeed.
Malformed/unknown/foreign/zero/negative/oversized output and child failure reject
with sanitized error; cancellation preserves context classification. A specific
G204 annotation documents this required trusted executable boundary, following
the existing parser adapter pattern; no global security rule was disabled.

Bindings require an existing host attempt deadline and configure the actual
optional structured planner/assessor clients with fixed executable response
schemas/instructions and 30-unit mock call price. The separate consumer module
locally imports the optional adapter; runtime core acquires no provider dependency.
Actual HTTP/subprocess protocol tests exercise both bindings and exact declared
20-input/5-output/30-cost fixture usage once per stage. Credentials set in the
parent test environment do not reach the counter child. Additional validators
reject missing/null/unknown output fields while accepting explicitly empty
planning and false sufficiency.

Initial test fixtures omitted required transport role/total_tokens and used an
incorrect binding helper name; these setup errors were corrected. The child test
binary's race-runtime exit delay required the declared two-second counter budget;
race instrumentation was retained. New fixtures use a fixed count and scripted
HTTP output: they establish no exact tokenizer qualification or model quality.
Full BM25/recipes capture orchestration and live comparison remain outstanding.

Final make lint and make test passed across all modules/race/examples/actual PDF
and the new consumer port tests. Logs: results/recipe-model-ports-full-{lint,test}.txt.
Matrix/design/consumer documentation retains partial experiment acceptance. No
release, issue closure or goal completion is claimed; final independent audits
remain outstanding.


## Complete text baseline/recipe capture consumer

The executable consumer now exposes separate explicit -capture and offline -input
commands. Capture executes an owned scoped readonly BM25 snapshot, five baseline
and fifteen actual recipe attempts, optional structured model clients and the
host counter. It preserves selected IDs, exact original source/revision/
representation references, actual boundary retrieval/HTTP calls, settled aggregate
usage plus independent per-model-stage raw usage, scope/publication and elapsed
nanoseconds. Fixed corpus content/tenant/ID admission verifies source supports.
Foreign adversarial corpus payload never reaches model input or exported captures.
Source reference association validation rejects foreign/missing references in
offline observations; raw model stages do not expose BYOT metadata or credentials.

A full contract-fixture run passed under race: twenty samples and thirty actual
HTTP dispatches, once per model stage, with the real BM25 engine and executable
counter protocol. Its scripted response/count remains explicitly contract-fixture,
not live quality. HTTP failure independently verifies one dispatch/no retry, failed
empty selected payload, preserved raw stage accounting and explicit unknown usage.
Canceled/unconfigured capture rejects, and missing live credentials writes no file.
The actual CLI without credentials returned the configuration error before artifact
creation; results/recipe-live-preflight.txt is a failed preflight, not experiment
completion or a measured negative quality result.

A specific G703 annotation documents read-only stat of the explicit absolute
trusted host executable, following the existing G204 boundary. No global lint
rule was disabled. Configured lint and every module/race/examples/actual PDF test
passed on the final code; results/recipe-capture-full-{lint,test}.txt. Consumer
docs/design and RECIPE-12 now describe implemented orchestration while retaining
live/tokenizer comparative acceptance as unperformed. Live model/tokenizer settings
were requested asynchronously; other requirement work remains available. Neither
release nor issue closure nor final goal completion is claimed. Both final
independent audits remain outstanding.


## Combined HTTP extraction, history and graph publication

`adapters/openai/structured/publication_integration_unix_test.go` executes
HTTP model protocol→bounded core extraction→typed resolver→immutable filesystem
history→materializer→durable lifecycle executor→managed graph publication→scoped
pinned FindByIDs→tombstone publication→physical cleanup. The fresh history store
reopens the retained record after graph retirement; explicit host original-locator
admission governs that retention. The stale graph pin returns ErrUnavailable,
and the new tombstone snapshot yields no node. The integer 9007199254740993,
canonical entity ID and exact original source reference survive every boundary.
The fixture observes one HTTP call and settled cost 45, without retries.

This is a real transport/storage/library integration with scripted model output
and fixed tokenizer count. It does not establish live model quality, the complete
Service/Database/Team reference corpus extraction, or configuration recomputation
from a live provider. GRAPH-01/03/04/05 remain partial for those specific remaining
acceptance requirements. Independent alias/shared support/policy history fixtures
continue to supply their separate evidence.

Configured `make lint` and `make test` passed for the final code across all
modules, race checks, examples and actual PDF parser tests. Evidence:
`results/graph-publication-full-lint.txt` and
`results/graph-publication-full-test.txt`. No lint rule or test instrumentation
was disabled. The goal remains active; live comparisons and final two independent
audits remain outstanding.


## Summary invalidation through durable source tombstone and graph cleanup

`recipe/graphsummary/lifecycle_integration_unix_test.go` exercises both community
and global summaries against actual staged/published managed graph revisions and
a scoped source.Reader. The injected host catalog reads the durable lifecycle
publication on every admission and explicitly permits only live source revisions.
Source payload rows remain retained, demonstrating that physical availability
does not substitute for permission. Durable tombstone publication denies summary
Resolve before another source payload load, even while the old graph pin is still
readable. Actual cleanup makes that graph pin ErrUnavailable. A freshly captured
publication cannot be adopted by the already-created immutable summary.

Both profiles preserve model call bounds (one community call or three global
calls), add no model dispatches on denied Resolve, and expose no summary text on
protection failure. The positive and negative combined cases passed under race.
An initial test setup omitted the PinnedPublication capability while preparing
an already-pinned binding; that helper was corrected without changing production
code. Configured lint for the package passes without rule suppression.

The model port remains a deterministic contract fixture, and original retention
policy remains host-owned. This integration does not claim live summary quality
or automatic source deletion detection by ragy without host admission. GRAPH-06
retains partial acceptance until the full live comparative experiment.


## Full text experiment configuration and seed policy

The text capture now embeds an executable reference configuration instead of only
a human-readable config label. Its SHA-256 binds corpus, instruction/schema
identities, explicit BM25 K1/B, TopK/fusion, query/retrieval/model limits, token/cost
caps, per-call reservations, timing and byte bounds. The offline evaluator rejects
missing or altered configuration even when its self-hash matches. Constructor
checks match each recorded strategy limit to the actual recipe configuration.
Seed policy states that no provider seed is requested; deterministic sampling is
not asserted. Cost units remain the explicit fixture budget, not billing currency.
No credentials or host executable paths are captured.

Both summary lifecycle profiles and all configured module/race/example/PDF checks
passed before the experiment configuration addition. Logs:
`results/summary-lifecycle-full-lint.txt` and
`results/summary-lifecycle-full-test.txt`. Configuration acceptance tests reject
altered seed policy, corpus identity, token/call limits and mismatched digest.
An initial configuration test used a float for FusionK while the public recipe
contract uses int; the artifact field now matches that contract exactly.

The full configured `make test` passed after the configuration implementation,
including the actual BM25/HTTP/subprocess capture fixture under race. Configured
`make lint` passes on the final tree. A repeated test-case label initially triggered
goconst; renaming that label resolved it without disabling the rule or changing
execution controls. The affected configuration tests were rerun under race and
passed. Evidence: `results/experiment-config-full-test.txt`,
`results/experiment-config-full-lint.txt`, `results/experiment-config-test.txt`.
Live configuration was rechecked without revealing values: OPENAI_API_KEY,
RAGY_EXPERIMENT_MODEL, RAGY_EXPERIMENT_TOKENIZER and
RAGY_EXPERIMENT_TOKENIZER_ID remain absent. Live experiments are unperformed,
not negative measured results. Other implementation/acceptance work remains;
no completion, release or issue closure is claimed.


## Tensor experiment configuration, scope and input byte bound

The actual persistent tensor comparison now records canonical corpus/qrels SHA-256
and an exact hashed execution configuration: both local filesystem adapters,
scoring modes, TopK/candidate/repetition values, storage/scan/byte/deadline limits,
float representation and quality thresholds. Seed policy explicitly records saved
hand-defined data without random generation; no generated seed is invented. Both
actual scope snapshots and full exact source references accompany each ranking.
The stored tensor-comparison.json was regenerated by the real consumer and actual
persistent targets, not by patching reported observations.

Code review found that a LimitReader ending at the input limit could conceal
trailing bytes behind artificial EOF. The loader now reads at most limit+1 and
rejects an oversized file before JSON decoding. A valid fixture followed by excess
whitespace reproduces the input condition; the regression rejects it. This is a
consumer experiment input defect, not a claimed core authorization leak. The
configuration digest changes when saved query data changes, and integration checks
match each retrieved hit to its exact dense-vector/token-matrix source reference.
Synthetic quality limitations and null latency percentiles remain unchanged.

Configured `make lint` passed across all modules. `make test` initially found a
new source comment containing the repository-banned phrase "instead of"; the
comment was rewritten directly and the full `make test` rerun passed, including
race/examples and actual PDF parser. Results:
`results/tensor-config-full-lint.txt`, `results/tensor-config-full-test.txt`.
Tensor configuration/input-bound tests passed under race, and the actual command
regenerated `results/tensor-comparison.json` successfully. GATE-04 was reconciled
against current named actual parser/index/retrieve/retained-resolution tests; its
explicit OCR simulation does not claim a working OCR model. GATE-05 continues to
track unperformed live text and graph comparative acceptance.

The complete external consumer package was rerun with GOWORK=off and race, using
its own public dependency declarations: all TestExternal cases passed. Evidence:
`results/tensor-checkpoint-external-conformance.txt`. This confirms GATE-01's
external-module execution requirement; it does not certify arbitrary host adapters
or finish every cross-path requirement. The goal remains active, with graph
comparative implementation, live experiment qualification and the two final
independent audits still outstanding. No release or issue closure was performed.


## Graph comparative fixture and strict offline evaluator

The external consumer now includes graph_comparison with the reference
Service/Database/Team ontology corpus, explicit production Billing/Pay alias and
staging separation, shared dependency source supports, C1/C2 host membership,
three recipe questions and exact gold support sets. A fourth source supplies
Search/IndexDB/Team B evidence for C2. The fixed comparative state is nonconflicting;
independent core fixtures retain conflict/ambiguity/recomputation acceptance.

The executable offline evaluator requires exactly six baseline/matching-recipe
observations with compatible scope/publication and exact original reference tuples.
It rejects missing/duplicate rows, foreign/wrong-revision/duplicate supports,
missing outcome and failed payload, and incompatible hashed configuration. It
calculates bounded support Recall@3 against the full gold denominator, retains
negative gain, reports unknown/failing usage as unverified budget acceptance, and
preserves raw observations. Exact uint64 roundtrip keeps overflow costs visible.

Tests explicitly use contract-fixture data; they establish no live graph experiment.
Actual hybrid/extractor/graph recipe capture orchestration remains outstanding.
The README states this limit and does not expose an unimplemented live command.
GRAPH-12 remains partial; this step supplies the executable metric/protocol contract
needed for actual producer integration, not a substitute for that integration.

Configured `make lint` and `make test` pass on the graph evaluator checkpoint
across all modules/race/examples/actual PDF integration. The external evaluator
package also passes GOWORK=off race checks. Logs:
`results/graph-evaluator-full-lint.txt`, `results/graph-evaluator-full-test.txt`,
`results/graph-evaluator-test.txt`. Initial test rows omitted the newly required
outcome/stop fields; fixtures were corrected and rerun. Reference bounds use named
constants shared by configuration and budget evaluation; no lint rule was disabled.
Real capture implementation and live model/tokenizer qualification remain separate
unperformed acceptance. No graph quality report, release or issue closure is claimed.


## Actual graph-comparison hybrid baseline producer

The external consumer now builds actual per-source dense publications through the
durable executor, reopens the persistent adapter, and creates an owned BM25 snapshot
on a shared scoped publication binding. Fixed source/question vectors are saved
normalized float32 data, not live embedding results. The shipped RRF deduplicates
by explicit BYOT source metadata while preserving actual original mappings. Each
leaf is preflighted before either dispatch; both execute once under the bounded
parent context. Recorded support references are taken from actual source mappings.

An adversarial foreign tenant is physically staged in dense and included in BM25
input; its source never appears in returned supports. Actual hybrid calls produce
two retrieval dispatches, no model dispatches, and measured raw elapsed time. Three
fixed query baselines each measured support Recall@3=1 in this synthetic saved-vector
fixture; that does not establish a comparative recipe result or production quality.
A canceled parent yields no observation and zero dispatch count.

Initial host schema used reserved source_id and was corrected to source_key, with
no production contract changes. Named baseline controls are embedded in configuration:
BM25 K1/B, RRF k, vector dimension, storage/record bounds and saved-embedding policy.
Package integration checks pass under race and configured lint. Full extractor/
materializer/graph recipe orchestration and live/tokenizer acceptance remain pending.

Configured `make lint` and `make test` passed on the final hybrid producer state
across all modules/race/examples and actual PDF integration. Saved logs:
`results/graph-hybrid-full-lint.txt`, `results/graph-hybrid-full-test.txt`,
`results/graph-hybrid-test.txt`. The broad checks include the offline evaluator
with the updated vector-bearing fixture and exact configuration digest. The
consumer's combined read pin binds the actual durable dense snapshot and owned
readonly lexical corpus; it is a host read binding, not a cross-store transaction
claim. No live graph comparison artifact or completion/release is asserted.


## Actual reference graph resolver/materializer publication producer

External source-bound extraction batches now combine with owned source-prefixed
mention IDs, exact original-locator admission and trusted namespace checks. The
actual typed resolver applies only the explicit production Pay/Billing alias. The
materializer stages source-specific graph payloads through the durable executor;
actual projected payload hashes are validated at executor dispatch. Scoped managed
traversal checks all eight gold canonical entities, five edges and exact original
source reference inventories, with separate staging identity and shared supports.

A two-source typed owner conflict retains both resolver variants and both supports.
Each source can publish its own supported variant; the managed read exposes the
conflict and excludes a chosen Billing winner. The initial negative test assumed
inter-source conflict forbids per-source publication, which contradicts the
materializer's source-selection contract; the test was corrected to assert the
actual required conflict record/no-winner behavior. Consumer graph metadata now
retains owner attributes rather than dropping them during projection. Namespace
poisoning, cross-source support, duplicate and missing source batches reject before
target construction. The extraction inputs remain explicit deterministic contract
outputs; actual core/provider extraction and full recipe capture remain pending.

Configured `make lint` and `make test` passed on the final graph producer state
across all modules/race/examples/actual PDF. Evidence:
`results/graph-materialization-full-lint.txt`,
`results/graph-materialization-full-test.txt`. The final broad checks include
actual gold graph traversal and conflicting-owner no-winner assertions. The
consumer projection correction preserves attributes; it does not change graph
core's source-specific materialization contract. Extraction is still deterministic
fixture input at this checkpoint. Provider extraction, complete recipe capture,
live comparative acceptance and both final independent audits remain outstanding.
No completion, release or issue closure is claimed.


## Actual local graph capture with shared hybrid binding

The consumer now executes shipped graphexpand against the actually published
reference graph. A trusted policy-derived Team A seed, undirected depth two and
the declared node/edge/call bounds yield Billing→LedgerDB evidence with exact
original s1/s2 supports. A fresh per-attempt ledger and parent-bound context record
one graph call, zero model calls/tokens/cost and actual elapsed time. Hybrid and
local captures use the same scope/publication. Canceled/wrong-profile admission
returns no manufactured successful observation.

Raw capture now carries calls_known separately from usage_known. An error result
can lack actual dispatch information, so the local consumer preserves that
uncertainty and fails budget acceptance rather than inferring zero calls. Success
uses the actual core graph result count; the standalone evaluator preserves support
metrics while refusing unknown accounting. The deterministic extraction fixture
is still explicitly contract-only. Provider extraction and community/global
capture/live comparison remain outstanding.

Configured lint initially caught a malformed failed JSON tag introduced while
extracting a repeated outcome literal into a named constant. The tag was restored
to the exact wire field name; no rule was disabled. All graph consumer tests were
rerun with GOWORK=off under race and passed, including actual shared-binding local
capture, canceled admission and unknown-call budget refusal. Final configured
`make lint` passes. Evidence: `results/graph-local-capture-test.txt` and
`results/graph-local-capture-full-lint.txt`.

The full configured `make test` was rerun after the wire-tag correction and passed
on the final tree, including all modules/race/examples/actual PDF integration.
`results/graph-local-capture-full-test.txt` records that final run. No model-backed
community/global capture or live graph experiment is claimed. Final independent
audits remain outstanding and the full goal remains active.

## Actual community/global capture with pinned graph membership and original source reads

The external graph comparison consumer now validates each declared canonical member
against actual pinned graph nodes and captures actual node support inventories.
Exact node sets are checked, including substitution/duplicate negatives. Expected
gold supports do not construct producer membership. Preparation performs two actual
graph lookups, separate from per-attempt summary model accounting.

Original snippets load through the scoped source Reader using exact original
references. Host current-retention admission requires a non-tombstone graph source
publication with matching namespace/source/revision/access plus an exact known
original representation. Both Describe and Load observe current durable metadata.
Locator and membership poisoning deny before original payload load. Tombstone
retirement denies retained original payload through Catalog admission.

Each consumer summary attempt creates an independent source Reader/counter pair
and ledger. The shipped community/global recipes execute against C1/C2 actual
membership and supported canonical members. Export re-resolves immutable derived
summaries under the same scope/publication and preserves original support tuples.
Actual original metadata/payload calls, host model invocations, settled usage,
outcome/stop and elapsed time are recorded. CallsKnown describes observed host
invocations; individual provider wire dispatch provenance still requires the final
capture orchestration. Unknown usage never certifies budget acceptance.

The optional provider summary factory uses actual structured HTTP transport,
strict host output schema and parent-attempt-qualified request counter callback.
Reference reservations are map 1024/256, reduce 2048/512 tokens and thirty fixed
experiment cost units per callback. Controls now hash instructions, schema,
reservations and request/summary byte bounds. These experiment units do not claim
monetary billing. Scripted HTTP/core-model/counter fixtures remain explicit contract
checks, not live model or tokenizer quality acceptance.

Race integration confirms one community callback and three global callbacks;
original s1/s2 and s1/s2/s4 support unions; matching hybrid scope/publication;
independent source counters for four concurrent requests; unknown provider response
with a single attempted callback and no retry; source retirement during token
counting with no model callback; source retirement during callback with known usage
settled and no exported payload; and canceled admission with no callback. HTTP
integration independently observes one provider request with scoped original text
and exact usage receipt. The current transport check is community-only; full global
HTTP/live capture orchestration remains outstanding.

Final configured `make lint` and `make test` passed after exact membership and
namespace/source admission checks were added. All module/race/example/actual PDF
checks use the writable task caches. Evidence:
`results/graph-summary-capture-full-lint.txt`,
`results/graph-summary-capture-full-test.txt`.
The external package had also passed GOWORK=off race/lint, recorded in
`results/graph-summary-capture-test.txt`; the full final run contains the additional
exact-membership test. Independent coverage/lifecycle/evidence schema checkers pass
7/4/16 positive and 12/13/32 negative fixtures respectively. The production docs
language contract passes.

OPENAI_API_KEY and RAGY_EXPERIMENT_MODEL/TOKENIZER/TOKENIZER_ID remain absent in the
host environment at this checkpoint. No secret values were printed. Provider
extraction, complete capture command, tokenizer qualification and live comparative
artifacts remain outstanding; requirements stay partial. Two final independent
audits have not started because mandatory implementation/experiment work remains.
The active goal is not complete; no release/publication/issue closure is performed.

## Full graph provider extraction and capture command

The reference consumer now has a real bounded provider extraction factory with
strict required typed attributes, ontology validation, source-supplied namespaces,
original support projection and a per-source reservation/ledger. The model returns
local mention IDs and snippet ordinals; it cannot supply source references, access
or canonical namespaces. Four actual HTTP extraction requests feed core extraction,
resolver, materializer and durable publication. Final traversal matches all eight
entities, five edges and exact original gold support inventories. Actual conflict
and source-binding regressions remain independent checks.

Full orchestration creates the dense corpus, extracts all four sources, publishes
graph facts, binds one shared hybrid/graph publication, validates actual host C1/C2
membership, and executes three baselines plus local/community/global recipes.
Preparation receipts and two membership graph calls are separate from query budgets.
Each provider attempt owns a new client/HTTP tracker. Actual HTTP RoundTrip counts
and logical core callback counts have independent fields. A callback can fail before
HTTP dispatch; excess observed HTTP counts fail budget certification. Live-labelled
preparation/summary rows must include observed transport markers. Exact extraction
configuration, prompt/schema hashes, reservations and overall capture deadline are
part of configuration identity.

`-capture` requires OPENAI_API_KEY, explicit model and absolute executable qualified
host tokenizer configuration. It executes the full path and writes raw observations;
`-input` scores them independently. Preflight rejects missing credentials or invalid
host executable selection before artifact/publication. Failed extraction retains
actual partial preparation with a nonzero exit, and the evaluator refuses to certify
that incomplete capture. Consumer-owned temporary storage is cleaned after the run.
The exact tokenizer implementation is shared by text/graph examples in consumer-only
internal/modelcounter; root ragy gains no dependency. Existing actual executable
text counter tests continue to pass with cancellation, size, unknown receipt,
identity, environment and sanitized-error checks.

The combined contract run uses actual durable storage, source readers, core recipes,
eight HTTP requests and an actual trusted local executable emitting a fixed receipt.
It is explicitly not a qualified tokenizer or a live model. Its retained artifacts
are `results/graph-contract-capture.json` and `results/graph-contract-report.json`;
creation is verified by `results/graph-contract-artifact-test.txt`. The artifact has
four extraction receipts, two preparation graph lookups and six query observations.
All contract support Recall@3 values are 1, gain is 0, budgets pass and default stays
hybrid. These numbers validate execution/scoring mechanics, not model quality.
The evaluator has missing/duplicate/foreign/failed preparation and unknown/over-limit
accounting regressions, including inconsistent HTTP/model invocation counts.

The actual live CLI was executed without available credentials and exited before
artifact creation: `results/graph-live-preflight.txt`. No live capture was produced.
Actual tokenizer/model qualification, live text/graph comparative acceptance,
remaining graph provenance/evidence integration reconciliation and the final two
independent audits remain outstanding. No completion/release/issue close is claimed.

Final configured `make test` passed across every module/race/example/actual PDF
profile on the completed capture and shared-tokenizer tree, including the saved
artifact path, full HTTP reference gold traversal, transport contradiction gate,
preflight and partial failure retention. Evidence:
`results/graph-live-capture-full-test.txt`. Full configured lint initially flagged
G703 on the capture writer's caller-selected output path. That exact statement
now documents the trusted CLI/test destination boundary: source/model/captured
references cannot select or alter the path. The only follow-up source change was
that narrow explanatory lint annotation. Final `make lint` passes:
`results/graph-live-capture-full-lint.txt`. No global lint rule was disabled.
Independent coverage/lifecycle/evidence schema suites and production-doc language
contract also pass on this checkpoint. The goal remains active with live model/
tokenizer and final requirement reconciliation/audits outstanding.

## Model configuration bound to graph publication and decision history

Current source review found that core materialization already fingerprints ontology
and policy but the consumer supplied a fixed model-extraction transformation. The
consumer now supplies the actual extraction configuration fingerprint. Provider
model/endpoint, request template/schema, token/cost/byte bounds and tokenizer identity
partition it. Source batches require a valid fingerprint before target construction.
Preparation receipts and summary observations retain their bound configuration.
The model metadata must match the actual provider binding before dispatch. Contract
capture model identity was corrected to the actual contract-model request name;
the explicit contract-fixture execution label continues to distinguish scripted
responses from live quality evidence.

The consumer now captures combined extraction inputs and typed decisions through
immutable history, syncs the FileStore append and reopens/reads the archive before
publication. Its metadata fingerprints all per-source configurations. Capture
retains the resulting history ID. Model configuration changes produce different
publication targets/history IDs while exact gold facts and original supports stay
unchanged. The same reopened archive retains both configurations; a recomputation
snapshot explicitly references the prior snapshot as Parent. There is no implicit
latest selection or automatic history overwrite.

Tests: TestGraphModelConfigurationChangesPublicationAndRetainsHistory;
TestProviderConfigurationIdentityBindsModelTokenizerAndRequestControls;
TestCaptureRejectsDeclaredModelMismatchBeforeDispatch;
TestSummaryDeclaredModelMismatchStopsBeforeDispatch; invalid graph source
configuration before target construction; complete HTTP capture/reference gold
regressions. GOWORK=off package race/lint pass in results/graph-provenance-test.txt.
The regenerated contract capture/report include per-source configuration, actual
model name, resolution history ID and summary request fingerprints.

During this contract extension two formatted test fixtures retained missing config
fields/old descriptive model labels, and the first predecessor edit did not populate
Metadata.Parent. The failures were reproduced by the package suite; all affected
fixtures and the actual archive metadata assignment were corrected. Final race
checks include the explicit predecessor assertion and exact model mismatch gates.
No compatibility fallback was introduced for missing preparation fingerprints.
Live model/tokenizer acceptance, remaining actual evidence integration/reconciliation
and two final independent audits remain outstanding.

Final configured make lint and make test passed on the complete configuration/history
state across all modules/race/examples/actual PDF. Evidence:
results/graph-provenance-full-lint.txt and results/graph-provenance-full-test.txt.
The final offline scorer successfully regenerated graph-contract-report.json from
the new capture. Production docs language and independent coverage/lifecycle/evidence
schema checks also passed. Goal remains active; this is a verified checkpoint,
not an assertion of all required experiments or final audit completion.

## Actual graph evidence recording and source-gated export

The external graph consumer connects delivered source supports to the existing
immutable evidence.Capture/evidence.Run contracts using typed scalar metadata,
mandatory schema admission and current exact source Reader lookup. Its policy
allowlists declared identifiers/numbers, omits raw query/source text, raw auth/
metadata/access fingerprint and sink errors, reports scores absent and judgments
ungradable. Internal hit observations are explicitly missing_observation; this is
support-list evidence, not full internal graph/model stage telemetry.

Full capture wraps each real hybrid/local/community/global attempt in required
recording and embeds the immutable wire record. A single parent five-second context
covers operation, export/source admission and recording; final raw timing includes
recording. Export Reader counters are independent per attempt and included in raw
source I/O. Model callbacks are never repeated because of sink failure. Protection
failure suppresses the entire result/receipt, including export counters.

Actual scoped local integration verifies disabled/best-effort/required failed sink
modes: one backend execution/graph call, zero sink calls when disabled, one sink
call and separate receipt failure when enabled, ordinary required recording error
with the completed retrieval fact, immutable source associations and text/auth/error
redaction. Source retirement before export (both s1 and s2) prevents sink dispatch
and suppresses result. Missing internal stage sources reject a consumer's required
all-stage SourceField. Unknown/large numeric diagnostics and uint64 call-sum overflow
remain unavailable, not rounded/zeroed observations. Tests:
TestActualGraphEvidenceRecordingModesExecuteOnceAndKeepPrivacy;
TestGraphEvidenceSourceRetirementBeforeExportSuppressesRecord;
TestGraphSupportEvidenceCaptureStandalone;
TestGraphRequiredAllStageSourcesRejectsMissingInternalObservations;
TestGraphDiagnosticNumbersDoNotRoundExactOverflowOrUnknownToZero.

Initial wiring incorrectly required SourceField across missing internal stages;
Capture correctly returned unavailable. The consumer now requires scope/publication
and explicitly tests full-stage SourceField refusal. The delivered support list
remains source-admitted and fully associated. The input contract was not weakened
to label missing stages observed. A mismatched WireHit label field in the new test
was corrected to Judgment. No source-error result or hidden count survives protection.
GOWORK=off full package race/lint pass: results/graph-evidence-test.txt. The regenerated
contract capture has a strict evidence record on all six observations. Live model/
tokenizer acceptance, full internal stage integration/reconciliation and final two
independent audits remain outstanding; no completion/release/issue close claimed.

### Graph evidence association validation

The preceding full `make lint` and `make test` executions completed with exit 0;
logs: `results/graph-evidence-full-lint.txt`, `results/graph-evidence-full-test.txt`.
The actual contract capture also passed offline evaluation with its six embedded
records. Evidence, coverage and lifecycle schema checkers passed 16/32, 7/12 and
4/13 positive/negative fixtures respectively; production documentation check passed.

The offline graph evaluator previously accepted arbitrary embedded evidence JSON.
It now uses `evidence.Decode` and checks retrieval ID, scope/publication, recipe
configuration, outcome/reason and each delivered ordered exact original source tuple.
A real local graph capture verifies acceptance; eleven negative cases alter scope,
publication, recipe, retrieval ID, outcome, source ID/revision, ordering, support-stage
presence/uniqueness or JSON shape. Missing optional evidence is not synthesized and
is not proof of tracing; internal stage observations and live experiments remain
pending. Association validation does not authenticate an external producer.

Focused verification completed: graph consumer race suite passed (`results/graph-evidence-association-test.txt`), final lint passed with zero issues (`results/graph-evidence-association-lint.txt`). The final helper refactor retained all eleven rejection cases in the targeted actual-capture test. Offline evaluation of the saved six-record actual contract capture passed with strict evidence association validation. These fixtures are protocol/contract evidence, not live quality acceptance.

### Actual hybrid stage evidence in graph comparison

The graph consumer baseline now projects the actual dense, lexical and RRF
ResultSets into owned stage observations before building its delivered source list.
Document IDs, numeric scores, semantics and ranks remain adapter/fusion values;
source associations explicitly project original mappings rather than treating an
indexed vector artifact as original text. All original references pass current
scoped source admission. Unfinished stages remain `missing_observation` individually.
No schema, model-provider or recorder dependency was added to library core.

`TestHybridEvidenceObservesActualLeafAndFusionScores` captures the actual published
baseline with required all-stage source fields, compares actual score/rank semantics,
checks foreign-source exclusion and privacy, and verifies immutable record ownership.
The complete graph consumer race suite passed (`results/graph-hybrid-stage-test.txt`);
final lint has zero issues (`results/graph-hybrid-stage-lint.txt`). Saved actual
contract capture has dense/lexical/RRF hit counts 3/3/4, 3/2/4 and 3/3/4 for the three
queries. Final delivered supports remain TopK 3; fusion evidence includes the fourth
real fused hit. Local graph and summary map/reduce individual observations, plus
live quality experiments, remain pending.

After the final per-stage missing-observation merge and test helper refactor,
the targeted actual baseline/complete-capture race checks passed
(`results/graph-hybrid-stage-final-test.txt`). Offline evaluation of the refreshed
actual contract capture also passed strict association validation.

### Graph traversal and map/reduce artifact evidence

Local expansion now snapshots the actual managed nodes/edges and each exact support
association into graph evidence. Fact IDs remain actual graph IDs; ranks are
unavailable and scores absent. Community and global captures resolve each immutable
community/map/reduce summary independently and observe the selected original source
associations. Generated prose is not exported, and neither support selection nor
snapshot iteration is claimed to be a similarity ranking or graded answer.

`TestGraphRecipeStagesCaptureUnrankedFactsAndSummaryAssociations` covers all three
real producer paths with required source fields for every completed stage, explicit
rank/score/judgment absence, privacy and foreign-source exclusion. Existing explicit
missing-observation test now removes producer observations deliberately; its required
capture remains unavailable. Full graph race suite passed (`results/graph-recipe-stages-test.txt`).
The contract capture was refreshed through the actual producer orchestration.

Final focused race checks passed (`results/graph-recipe-stages-final-test.txt`),
including the negative global reduce that selects only one required community:
its actual three dispatched model calls retain known usage, but no summary/support
or successful stage artifact is delivered. Final focused lint has zero issues
(`results/graph-recipe-stages-lint.txt`). The earlier speculative test of a successful
reduce omitting C2 was rejected by the existing full-community protocol and replaced
with this explicit negative acceptance test; no core contract was relaxed.

The refreshed actual contract artifact observes local graph 5 facts/2 original
supports; community 2 selected sources; global map C1=2, C2=1, reduce=3 selected
source associations. Its six nested records pass offline strict association
validation. These are contract captures with scripted model outputs, not live quality
results. Broad all-module checks are recorded separately when terminal results arrive.

All six actual captured nested records also passed the independent executable
`evidence.schema.json` Draft202012Validator, in addition to core Decode/association
validation; the standalone schema fixture checker passed 16 positive/32 negative.

Broad final checkpoint: `make lint` and `make test` both completed with exit 0,
including all root/adapter/example modules, race checks and actual PDF engine.
Logs: `results/graph-recipe-stages-full-lint.txt`,
`results/graph-recipe-stages-full-test.txt`. This checkpoint confirms current checks,
not live experiment acceptance or overall completion.

### Actual PDF layout through durable lifecycle and retained publication

`TestActualPDFLayoutDurablePublicationReopenAndRetainedRevision` runs the real PDF
engine twice for r1/r2, separately applies an explicit unreadable OCR simulation,
projects original page/cell and derived image text, and stages/publishes those actual
records through Executor/filestore CAS into persistent dense storage. Both adapter
and ledger are reopened. The old publication reads r1 while r2 is active: original
mapping/geometry/source revision, derived image origin and typed partial coverage
survive the saved catalog/payload path. Exact original locator resolution uses a
host retention implementation with both revisions; deleting or denying r1 fails
before payload Load and never substitutes r2.

Coverage is host-owned parser metadata in the indexed payload, covered by the
manifest payload fingerprint. The specification requires preserved coverage and
versioned locator envelopes, not PDF-specific fields in generic lifecycle.Manifest;
no parser dependency or invented coverage promotion was added to lifecycle core.
Original blob retention remains a host responsibility: the retained layout host in
this integration is in memory, whereas manifest/index payloads use actual durable
files. Existing parser/resolver tests independently cover typed diagnostics,
coordinate transforms and locator negatives; this integration does not claim OCR
accuracy, durable original blobs or a complete cross-capability audit.

Full PDF adapter race suite passed (`results/pdf-durable-lifecycle-test.txt`). Final
OCR/ledger-reopen/deleted-and-denied scenario verification is recorded separately in
`results/pdf-durable-lifecycle-final-test.txt`; focused lint log is
`results/pdf-durable-lifecycle-lint.txt`.

The final targeted race execution completed successfully; focused lint completed
with zero issues, and production documentation checks passed. No release or overall
completion is claimed by this integration checkpoint.

### Standalone locator envelope and typed host extension

Implemented strict `source.EncodeLocator` / `DecodeLocator` and executable
`schemas/locator.schema.json`, sharing the six declared union shapes with the
existing evidence contract while retaining actual source.Reference identities.
Independent validator passed 6 positive/22 negative cases. Go tests cover six-kind
round-trip/citation identity and duplicate/missing/unknown/case-alias/null/trailing/
invalid UTF-8/incompatible schema rejection. Typed host extension JSON round-trip
and annotation changes leave the canonical location identity unchanged.

Actual PDF durable r1/r2 publication/reopen test now passes every retrieved original
support locator through this codec and verifies equal identity before exact retained
resolution. Source race suite passed (`results/locator-envelope-test.txt`); actual
PDF targeted race integration passed (`results/locator-envelope-pdf-test.txt`).
Host extension validation/cloning remains host-owned; this codec checks geometry
and representation shape without granting authorization or original-blob retention.

Final locator source race checks passed (`results/locator-envelope-final-test.txt`).
During this checkpoint the host default toolchain is now Go 1.27.1 with golangci-lint
2.14.0; earlier successful broad checks used the preceding host setup. The new lint
found existing formatting/modernize/unused-directive issues. Applied its checked
AsType/format/directive rewrites in root, observability adapter tests and planner
example tests. `errors.AsType` exists in the declared Go baseline's API manifest;
focused access/retrieval error tests also pass on local Go 1.26.5
(`results/toolchain-baseline-errors-test.txt`). Module go directives were not changed,
and no lint rule was globally disabled. Updated broad lint/test executions are
recorded in `results/locator-envelope-full-lint.txt` and
`results/locator-envelope-full-test.txt`; their terminal outcomes must be checked
before this checkpoint is used as all-module evidence.

Updated broad `make lint` and `make test` both completed with exit 0 on the current
host toolchain, including every module, race checks, actual PDF and examples.
Logs: `results/locator-envelope-full-lint.txt`,
`results/locator-envelope-full-test.txt`. The subsequent non-semantic formatter
correction in the contract summary fixture was included in the final lint; its
protocol/usage behavior is unchanged. There are no disabled lint checks or module
Go baseline changes. Live experiments and final independent audits remain pending.

## CLI consumer calibration after clarified acceptance

The author's issuecomment-6011559475 permits a separate calibrated advisory-token CLI host profile while preserving all reference hard-budget fixtures. Task12§7.1 and R-19 were synchronized with that explicit decision; the195-atom scope is unchanged.

Consumer-only internal/codexcall and typed planner/assessor ports now execute actual structured CLI responses. Read bindings are checked before and after dispatch. Assessment exposes admitted snippet text/query ordinals only; qrels/expected IDs are not part of model inputs. Each call owns a temporary model working directory, bounded input/schema/trace buffers and a45-second subprocess deadline. Known tool events invalidate the call; absent or malformed input/output usage does not become zero. Exact generation token bounds and underlying provider dispatch/retry count remain unverified; no monetary cost is inferred from token usage. This consumer work does not change core interfaces or reference profile behavior.

Current affected external-module lint terminated exit0 (`results/codex-consumer-lint.txt`); `go test -race ./internal/codexcall ./recipe_comparison -count=1` terminated exit0 (`results/codex-consumer-test.txt`). This includes existing reference HTTP/budget/receipt tests after replacing concrete consumer model fields with typed ports.

Actual calibration command used the existing ChatGPT CLI authorization and the configured model. `results/codex-text-calibration-process.txt` process terminated exit0; `results/codex-text-calibration.json` retains exact port inputs/instructions/schema, output, JSON events and reported full usage. Both calls were successful with no observed tool activity: planning7.323762083s/5327input/23output; assessment8.513481583s/5354input/19output. Actual scoped readonly BM25 ran between those ports. This is calibration-only, not a complete20-row text grid or graph experiment. Existing95.90% is a historical independent checkpoint; new experimental requirements remain uncredited until full runs and both audits.

Next required work: fix the host profile before the comparative run, execute all text profiles/cases through the actual recipes, implement equivalent graph model bindings and preparation/query accounting, execute graph comparisons, save the source-grounded consumer summary rubric and re-run final gates/audits. Do not raise per-case limits retrospectively or substitute calibration responses for live port calls.

### Actual CLI text comparison (2026-10-06)

Following issue #3 comment 6011559475 and task12 §7.1, the calibrated, frozen CLI host profile completed its entire 20-row grid. Artifacts: `results/text-cli-live-capture.json.profile.json`, `results/text-cli-live-capture.json`, `results/text-cli-live-report.json`; process and offline evaluator both exited 0. Thirty actual CLI model invocations reported 160992 input and 1075 output tokens, including executor context. Raw request/instructions/schema/output/events/usage and end-to-end timings are retained per call. No observed tool activity; extra context and internal provider retries remain unverified.

| Profile | Recall@3 | MRR@3 | No-answer errors | Failed executions |
|---|---:|---:|---:|---:|
| Baseline | 0.5 | 0.5 | 0 | 0 |
| Single rewrite | 0.5 | 0.5 | 0 | 0 |
| Multi-query | 0.5 | 0.5 | 0 | 0 |
| Decomposition | 1.0 | 0.875 | 0 | 0 |

All actual scope/publication/source references survive offline validation. This is a five-query consumer experiment, not an isolated retrieval comparison or a production quality claim. No percentiles/significance are inferred. Provider monetary price is unavailable; zero unpriced ledger units do not mean zero billing. Token bounds are advisory; recipes do not qualify the unchanged reference budget profile. Default remains baseline. This checkpoint does not establish final 100% acceptance: graph live comparison, external summary review and final independent audits remain outstanding.

### Actual CLI graph comparison (2026-10-06)

Calibration completed with two actual valid model responses, then `graph-cli-live-capture.json.profile.json` was frozen. The complete live capture and offline evaluator exited0. Four source extraction calls passed typed ontology/identity/support validation and were materialized through actual persistent dense/managed graph producers. All six baseline-vs-recipe observations and immutable evidence records survive validation. `graph-cli-live-capture.json` retains actual input/output/raw events/full usage and end-to-end timings.

Preparation:4 model calls,22322input/642output tokens,41.254875543seconds summed source-attempt time; membership graph calls are reported separately. Query-time:4 model calls,21606input/271output tokens, separate from extraction. Unknown monetary prices are explicit; no precomputation is counted as free.

Support Recall@3 baseline→recipe: local1→1; community1→0; global1→2/3. Community summary returned abstention with no selected snippets, rejected by the actual core validation and captured as failed. Global summary covered both configured prod-community dependency facts using original s1/s4 supports but omitted the redundant s2 qrel support. Negative gains are retained, with baseline as default. No per-case retry or post-run limit adjustment was performed by the consumer.

External consumer rubric is preserved in `results/graph-cli-summary-review.md`: every checkable claim is compared to actual inputs/source refs, with omissions, namespace ambiguity, local unknowns and abstention assessed. Independent auditors must verify it before final acceptance. This run is not an isolated retrieval comparison and does not prove original reference budgets, hard CLI input/output bounds, hidden provider dispatch count or general summary quality.

### Final source verification

`make lint` and `make test` after all trace/accounting/Unicode source corrections terminated exit0. Independent final correctness overlay8adversarialcases and76actual traces across both separate runs passed on the final decoder. V2text20rows/30calls and v2graph4extractions/6rows, frozen calibrated profiles, raw stdout/stderr/events/settlements and external summary review are independently examined. No paid model rerun is credited for scripted tests; the final failure-path corrections are independently exercised with adversarial traces. Final completeness is the independent195/195=100% verdict; exact195unique ID inventory has no missing/extra requirements. No release/publication/issue closure performed.
