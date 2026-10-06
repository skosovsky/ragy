# Bounded text retrieval recipes

`New(Config).Run(ctx, request, ledger)` provides three explicit opt-in strategies:

| Strategy | Execution |
|---|---|
| SingleRewrite | Original retrieval, one text planning call, at most one rewrite retrieval, one assessment call |
| MultiQuery | Original retrieval, one planning call, at most two variant retrievals, one assessment call |
| Decomposition | One planning call, at most three independent subquestion retrievals, one assessment call; depth is always one |

There is no default recipe, recursive planning, hidden retry, autonomous action or
final answer generation. Queries execute sequentially and reserve an explicitly supplied atomic ledger.
Concurrent Run calls can share that ledger; RunOwn explicitly creates a private
ledger. Neither path enforces an organization or conversation quota. Host ports
must support concurrent calls, and caller inputs must remain stable during capture.
All limits, duration, recipe identity, RRF configuration and
maximum result sizes are explicit host configuration. Reference fixtures set
2/3 retrieval calls, at most 2 model calls, 2048/512 input/output tokens, duration
5 seconds and cost cap 100; these are not universal production defaults.

## Ports and ownership

The typed backend retains the original intent and request metadata. Planner output
contains text only; it cannot replace mandatory filters, publication, host intent
or policies. Existing plan filters/ranges remain attached when text changes.
Precomputed `Options.Vector` and graph seeds are rejected before dispatch: text
variants cannot silently reuse an embedding or traversal for the original query.
This declared profile requires model-free backend/admission/pricing ports. Model
work runs only through reserved planner/assessor and optional QueryEncoder ports.
QueryEncoder separately admits strict remote bounds and receives each rewritten
text with its actual reservation. Passing an arbitrary backend does not sandbox its implementation.

Admission must negotiate every reachable target and original/preplanned predicate
before payload/model I/O. It must not mutate the request or perform payload work.
Each executed variant is admitted again; this declared profile requires stable
coverage across variants and retains explicit partial admission in the result.
Unsupported admission fails closed and cannot become an unrestricted retry.

CloneIntent, CloneRequestMeta and CloneMeta capture BYOT ownership. Inputs must not
be concurrently modified during capture; host callbacks must be concurrency-safe.
Planner and assessor receive separate owned snapshots. Result query observations,
selected metadata and backend storage do not alias one another. Result owns its
public slices; do not mutate them concurrently with use or export. Immutable wire
record export is a separate required capability, not implied by this envelope.

Supports must authorize and resolve every original locator against the supplied
binding before returning it. Recipe validates locator shape and pinned namespace,
source, revision and access identity. Publication membership alone does not grant
authorization for another fragment of the same source. Original representation
transformations may differ from index transformations. Empty, foreign or newer
revision supports fail; source retention/catalog validation belongs to the host
support resolver. Identity defines deduplication explicitly, independent of scores.

## Budgets, outcomes and failures

Pricing is a pure host quote port; it must not invoke a model. Model adapters must
enforce declared maximum tokens and report actual usage even on errors when known.
Every recipe-owned retrieval/planner/assessor dispatch is reserved atomically.
Retrieval has no model token usage. A nonzero retrieval cost quote is conservatively
retained as a reservation because the generic backend does not report actual cost;
it is not mislabeled as observed actual cost. Unknown usage keeps its reservation.
Unknown required price stops before dispatch. Advisory unknown model pricing
conservatively marks full usage unknown and retains token reservations, even when
the model reports token counts; it does not claim a known actual cost. Advisory
unknown price remains
explicitly diagnostic and cannot claim cost compliance for that call.

The earlier parent deadline propagates to all dispatch ports. A real attempt timer
and injected budget clock both stop further callbacks/dispatch. Protection failure
or parent cancellation suppresses every payload and side output. Budget exhaustion
or an attempt-local deadline produces bounded partial/insufficient evidence already
captured; final RRF assembly uses captured identities and metadata without calling
host projectors after the attempt deadline. Actual usage overruns and malformed
outputs are errors, not successful partial attempts.

Assessment selects executed query indices and signals sufficiency. This is a
strategy signal, not proof of truth. Selecting original index 0 rejects a worsening
rewrite. With no selected evidence, outcome is insufficient; missing decomposition
parts or explicit partial admission prevent complete. When stopped before assessment,
available evidence is explicitly unassessed partial/insufficient. No caller should
interpret partial as a complete answer.

Existing RRF fuses selected sets with declared score semantics. Contributor records
retain each query index, original document ID/rank and original source supports,
including dedup contributions. Query observations retain native scores/history.

Malformed/oversized planning or results, invalid selection, source support failure,
callback errors and budget overrun return no payload from `Recipe.Run`. Errors preserve context and
protection classification; there is no rescue which widens access. Optional strategy
quality recommendations still require the separate actual-adapter experiment.

## Optional recording

`recipe/recording.Run` connects one recipe attempt to an `evidence.Sink` under
explicit disabled, best-effort or required policy. Provide a metadata codec/schema,
metadata clone port and source admission port for scoped export. Source admission
must authorize the original reference against the captured binding and host
catalog; publication membership alone does not authorize individual fragments.
Required unsupported judgment export and denied query/snippet policies fail before
recipe execution. The wrapper does not supply a grader.

Native observations are exported as `retrieve/N` stages in execution order;
`fusion` contains the selected RRF observations and the union of original source
references from all contributors. Planner/assessor dispatch events contain empty
observed document sets; unexecuted model stages are `not_run`. Dispatched retrieval
without retained observations is `missing_observation`, never fabricated empty
success. Explicit `Recipe.RunObserved` retains owned prior query observations and
settled usage after ordinary errors, with outcome `failed`, stop `stage-failure`
and no selected hits. `Recipe.Run` suppresses error payloads. Enabled recording
uses RunObserved, so failed planner/assessor calls no longer erase earlier hits.
Unstarted fusion is `not_run`; dispatched fusion with no retained result is
`missing_observation`; completed fusion is `observed`. Protection failure and
parent cancellation suppress the complete journal and receipt. Exact query
contributions remain available in typed results and export under explicit
evidence.Policy.AllowContribution. Original query/dedup support locations require
Policy.AllowLocation. Recording verifies query/document/list-position/support tuples
and preserves both links after dedup.

Recording cannot repeat retrieval/model calls. Required sink failure returns the
completed result and immutable record with a recording error; callers must not
claim overall success. Protection failure suppresses both result and receipt.
Default policy excludes query text, snippets, arbitrary metadata and error text.
Unknown accounting and integers outside the exact JSON-number range are
unavailable, not rounded or replaced by zero. `recipe.SnapshotResult` owns mutable
slices and metadata through the host clone port and suppresses its envelope if the
final freshness gate fails. Do not mutate input results concurrently with export.


TASK-16 execution contract: `Run` and `RunObserved` require a caller ledger.
`RunOwn` and `RunOwnObserved` explicitly create an independent ledger from Config.
Recording accepts optional Config.Ledger; omission uses the named own-ledger path.
The effective callback deadline is the minimum parent, recipe duration, and shared
ledger remaining duration. Injected clocks govern admission; real context timers
also stop cooperative callbacks. Calls are never refunded.

Config.BackendModelFree must explicitly attest that backend retrieval introduces
no hidden model operations. Generic backend interfaces cannot prove this property.
Optional Config.QueryEncoder has pure Admit and one Encode dispatch, independently
priced/reserved with ModelLimits and dense Purpose=Query. Every text variant gets
its own encoding. `UnsupportedQueryEncoderBridge` rejects supplied adapters without
hard remote token limits before provider I/O. Hosts may provide capable encoders;
unknown observed usage retains the complete reservation and result retains Space.

Config.Artifact optionally renders the fused selection using explicit full-output
resource policy. Queries retain retrieved evidence; Selected retains post-TopK
contributors; Coverage separately records selected and delivered evidence per
query. All planned decomposition parts must survive packing before Complete.
Partial/derived/unattributable snippets cannot assert full delivery. Without artifact
rendering, the full selected documents are the delivered payload and no tokenizer
is required. Recording exports fusion and optional delivery as separate stages,
and reports this run's own stages rather than other callers' shared ledger usage.

Packed snippet contributors address the original selected document ordinal. Document
IDs are display/storage identities and cannot prove which independently keyed
fragment survived packing. Equal packed text can retain several input contributors
with individual full/uncertain flags; discarded different text contributes nothing.
Recording and snapshots preserve those actual input contributors.

Optional diagnostics use a bounded `observation.Session` on the caller context.
Recipe, encoding, model, fusion and requested artifact delivery report actual local
starts and completions. Query ordinals match `Result.Queries` indices; events
contain no query text, document IDs, metadata, provider messages or host revision
labels. Model token counts come from the dispatched port; host-priced cost is not
reported as provider billed units. Unknown accounting remains unknown. A local
cancellation does not confirm remote cancellation or billing.

Diagnostic callback errors and panics cannot change or repeat an attempt. Required
recording remains a separate contract. `Result.Sufficiency` is unavailable until a
validated assessor response supplies the signal, and is not a truth guarantee.
`Stage.Completed` distinguishes validated retained output from a dispatched call
whose output is missing or rejected; actual usage may still be known on failure.

`Result.ArtifactRequested` retains the actual configured delivery mode. If false,
selected documents are delivered directly. If true with a nil artifact, rendering
was requested but its output was not retained; export must keep delivery uncertain.
This fact is independent of document IDs and query coverage and survives snapshots.

Local deadline never erases callback or settlement failures. Joined protection, protocol, ordinary callback failure or usage overrun remains failure, even when it also matches DeadlineExceeded. Only a proven attempt-local timer/clock stop without independent failure can return bounded evidence. Parent/read failure suppresses the complete journal; all already observed causes remain inspectable through errors.Is/As. Callback output shape is checked before timing can hide protocol failures. Known overrun stays visible in the failed stage observation while the ledger conservatively retains its reservation.
