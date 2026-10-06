# Community and global summary artifacts

This optional recipe accepts host-declared community membership, admitted original
source mappings and BYOT access metadata. `Community` uses one bounded map call.
`Global` accepts one through `MaxCommunities` communities, uses one map per
community and one non-recursive reduce, including for a single community. Required
map/reduce calls must fit `MaxModelCalls` before any dispatch. `MaxSnippets` is a
per-community host limit with no built-in twenty-snippet cap. `MaxSupports` and
source text plus question bytes are aggregate admission limits. `MaxInputBytes`
also bounds each complete JSON `ModelInput`, including escaping, labels, question
and reserved token limits, before reservation/dispatch. Reducer overflow returns
budget-exhausted with admitted partial maps. Members, generated text and duration
have explicit limits. Configure the reference ledger with 4096 input tokens, 1024 output
tokens, three model calls and a 5-second deadline. The ledger is shared across stages.
The host may also share this ledger across concurrent text/graph recipes within an
attempt. The effective context deadline is the earliest of parent, recipe duration
and shared ledger remaining time. Reservations remain atomic: a variant refused the last available model
call returns budget-exhausted without dispatch or retry. Host callbacks must be
concurrency-safe; caller request data must remain stable during capture.

The host owns canonical members, membership validation, metadata schema/cloning,
retained-source admission, pricing, exact request token counting and model transport.
The recipe does not detect communities, resolve identities, manage conversation or
generate the application's final answer. Model input contains question, stage and
ordinal/text snippets with reserved limits. It contains no binding, source refs,
access metadata, canonical community/member IDs or credentials. Model output contains
bounded UTF-8 text and selected input ordinals; foreign/duplicate/empty selections
fail. The injected client must enforce reserved token limits and execute once.

All input shapes, source revisions and metadata are admitted before source/model
ports. Source locators must belong to the pinned read's namespace/source/revision/
access inventory. The host source port verifies actual original quote/representation
and retention permissions. Fresh source checks surround stages, follow pricing/token
counting, precede dispatch and delivery, and guard reduction of existing summaries.
Parent cancellation, revocation or deadline suppress output. Pure bounded host ports
must not perform hidden model calls/retries; arbitrary Go callbacks are not sandboxed.

Map selections retain only their original supports. Declared snippet-member coverage
must cover the community for complete output; missing coverage is insufficient and
does not proceed to reduce. `CoversMembership` describes this mechanical declaration,
not whether model prose is semantically correct. Global reduce must select every map
summary and retains support from all admitted communities. Natural-language quality and
entailment remain external evaluation responsibilities.

`Summary` keeps generated text, membership/support snapshot and binding identity
private. `Resolve` rechecks the exact scope predicate/snapshot/publication and all
retained supports before exposing a support-only derived `MappedText`. It never
claims an exact original quotation. A changed binding/publication, expired/revoked
authorization or unavailable/deleted support cannot silently reuse an old artifact.
Historical source access remains an explicit host retention decision under the
matching pinned binding. Accessors return independent support/community slices.

Quotes reserve a model call and input/output/cost before dispatch. Known usage settles
even when a call fails. Unknown accounting retains reservations; known token overrun
is rejected even with advisory unknown price. Budget/required-price refusal yields
insufficient, or partial with completed community artifacts and no fabricated global
summary. Retained partial artifacts are revalidated before delivery. There is no retry,
recursive planning or hidden fallback. Counter/model input slices are independent.

Load original inputs through the scoped `source.Reader` before constructing snippets.
Original source reader targets must be explicitly pinned with their own exact
transformation inventory; a graph index transformation is not an original source
transformation. The same binding may contain both target inventories. The optional
structured HTTP `Summarizer` exposes matching model/counting ports; configure its
schema, validator, tokenizer, instructions/model and pricing explicitly.

Tests cover both bounded recipes, complete/partial/insufficient outcomes, original
support retention, derived precision, snapshot ownership, source-reader batch
admission, model privacy, invalid selections, usage overruns, deletion/revocation/
expiry and publication invalidation. HTTP protocol fixtures and scripted prose do
not establish live model quality or exact provider token counts.

A caller `observation.Session` observes the attempt, each summary map/reduce stage
and each actual model dispatch. Numeric community ordinals identify map branches;
no question, summary text, community ID or source support enters diagnostics.
Token usage is observed only from the dispatched model port; unknown usage and
local cancellation do not imply zero billing or confirmed remote cancellation.
Diagnostic exporter failure never retries a summary or changes its result.

Model, settlement and freshness errors are preserved together, including simultaneous usage overrun and expiry. Only direct pre-dispatch budget/price admission failures can produce bounded partial summaries; joined callback failures are not rescued by matching a budget sentinel. Protection always suppresses payload.

## Selected citations and future reads

The model sees every admitted snippet; Selected specifies user-facing citations,
not all information that influenced its text. Supports retains only their union.
Summary.Resolve verifies the original binding and current access to these retained
supports. Revocation of an unselected input after successful construction, while
the original binding remains valid, does not by itself revoke that Summary.
This selected-citation policy is structural association, not information-flow
taint, semantic truth or proof that other model inputs had no influence.

A product requiring full derivation revocation must separately retain the complete
map input inventory and reduction ancestry and authorize it externally before
Resolve. Summary has no complete-dependency inventory; callers cannot recover it
from Supports or Selected alone. If scope/publication freshness changes or a
selected support is denied/deleted, Resolve fails closed. It grants no new binding.

Summary hides its text until a fresh Resolve. The returned source.MappedText is
owned admitted data: the host can retain its bytes after that delivery, just as
with retrieval results. Future checks cannot claw back already delivered strings.
Resolve returns derived support-only text, never an exact original quotation.

## Coverage and cost

CoversMembership checks declared member sets of selected snippets, not generated
prose quality. MissingCoverage stops at the first incomplete community and skips
later maps and reduce: reduction cannot claim complete declared membership when
an input community is incomplete. Host may evaluate quality or explicitly start
a new bounded attempt; core never silently repairs or retries.

Stable-order support union/member subset scans can be quadratic within admitted
counts. Finite capacities do not imply linear CPU, byte or peak-memory bounds.
[Reference scaling measurements](../../graphingest/resolution/scaling.md) retain
current algorithms and record actual workloads without a speedup claim.

[Source/lifecycle integration](lifecycle_integration_unix_test.go) exercises retained
source evidence, fresh Resolve and explicit pin/publication behavior.

Complete input evidence is refreshed before/after each map and reduce, including
immediately before reservation/dispatch. With C calls and S admitted support
occurrences, repeated freshness checks cost O(C×S) callback work plus final selected
Resolve checks. A positive observation is not an authorization lease; the host may
implement a cheap captured-policy check only with an explicit freshness contract.
Core does not memoize permission.
