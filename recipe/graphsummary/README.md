# Community and global summary artifacts

This optional recipe accepts host-declared community membership, admitted original
source mappings and BYOT access metadata. `Community` uses one bounded map call.
`Global` accepts exactly two communities, uses one map per community and one
non-recursive reduce: at most three model calls. Each community has at most twenty
source snippets; bytes, members, supports, generated text and attempt duration have
explicit limits. Configure the reference ledger with 4096 input tokens, 1024 output
tokens, three model calls and a 5-second deadline. The ledger is shared across stages.
The host may also share this ledger across concurrent summary variants within an
attempt. Reservations remain atomic: a variant refused the last available model
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
not whether model prose is semantically correct. Global reduce must select both map
summaries and retains support from both communities. Natural-language quality and
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
