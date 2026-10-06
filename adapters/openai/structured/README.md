# Bounded structured completions

This optional adapter uses one Chat Completions POST with a host-selected model,
explicit `max_completion_tokens`, non-streaming strict `json_schema` response
format and `store: false`. It follows the provider's documented
[structured output protocol](https://developers.openai.com/api/docs/guides/structured-outputs).
Refusal and incomplete output are separate errors; available usage survives both.
No model, domain ontology, prompt policy, prices or tokenizer is chosen implicitly.

`Config` requires request/response byte limits, a positive attempt duration, exact
host token counting and an executable domain schema validator. `Schema` is copied
at construction and sent to the provider. The host validator must implement that
same immutable schema (including required fields, nullability and BYOT attribute
constraints), rather than merely check whether the output is valid JSON. Schema
support depends on the explicitly configured model. Constructor validation checks
schema syntax/object shape; it does not implement a universal JSON Schema engine.

`CountTokens` receives the entire serialized provider request: messages, model,
instructions, schema and output cap. It must account for model-specific framing and
schema overhead using the actual tokenizer. A character count or scripted return
value is not acceptable for live budget enforcement. Counting performs no I/O.
`CountInputTokens` and `Call` use the same request construction; counting occurs
again before dispatch to reject an oversized input even for direct transport use.
The caller reserves the model call and host-defined cost before `Call`; the adapter
does not introduce a billing catalog. Completion usage includes the provider's
reported completion total, rather than visible content length.

The client bounds the response body, rejects malformed JSON, duplicate keys,
invalid UTF-8, trailing data, missing/inconsistent usage, excess token counts and
unsupported response shape. Typed output uses `UseNumber` and rejects unknown
fields. Domain validation receives an independent byte copy. Context cancellation
after validation still suppresses output. All transport/protocol errors omit
response content, refusal text, queries, credentials and callback diagnostics.

The HTTP client configuration is copied and redirects disabled. There are no
adapter retries, fallback calls or background tasks. Custom `Transport` and all
host callbacks must be bounded, concurrency-safe and perform no hidden retries;
arbitrary injected Go implementations cannot be sandboxed. A timeout bounds the
attempt, in addition to any earlier parent deadline. Provider token caps do not
replace local input admission.

## Typed graph extraction binding

`NewExtractor[TKind, TRel, TAttr](config, actualPrice)` exposes `Model` and
`CountInputTokens` directly for `graphingest/extraction.Config`. The configured
schema describes `extraction.ModelOutput[TKind, TRel, TAttr]`: entities and
relations with local IDs, BYOT kinds/attributes and snippet ordinals. It must not
request canonical IDs, source revisions, namespaces or permissions from a model.
Those identities are derived by core extraction from admitted source mappings.

`actualPrice` returns host-defined integer units using the same policy as the
core reservation. If actual pricing or usage cannot be determined, accounting
stays unknown and the core retains the full reservation. Known token overruns
are rejected even when pricing is unavailable. Wire/schema errors preserve known
token/cost usage so the ledger can settle failed calls without retrying them.

This HTTP adapter does not decide source access, validate source retention, resolve
entities, generate final answers or store graph facts. Configure those existing
core ports explicitly. Baseline retrieval remains independent of this module.

HTTP regression servers supply deterministic responses and prove transport
behavior. They are not evidence of live model quality or tokenizer accuracy.

## Community/global summary binding

`NewSummarizer(config, actualPrice)` exposes `Model` and `CountInputTokens` for
`recipe/graphsummary.Config`. Configure the output schema for `text` plus selected
input ordinals. Map calls receive admitted original snippets; reduce receives two
derived community summaries. The core resolves ordinals back to immutable original
supports and requires both communities in global reduction. Source refs, canonical
membership and access metadata stay outside model requests. The same generic HTTP
limits, schema validation and usage preservation apply. Successful HTTP fixtures
do not substitute for the comparative live model experiment.

## Retrieval planner/assessor bindings

`NewPlanner[TIntent,TRequestMeta]` selects one explicit recipe strategy and bounds
query UTF-8 bytes. Its `Plan` method plugs into `recipe.Config.Planner`. The host
schema describes `PlannerOutput` (`queries`); at most one, two or three derived
queries are accepted for rewrite, multi-query or decomposition respectively.
Empty/duplicate/oversized queries fail without redispatch. Model requests expose
only the effective original text, strategy and query cap.

`NewAssessor[TIntent,TRequestMeta,TMeta]` provides `Assess` for
`recipe.Config.Assessor`. Its schema describes `AssessorOutput` (`selected`,
`sufficient`). Configure maximum executed queries, total documents across queries
and aggregate UTF-8 query/snippet bytes. The model sees original text, query indices,
query text and admitted document content. Document IDs, supports, scores, BYOT
intent/metadata, filters and authorization/publication identities are excluded.
Selections must refer to actual executed indices and contain no duplicates.

Both methods receive `recipe.ModelLimits` from the concrete reservation. Input
counting uses those limits, and the wire completion cap uses the exact output
reservation. Scope freshness is checked before projection, immediately before HTTP
(after the host token counter), and after output validation/price calculation.
Known failed-call usage is preserved; unknown actual price retains the reservation.
These optional bindings do not supply a model choice, exact tokenizer, instructions,
executable schema or price policy. Live comparative recipe acceptance still requires
that explicit host configuration and execution.
