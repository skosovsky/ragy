# Ragy provider adapters review

Scope: SHA b63d5e19; adapters/openai/{dense,structured}, gemini/{dense,multimodal,internal/wire}, jina/{dense,tensor}, cohere/rerank, pdf; embedding/embedding.go, dense/embedding.go, internal/providerhttp. No production edits, no live or paid provider calls. Full module tests owned by parent. Deterministic public API reproducer: `/tmp/ragy-provider-review/main.go`, module `/tmp/ragy-provider-review/go.mod`, log `/tmp/ragy-provider-review/repro.log`; `GOCACHE=/tmp/ragy-review-go-cache go run .` exits 0 while exposing both defects.

## Confirmed defects

### P2 P-01 — structured completion loses cancellation/deadline during body read

Reference: `adapters/openai/structured/client.go:227–232`. Successful headers followed by a response body that blocks until the request deadline and returns `req.Context().Err()` produce `ragy.ErrProtocol`. The context gate is after the early `if err != nil` return. This is the normal cancellation shape of an HTTP body; not a hostile callback. Callers cannot distinguish timeout/cancel from malformed protocol via errors.Is, so error accounting and retry decisions differ by the stage at which cancellation occurs. Shared `internal/providerhttp/client.go` already sanitizes cancellation separately.

Reproduction output:
```
body deadline: error=protocol error IsDeadlineExceeded=false IsProtocol=true
```

Required: preserve canceled/deadline errors before returning protocol for body read errors; check after HTTP headers as well. Keep empty output and unknown usage if response cannot be read. Do not introduce hidden retries or parse truncated JSON to claim known usage. Review transport-error classification (`client.go:216–221`) alongside shared adapter policy without leaking transport content.

AAA: arrange a context-aware response body and an otherwise valid Config; act Call with short per-attempt deadline (and separate parent cancellation); assert errors.Is(..., context.DeadlineExceeded/context.Canceled), zero output, unknown usage, one dispatch and closed body. Include an unrelated I/O read error returning sanitized ErrProtocol and valid body baseline.

### P2 P-02 — structured BaseURL accepts empty query and appends endpoint inside query

Reference: `adapters/openai/structured/client.go:74–86`. `https://provider.example/v1?` passes URL validation because RawQuery is empty; the string concatenation yields `/v1?/chat/completions`. Result: request path `/v1`, query `/chat/completions`, instead of `/v1/chat/completions`. Shared providerhttp.New rejects ForceQuery and does not have this defect. This is a bounded local config bug, not cross-host credential exfiltration.

Reproduction output:
```
force-query: constructor err=<nil>
force-query: path="/v1" query="/chat/completions"
```

Required: reject ForceQuery consistently, construct normalized endpoint from parsed URL, document trailing slash/query policy. No need a new exported transport framework.

AAA: New with trailing bare `?` fails ErrInvalidArgument before dispatch; query, fragment, credentials and opaque URL fixtures rejected; ordinary HTTPS base with zero/one trailing slash produces correct pathname; custom base path retained.

## Separate design/naming/oddity decisions (not confirmed defects)

1. **Two HTTP implementations diverge.** Structured legitimately needs usage preservation, exact token accounting and admission between host callback and dispatch. Share only small internal URL/error helpers, or maintain a common contract suite; do not force structured response semantics into embedding transport. P-01/P-02 demonstrate actual drift. `structured/client.go`, `internal/providerhttp/client.go`.
2. **JSON strictness is different.** Structured explicitly rejects duplicate keys and excessive depth, shared providerhttp merely decodes one object and allows unknown fields. Unknown fields are an intentional evolution contract; duplicate keys/case aliases/escaped lone surrogate normalization need an explicit policy distinct from unknown-field tolerance. Do not label acceptance of them a security exploit without a concrete trust-boundary impact. `structured/decode.go:88–150`, providerhttp decodeObject.
3. **Structured choice index presence.** `response.Choices[].Index int` accepts absent or null index as zero (`decode.go:14,49`). Prefer pointer required-field validation for provider envelope invariants; add missing/null tests, matching existing embedding index handling. The model/domain required-field contract already belongs to executable host schema; do not recreate universal JSON Schema.
4. **Error classes vary by transport.** Shared embedding transport sanitizes transport failures as ErrUnavailable; structured uses ErrProtocol. Decide consistent taxonomy for cancellation, I/O, response syntax, provider status and unsupported profile; document errors.Is contract, not raw provider diagnostics. Preserve existing successful non-leaking behavior.
5. **Credentials validation differs.** Gemini and structured reject CR/LF during construction; OpenAI dense, Jina, Cohere mostly reject blank keys (Cohere also size). Standard HTTP later rejects invalid headers, so not header injection evidence. Centralize safe admission if desired and avoid deferring invalid config to ErrUnavailable.
6. **Config.Model duplication.** Embedding adapters expose both Model and Space.Model, require equality when both present; can remove optional Model at clean break or mark convenience alias explicitly. Space identity remains authoritative; no implicit model selection.
7. **Host attestation vs remote fact.** Space.ModelRevision, Configuration and VectorSpace are declared identity, not validated remote attestations. Provider model echo is checked when present but often absent accepted. Keep documentation exact; do not invent model-revision certainty, silently normalize vectors or remap dimensions.
8. **Mixed defaults and strict configuration.** embedding.Limits uses finite zero defaults; structured and PDF require positive explicit limits. Valid choices for different surfaces but put a common configuration table in docs. MaxOutputTokens in embedding means matrix rows, not completion budget; rename to MaxVectorRows or MaxTokenVectors at clean break to reduce cross-package ambiguity.
9. **Output admission vs allocation bounds.** Request bodies marshal before byte cap, vector JSON arrays allocate before dimension/row validation; response bytes do cap retained wire. PDF extracts page words/tables before counting returned elements, and Python page discovery may inspect beyond page cap. Clearly distinguish output admission/byte limits from process RSS/CPU guarantees. Host process/container isolation belongs to services; no home-grown sandbox in ragy.
10. **Cohere MaxInputs includes query.** `rerank/client.go:235–247` first bounds documents and then validates query+docs. Maximum usable documents is MaxInputs-1, so MaxInputs=1 permits no nonempty rerank. This is explicitly commented, not a hidden implementation bug. Prefer MaxDocuments adapter-facing config or doc table including query semantics; unify preflight accordingly.
11. **Cohere partial/error contract.** Runtime errors can preserve input/validated prefix with error, followed by final access DeliverRead; this is explicit and not a hidden model fallback. Document consumers must inspect error and cannot interpret retained score as reranked. Consider one clear helper/typed failure representation if API changes, without swallowing errors or retrying.
12. **Usage on failed embedding materialization.** Structured preserves known usage across refusal/schema failure; embedding results discard it for malformed vector/cardinality. Cohere exposes billing before applying result indices. Decide a consistent observable-call accounting contract; do not claim usage unavailable just because data is rejected, but do not make unknown counters estimates. Price catalog and billing policy remain host-owned.
13. **Purpose/metric semantics are well bounded.** Typed query/document/similarity mapping, no implicit normalization, explicit model/dimension allowlists and RequireRemoteTokenBound rejection are justified. Do not add fallback model selection, hidden tokenization, provider upload orchestration or pricing into adapters. Query/document compatibility is host-attested configuration; purpose-specific vectors may intentionally share Space.
14. **PDF catch-all hides implementation failures.** `engine.py:88–96` maps every non-overflow/non-geometry exception to invalid_pdf. Sanitization is good; a programming/dependency failure can appear as invalid user input. Define a bounded sanitized engine_internal_error class separate from invalid_pdf, while avoiding raw exception/source data. Not demonstrated against valid fixture as a confirmed defect.
15. **PDF reproducibility.** Transformation fingerprint is explicitly host-owned and must include engine/config/dependency versions. Document tested Python/pdfplumber/pypdf versions plus fixture-generator/verifier invocation. Keep real-engine tests separately visible: deterministic fake executables do not prove PDF layout quality. This adapter is appropriately separate from core and OCR remains optional host/adapter responsibility.
16. **GoDoc discoverability and examples.** Thin embedding adapters mostly export Config/Client/Space/Embed with uneven type/method docs. Add minimal compilable client examples, concurrency/payload ownership, supported profiles, known/unknown usage, error taxonomy, limits table, BYOT identity guidance and external integration opt-ins. README already explicitly distinguishes fixtures/live quality; retain that statement.
17. **Dense metric implementation.** Similarity correctly validates exact Space compatibility, finite vectors, normalized-dot tolerance and zero-norm cosine, accumulates float64 and treats negative squared L2 as larger-is-better. Validation repeats space/vector work for both inputs and computes dot/norm/distance for every metric; optimize only with measured evidence, retaining cancellation and explicit identity checks. No need a metric registry/strategy framework for four supported metrics.
18. **No extra mandatory library needed for adapters.** HTTP envelopes, supported profile mapping and layout normalization are legitimate ragy adapters; host pricing/tokenizers/authorization/lifecycle remain passed ports. Agent UI/modes, orchestration retries, model routing, background ingestion, universal schema engine and tenant credential stores are outside scope.

## Positive guarantees observed

Embedding transport makes one bounded exchange, rejects redirects for standard clients, sanitizes raw errors, preserves ordered output through validated unique indices, and rejects cardinality/finite-value/space errors. Shared transport validates input UTF-8 before JSON marshaling. Structured uses host executable schema, independent validation copy, UseNumber, duplicate-member rejection, finite response bytes, explicit usage reconciliation and post-validation cancellation gate. PDF does not execute input as shell code, snapshots authorized bytes, keeps original page/locator identity and fails explicitly for unsupported geometry; no hidden OCR or provider fallback. Full test acceptance and live availability are outside this subreview and must be reported by parent separately.
