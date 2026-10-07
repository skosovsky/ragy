# Provider identity, limits and accounting

`Space` is the host declaration of model, revision, preprocessing configuration,
vector-space identity, dimension and metric. Revision/configuration are host
attestations; absent provider model echo does not attest remote revision. A supplied
contradictory echo rejects. Optional adapter `Config.Model` asserts equality with
`Space.Model`; it selects no default model. Query/document compatibility is a host
contract. No adapter normalizes vectors or silently changes model, dimension or
purpose. Custom encoders can enforce hard remote token bounds; supplied adapters
reject `RequireRemoteTokenBound` before dispatch.

| Setting | Unit and zero/default policy |
|---|---|
| `embedding.Limits.MaxInputs` | 128 ordered inputs; Cohere counts query plus documents, so at most 127 documents by default and MaxInputs=1 admits no nonempty rerank |
| `MaxInputBytes` | 1 MiB total UTF-8 text bytes; media adapters have additional payload admission |
| `MaxRequestBytes` | 2 MiB marshaled JSON bytes, checked after marshaling |
| `MaxResponseBytes` | 16 MiB retained wire bytes before decode |
| `MaxVectorRows` | 8192 rows per returned token matrix; applies to Jina tensor, unrelated to completion tokens and dense dimension |
| `Timeout` | 30 seconds, earlier caller deadline wins |
| Structured limits | explicit positive per-call input/output token limits plus positive configured byte/duration caps; host tokenizer counts actual envelope |
| PDF limits | explicit positive input/output bytes, pages, per-page words/cells/images and timeout |

Negative embedding limits reject; zero selects finite defaults. JSON arrays can
allocate before shape/row validation and request marshaling allocates before byte
admission. PDF engines discover pages and extract elements before output admission.
These are response/input/output limits, not peak RSS/CPU or a process sandbox.
Hosts own process/container resource isolation and enforce cost/token policies.

After a fully decoded successful provider envelope, valid observed usage survives
rejected cardinality, model or vectors (including vector JSON type/float32 decode): embeddings are empty and error must be
checked. Missing counters stay unknown with zero values; no estimates, inferred
billing or price tables. Invalid counters reject protocol and do not become known.
Unreadable, truncated or malformed JSON never becomes usage evidence. OpenAI uses
prompt_tokens, Jina total_tokens, Gemini promptTokenCount; billed units remain
unknown. Cohere observes search_units independently of ranked-result admission.
Cohere input/prefix retained with error is not reranked success; inspect the error
and protected delivery before using scores. Prices and billing reconciliation are
host-owned.

One exchange, no library retries. Standard HTTP clients disable redirects; custom
Doers must honor context, perform no hidden retries/redirects, and be concurrency
safe. Clients capture scalar configuration, use per-call buffers and transfer
successful payload ownership to caller. Host input slices/payloads remain borrowed
and must not mutate during calls. Credentials reject blank, invalid UTF-8 and ASCII
control/DEL bytes before dispatch; accepted bytes are unchanged. URL policy rejects
credentials, queries including bare ?, fragments and opaque URLs. Transport errors
are sanitized ErrUnavailable, body/syntax/materialization errors ErrProtocol;
context cancellation wins. Provider non-success statuses retain core status classes.
Unsupported profiles are ErrUnsupported, local configuration/input ErrInvalidArgument.

Shared embedding/rerank JSON envelopes permit unknown fields. They retain standard
encoding/json semantics: last duplicate member wins, field names match without
case distinction, escaped unpaired surrogates and invalid UTF-8 in response strings become U+FFFD. This is an explicit
normalization policy, not strict syntax admission. Structured separately rejects
decoded duplicate members and excessive depth, validates required choice index,
and delegates domain schema to a bounded host callback. Case alias/surrogate
normalization remains standard there too.

Local wire tests and compilable examples certify configured behavior; optional
paid live profiles and quality/reference budgets are separate. No fixture proves
remote revision or token enforcement. Dense metric validation/float64 scoring is
retained without optimization or speedup claim; before/after measurements do not
apply to these contract changes.
