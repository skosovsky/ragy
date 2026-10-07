# Limits, units and defaults

Limits apply at named admission/delivery boundaries. They do not imply peak allocation, process RSS, arbitrary callback CPU, provider token enforcement or exhaustive recall. Host codecs, models, source stores and native drivers must uphold their own declared bounds. Negative values reject where specified; zero has a field-specific meaning, not a universal unlimited setting.

| Surface | Unit and default/admission policy | Boundary |
|---|---|---|
| RetrieveOptions.TopK / FetchLimit | Documents; both zero invalid, negative invalid. FetchLimit zero uses TopK; positive FetchLimit must cover positive TopK. | Candidate/output count, not bytes or scan work. |
| ScoreThreshold | Explicit state/scale, inclusive minimum; nil absent, zero/negative native values valid. | Terminal score filter, not normalized universal relevance. |
| BM25 Parameters | Nil selects K1=1.2, B=0.75; explicit K1>=0 and B in [0,1], both finite, zero supported. | Native scoring configuration, not a query CPU quota. |
| Managed lexical MaxCachedSnapshots | Required positive resident snapshot count. | Not metadata bytes, retained source payload, concurrent builds or global process memory. |
| Tensor CandidateBudget / persistent MaxRecords | Explicit candidate/record counts; persistent MaxRecords separately caps stage records, decoded catalog entries, candidates, inventory keys and retained inventory records. | Exact scoring within admitted candidates; catalog discovery can inspect a larger universe. |
| Persistent catalog/payload byte caps | Individual serialized input bytes; explicit configuration. | Host separately bounds shape, aggregate storage and memory. |
| embedding.Limits | Defaults: 128 inputs, 1 MiB total text, 2 MiB marshaled request, 16 MiB wire response, 8192 tensor rows, 30 seconds. Zero selects defaults; negatives reject. | JSON may allocate before shape checks; request cap is checked after marshaling. MaxVectorRows is not completion tokens or dense dimension. |
| Cohere MaxInputs | Counts query plus documents: default admits 127 documents; MaxInputs=1 admits no nonempty rerank. | Provider request cardinality. |
| Structured completion | Explicit positive per-call input/output token bounds and configured byte/duration bounds. | Host tokenizer counts the actual full envelope; schema callback is cooperative. |
| Graph extraction MaxInputBytes | Source text bytes only. | Ontology, configuration and JSON framing are separate; CountInputTokens must account for the complete provider envelope. |
| PDF | Explicit positive bytes/pages and per-page words/cells/images plus timeout. | Engine can discover/extract before output admission; no subprocess memory sandbox. |
| ArtifactResource | Explicit positive resource Limit/unit/profile, candidate/measurement/output byte caps and measurement callback. RuneResource helper: 1024 candidates, 1024 measurements, 4 MiB output. | Complete formatted output including framing/labels; packing status is not semantic sufficiency. |
| Budget Ledger | Explicit retrieval/model call counts, input/output tokens, integer cost units and deadline. | Reservations and observed settlement; unknown price/usage is not inferred billing. |
| Observation MaxEvents | Lifetime start/end callback capacity, two callbacks reserved per admitted operation. | Events counts attempts, Dropped counts rejected pairs, Failures counts failed/panicking callbacks; no automatic reset/end. |

Use the exact config GoDoc and package guides for required positive fields and overflow policies: [embedding/providers](../embedding/README.md), [lexical](../lexical/README.md), [managed lexical](../lexical/managed/README.md), [tensor](../tensor/README.md), [persistent tensor](../tensor/persistent/README.md), [budget](../recipe/budget/README.md), [observation](../observation/README.md), [source](../source/README.md) and [layout](../layout/README.md).

Resource usage dimensions remain separate. Tokens counted by a host tokenizer, vector rows, provider search/billed units and host integer costs cannot substitute for one another. A valid independent usage envelope may remain known after rejected output; truncated/malformed JSON never becomes known usage. Unknown values stay unknown. Caps do not authorize dropping mandatory facts, truncating arbitrary sources or skipping access gates.
