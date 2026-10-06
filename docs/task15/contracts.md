# TASK-15 embedding contract

Baseline: `6e2d86d`. Status: implementation in progress. Original task remains mandatory.

## Decisions before implementation

- A provider-neutral `embedding.Space` carries model, host-declared revision/configuration, vector-space identity, dimension and explicit implemented metric. Dense vectors and token matrices remain different `dense.Embedding` / `tensor.Embedding` types. Query and document purpose select wire encoding without changing compatible retrieval-pair identity. No automatic normalization. Dense supports normalized dot, cosine, dot and negative squared L2; tensor supports normalized dot and dot MaxSim only.
- Replace all bare-output Embedder methods with request/result methods: `Space() embedding.Space`, `Embed(context.Context, Request) (Result,error)`. Requests carry explicit purpose and optional strict remote token-bound requirement; results carry typed embeddings and explicit known/unknown usage. Dense, tensor and multimodal inputs stay distinct. Host supplies pricing/tokenization; current embedding protocols cannot enforce hard remote token limits and must reject that strict requirement before dispatch.
- Dense records and vector retrieval options carry mandatory Space when vectors are present. Persistent dense envelope uses the record's space rather than a second competing declaration. Remote DB stores require a host-configured profile; this does not attest service configuration. Persistent formats are versioned and reject former envelopes; reindex instructions do not delete data.
- Adapter constructors declare configured space and bounded batch/input/request/response/tensor-row sizes and timeout. Default limits are documented finite values; invalid limits reject. One dispatch, no library retry or redirect. Parent deadline/cancellation propagate. Custom Doer explicitly promises no hidden retries/redirects; arbitrary transports cannot be attested. Errors omit credentials, URL, raw request/response and transport error text.
- Gemini uses documented `:batchEmbedContents`, `requests[].content.parts` and ordered `embeddings[].values`; query/document taskType and observed usage metadata. Multimodal uses the same real content API with a supported explicitly declared model/modality profile; unsupported parts/models reject locally.
- Jina token matrices use `/multi-vector`, input_type=query/document, embedding_type=float, dimensions and data[].embeddings. Protocol fixtures cite official URL and verification date 2026-10-06. Indexed providers validate missing/duplicate indices, cardinality, dimensions, finite values and model when returned. No invented gateway endpoint remains.

## Mandatory acceptance matrix

| ID | Requirement |
|---|---|
| E01 | Explicit compatible space, metric, purpose and known/unknown usage contracts, distinct vector/matrix results |
| E02 | Finite/shape validation; equal dimension incompatible identities rejected; query/document retrieval pair supported |
| E03 | Implemented metric semantics without implicit normalization or model naming in core |
| E04 | All embedding interfaces/consumers/fakes/OTel/examples upgraded; weak former APIs removed |
| E05 | Dense/tensor record, query, persistent and remote host-configured store profile validation |
| E06 | Official Gemini dense/multimodal endpoint/body/response fixtures and local unsupported rejection |
| E07 | Official Jina dense/multi-vector endpoint/body/response/purpose fixtures and unsupported rejection |
| E08 | OpenAI/Cohere documented wire, bounded local work and observable actual usage |
| E09 | All HTTP clients: finite bounds, parent cancellation/timeout, no retry/redirect, sanitized errors |
| E10 | Adversarial cardinality/index/model/shape/finite/oversized/refusal/error/unknown-usage tests |
| E11 | Store/query integration retains space and actual score semantics; strict unsupported token bound rejects before dispatch |
| E12 | Opt-in provider smoke tests explicitly skip without credentials; fixtures are not a live-provider claim |
| E13 | Versioned persistent format/reindex guide, all affected nested module tests and checks |

No model registry/router, billing service, quota scheduler or hidden retry is introduced. SDK dependencies remain in adapter modules.

Sources verified at implementation: [Gemini API](https://ai.google.dev/api/embeddings), [Jina ColBERT](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/). Additional providers must cite their official API in fixtures.
