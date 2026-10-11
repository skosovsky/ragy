# Current → SDK → ragy matrix

SDK inventory verified from official public repositories and Go module registry
2026-10-11. Pins: OpenAI v3.76.0, genai v1.73.0, cohere-go/v2 v2.19.0.
Wire protocol/DTO/auth/errors are SDK-owned; domain admission/mapping is ragy-owned;
identity, credentials/HTTP transport/client lifecycle are host-owned.

| Operation / old API | Public SDK API | ragy mapping/status | Evidence / migration consequence |
|---|---|---|---|
| OpenAI dense New APIKey/Model/Space/BaseURL/HTTPClient/Limits; Embed | Client.Embeddings.New | Implemented; SDK injection replaces credentials/URL/Doer; ordered string batch, float encoding, dimensions except ada-002 | dense HTTP fixtures; explicit metadata index/cardinality/model/dimension/finite admission; prompt_tokens known by field metadata; billed unknown |
| OpenAI structured New[T]/CountInputTokens/Call and recipe bindings | Client.Chat.Completions.New | Implemented; chat preserved, strict JSON schema/store=false/max_completion_tokens; host validator/tokenizer retained | structured request/recipe/extraction/summary/publication fixtures; schema before dispatch, usage before refusal/domain reject, protected admission/final delivery retained |
| Gemini dense New/Space/Embed | Client.Models.EmbedContent, BackendGeminiAPI | Implemented; one batch, task types model001, text prefixes model2, output dimension | dense batch/local/adversarial fixtures; SDK-L03 unknown usage/echo fields; SDK-L01 byte representation change |
| Gemini multimodal New/Space/Embed | Models.EmbedContent with Content.Parts/Blob | Implemented; model2 text and inline PNG/JPEG, up to six images, one vector per input | multimodal HTTP fixture asserts bytes, MIME, content order and count; no text-only substitute |
| Cohere New[TMeta]/Rerank/RerankWithUsage | Client.V2.Rerank(V2RerankRequest) | Implemented; query/documents/top_n=count; complete ranking sorted finite scores, source docs/history/metadata/resolver retained | rerank access/bounds/client fixtures; observed search units retained before rejection; input tokens unknown; SDK-L04/L05 |
| All migrated encoded request/response caps | Public parameter/response DTOs | SDK-limited representation bounds, not wire budget | SDK-L01; SDK owns wire decoding, no alternate HTTP/parser path |
| Embedding RequireRemoteTokenBound | No enforceable public remote cap | Unsupported before I/O | existing local-rejection tests; bytes do not imply tokens/billing |
| Gemini Vertex, URL/file/audio/video/PDF embedding profiles | SDK offers broader backend/media APIs | Outside this bounded adapter profile; unsupported before I/O, host may use SDK directly | model/backend/MIME/PartURL preflight; no silently dropped prior operation, no fallback |
| OpenAI base64/token-array inputs/User and additional completion tools/Responses | Public SDK capabilities | Outside ragy text/structured port; host retains SDK client for direct SDK use | No previous ragy operation removed, no copy of all SDK options into adapter Config |
| Cohere partial top-N/priority/max_tokens_per_doc | Public V2 SDK fields | Outside current complete-ranking port; complete top_n explicit, service truncation unchanged | host selection; no hard token guarantee, no invented partial success |
| Jina dense/tensor | No official maintained Go SDK covering hosted dense + multi-vector operations found | Explicit inventory exception; unchanged integration | official [Jina org](https://github.com/jina-ai), [API schema](https://github.com/jina-ai/meta-prompt/blob/main/v1.txt), [on-prem API compatibility](https://github.com/jina-ai/jina-on-prem/blob/main/docs/API-Reference.md). OpenAI-compatible on-prem does not establish hosted tensor coverage; no community SDK substituted |
| Qdrant BYO Client bridge | Official [qdrant/go-client](https://github.com/qdrant/go-client) uses generated protobuf APIs | Inventory only; existing domain Client not directly implemented by SDK | README/Client interfaces require host mapping; no native driver added or interface deleted |
| Elasticsearch BYO Client bridge | Official [elastic/go-elasticsearch](https://github.com/elastic/go-elasticsearch) | Inventory only; host mapping/response lifecycle required | existing Client contract differs from official typed/raw client; no native bridge dependency |
| Neo4j BYO Runner bridge | Official [neo4j/neo4j-go-driver](https://github.com/neo4j/neo4j-go-driver) | Inventory only; host owns driver/session/transaction mapping | Runner contract differs from driver API; no new storage implementation |
| internal/providerhttp | Remaining Jina dense/tensor consumers | Retained shared helper as inventory exception | OpenAI/Gemini/Cohere production imports removed; Gemini own wire client replaced by internal/binding; no stale HTTPClient production option |

Unsupported means adapter capability, not that an entire SDK lacks the feature.
No mandatory prior operation is unavailable through the selected SDKs. No change
to stores/core algorithms/prompty/sibling applications. Exact dependency semantics
and accepted upstream defects are in limitations.md; offline evidence is not live
model/backend certification.

## Field and error mapping

| ragy request/config/result | SDK field / ownership / rejection |
|---|---|
| dense.Config.Space + optional Model | SDK model=Space.Model; constructor equality/profile validation, ErrInvalidArgument/ErrUnsupported; returned Space is unchanged host identity |
| dense.Request.Inputs/Purpose/RequireRemoteTokenBound | ordered SDK text inputs; OpenAI symmetric; Gemini task/prefix mapping; empty/count/UTF-8/aggregate bytes validation; remote-bound flag ErrUnsupported before dispatch |
| dense.Result.Embeddings/Usage | SDK ordered vectors or OpenAI indexed vectors → host Space + float32; invalid/missing/duplicate/range indices/cardinality/dimension/nonfinite ErrProtocol; empty embeddings on failure; valid independently exposed usage retained |
| multimodal.Request.Inputs[].Parts Kind/Text/MIME/Bytes/URL | ordered genai.Content/Part/Blob; text or inline PNG/JPEG; no fetched URL/file; Kind/MIME unsupported before I/O; borrowed immutable input bytes, transferred vectors |
| embedding.Limits MaxInputs/MaxInputBytes | local exact aggregate/count validation; Cohere query participates in count/bytes; negative invalid, zero finite defaults |
| embedding.Limits MaxRequestBytes/MaxResponseBytes/Timeout/MaxVectorRows | public DTO caps + earlier deadline; dense has no matrix rows; no remote token/billing bound; SDK-L01 describes changed wire guarantees |
| structured.Config Model/Instructions/SchemaName/Schema/Validate/CountTokens | SDK model + system/user messages, ResponseFormat JSONSchema strict=true; schema cloned and syntax/object/name validated before I/O; pure bounded host callbacks own executable validation/tokenization |
| structured.Config MaxRequestBytes/MaxResponseBytes/Duration | explicit positive limits; SDK parameter envelope serialized for CountTokens, public completion DTO for response representation cap; per-call earlier timeout |
| structured.Limits InputTokens/OutputTokens | count before I/O; output SDK max_completion_tokens int64; zero/overflow invalid; overrun budget.ErrUsageExceeded, admission budget.ErrExhausted |
| structured.Call[T] output/Usage InputTokens/OutputTokens/Known | SDK prompt/completion/total token metadata → validated consistent nonnegative uint64 counters; usage retained before refusal/incomplete/content/host validation errors; no content emitted on error |
| Cohere.Config Model/Limits, query + ResultSet[TMeta] | SDK V2 model/query/documents/top_n=count; source document projection validated; invalid query ErrEmptyText; access checked before payload and after completion |
| Cohere.Result Documents/Usage | index→original document, model-native score, stable descending order, prior scores observed, original resolver/BYOT metadata; search units independently admitted (SDK-L04); malformed rank preserves permitted error result, protection suppresses it |
| SDK API/transport/decode errors | preserve SDK cause/type behind providerpolicy.SDKError safe Error(); status classes via public SDK status APIs, decode/body ErrProtocol, transport ErrUnavailable; caller cancellation/protection still wins and preserves causes |
| legacy credentials/BaseURL/HTTPClient/Doer | removed from production adapter Config; host constructs borrowed concrete official client with public auth/URL/HTTP/retry settings; no adapter-owned client close/global config |

Embedding/completion/rerank operations are synchronous, with no adapter consumer
stream API. Stopping a host consumer propagates its context cancellation through
the synchronous call and existing core delivery gates. No claim of remote effect
rollback. In-flight/post-response cancellation and concurrency are fixture-tested;
SDK-L06 identifies the upstream headers-cancel body lifecycle limitation.
