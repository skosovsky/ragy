# Gemini embedding adapter

Protocol verified against [API reference](https://ai.google.dev/api/embeddings) and
[embedding guide](https://ai.google.dev/gemini-api/docs/embeddings) on 2026-10-06.
Both adapters use `models/{model}:batchEmbedContents`, ordered requests with
`content.parts` and ordered `embeddings.values`. Keys are headers, never URL parameters.

Supply an explicit `embedding.Space` and finite `embedding.Limits`. Zero limits
select core defaults: 128 inputs, 1 MiB aggregate input, 2 MiB request, 16 MiB
response and 30 seconds. Space configuration must identify this preprocessing
policy; compatible query/document profiles must use the same identity.
No vectors are silently normalized. Results carry provider token usage when
`usageMetadata.promptTokenCount` exists; otherwise token usage is unknown.
Billed units remain unknown. Hard remote token bounds are unsupported and fail
before dispatch. A custom HTTP Doer must honor cancellation and perform exactly
one exchange without retries or redirects.

Dense supports explicitly `gemini-embedding-001` and `gemini-embedding-2`.
Model 001 sends `RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT` or `SEMANTIC_SIMILARITY`.
Model 2 does not support taskType: text query uses the documented search prefix,
document uses `title: none | text:`, similarity uses sentence similarity prefix.
Dimension is explicit, at most 3072. Model availability is provider-owned.

Multimodal supports model 2 and a deliberately bounded subset: UTF-8 text and
inline PNG/JPEG, maximum six images per input. Each input aggregates its parts
into one embedding. Text-only inputs use the same model 2 purpose prefixes;
inputs containing images preserve original parts without task instructions,
as recommended by the guide. Query/document purposes therefore share the
model's cross-modal space. URLs, uploaded-file references, audio, video and PDF
are locally unsupported: the library neither uploads files nor decodes media
or attests remote duration/page limits. The host owns valid image payloads.

Protocol fixtures do not claim live provider execution. Live smoke tests require
`RAGY_LIVE_GEMINI=1` and `GEMINI_API_KEY`; multimodal additionally requires
`RAGY_GEMINI_IMAGE` pointing to a PNG. They explicitly skip otherwise.
