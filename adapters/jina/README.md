# Jina encoding adapters

`dense.New` requires host-declared `embedding.Space`: `jina-embeddings-v3`,
dimension 1–1024. `tensor.New` supports `jina-colbert-v2`, dimension 64/128,
and dot/normalized-dot MaxSim. Unsupported profiles reject locally. Optional
`Config.Model` must equal `Space.Model`. No automatic normalization occurs.

Dense query/document selects `retrieval.query`/`retrieval.passage`, similarity
selects `text-matching`. Late chunking is disabled. Matrices use
`input_type=query`/`document`; similarity rejects. Query/document retain one
host-defined compatible space. Hosts own revision and configuration attestation,
and must use a distinct identity for incompatible task families/preprocessing.

Actual endpoints: `/v1/embeddings`, `/v1/multi-vector`; matrices use
`data[].embeddings`. Observed `usage.total_tokens` becomes known input usage;
absent counters and billed units remain unknown. ColBERT may truncate documents
at 8192 tokens and queries at 32 tokens. Byte limits cannot enforce tokenization
or billing. Strict `RequireRemoteTokenBound` rejects before dispatch. Hosts own
pricing/tokenization.

Finite default limits: 128 inputs, 1 MiB combined input, 2 MiB request, 16 MiB
response, 8192 rows per matrix, 30 seconds. Negative limits reject. One dispatch,
parent cancellation, sanitized errors. Standard clients reject redirects; custom
Doers must honor context and perform no hidden retries/redirects.

Official sources verified 2026-10-06:

- [Dense API](https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/)
- [ColBERT API](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/)

Fixtures do not attest live service. Smoke tests require `RAGY_LIVE_PROVIDERS=1`
and `JINA_API_KEY`; absent opt-in/credentials explicitly skips. Enabled smoke
makes a billable request.
