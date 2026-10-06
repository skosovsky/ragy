# Cohere reranking

`rerank.New[TMeta]` implements the Cohere v2 `/rerank` protocol with explicit host model selection. `Rerank` satisfies the query reranker interface; `RerankWithUsage` also returns actual `meta.billed_units.search_units` as `Usage.BilledUnits`. Missing counters remain unknown. No input-token estimate or price is fabricated. Observed usage can remain available when delivery is revoked or malformed ranked results are rejected after a successful provider envelope.

`Config.Limits` bounds local work. Zero values select 128 inputs (query plus documents), 1 MiB aggregate UTF-8 text, 2 MiB request, 16 MiB response, and 30 seconds. The token-matrix row limit does not apply to reranking. There is one dispatch and no library retry; standard HTTP clients are cloned with redirects disabled. Custom Doer implementations must honor cancellation and must not hide retries or redirects. Raw bodies, credentials, URLs and transport diagnostics are excluded from returned errors. Parent cancellation and deadlines propagate.

These bounds are not a hard remote token or billing limit. Cohere may truncate documents according to its service policy. The adapter preserves original document content, metadata, resolver, score history and protected delivery behavior. Returned model-native scores are finite and retained without assuming normalization.

Protocol fixtures are grounded in [Cohere v2 rerank reference](https://docs.cohere.com/v2/reference/rerank), verified 2026-10-06. Ordinary tests use local fixtures and adversarial HTTP servers. Live smoke is explicitly skipped unless `RAGY_PROVIDER_SMOKE=1`, `COHERE_API_KEY`, and host-selected `COHERE_RERANK_MODEL` are supplied; enabling it authorizes a paid call.
