# OpenAI dense encoding

`New` requires API key and valid host-declared `embedding.Space` (revision,
configuration, identity, dimension, metric). Supported profiles:
`text-embedding-3-small` dimensions 1–1536, `text-embedding-3-large` 1–3072,
`text-embedding-ada-002` exactly 1536. Others reject locally. Optional
`Config.Model` must equal `Space.Model`. Ada omits unsupported dimensions.
Float encoding is explicit. Query/document/similarity use the same symmetric
encoder and retain identical space; no provider purpose field or normalization.

Indices, cardinality, model when returned, finite components and configured
shape are validated. Observed `usage.prompt_tokens` becomes known input usage;
absent counters and billed units stay unknown. Hosts own revision attestation,
pricing and tokenization. Strict `RequireRemoteTokenBound` rejects before dispatch.

Defaults: 128 inputs, 1 MiB combined input, 2 MiB request, 16 MiB response,
30 seconds. Negative limits reject. One dispatch, parent cancellation, sanitized
errors. Standard HTTP clients reject redirects; custom Doers must honor context
and perform no hidden retries/redirects. Byte limits do not enforce provider token
limits or remote cost; hosts must pre-tokenize when required.

Wire fixtures cite the [official API](https://developers.openai.com/api/reference/resources/embeddings/methods/create),
verified 2026-10-06; they do not attest live service. Smoke tests require
`RAGY_LIVE_PROVIDERS=1` and `OPENAI_API_KEY`, explicitly skipping otherwise.
Enabled smoke makes a billable request.
