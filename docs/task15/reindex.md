# Reindex after the embedding contract change

Dense and tensor persistent targets now require `ragy.dense-index/inventory/v2` / `ragy.tensor-index/inventory/v2` catalogs and matching `ragy.dense-payload/v2` / `ragy.tensor-payload/v2` payloads. Former envelopes fail validation. The catalog declares the full embedding space, including its explicit metric. Dense records carry the same identity in `Value.Space`; no competing outer record space remains.

The host must select a model revision, preprocessing configuration, vector-space identity, dimension and implemented metric. Query/document purposes may encode differently while sharing this retrieval-pair identity. Do not label old vectors with a guessed identity or silently normalize them.

Rebuild embeddings from retained source revisions into a separate target directory and lifecycle target under the chosen profile. Stage and validate all source-bound records, publish the new target revision, then switch host query configuration and cache index revision. Keep the old target and manifest ledger until the host's retention policy permits cleanup. This library does not delete or convert historical data during format rejection.

Dense scoring supports normalized dot, dot, cosine and negative squared L2 (larger is better). Tensor MaxSim supports normalized dot and raw dot only. Remote stores use host-configured profiles; the declaration is not an attestation of service settings.
