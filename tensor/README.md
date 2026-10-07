# Tensor retrieval

MaxSim sums, for each query token, the largest document-token dot product.
`embedding.Dot` preserves native magnitudes; `embedding.NormalizedDot` requires
unit rows under the space's validation tolerance. Scores can be negative or exceed
one even with normalized rows. No clamping or silent normalization occurs.
`MaxSimSemanticsFor` records the actual metric. `MaxSimSemantics` names only the
normalized-dot profile.

`Rerank` computes exact scores within supplied candidates. It reports their IDs,
budget, native scores and output ranks; it does not establish exhaustive recall or
ANN behavior. CandidateBudget caps candidate count and TopK caps returned ranking.
Neither caps tokens, dimensions, bytes or CPU: validation reads all components;
scoring costs query tokens × document tokens × dimension per candidate. Hosts must
bound these input sizes and own immutable matrices for the operation.

`Embedding.ValidateContext`, MaxSim and Rerank observe cancellation during row
validation, between candidates and token pairs, and before/after ranking. One
component row and a standard-library sort remain indivisible cooperative work;
this is not a real-time cancellation guarantee. Errors return no usable score or
ranking. `Embedding.Validate` is the context-free validation convenience.

The [query adapter](query/search.go) asks the candidate backend for at most
CandidateBudget documents and rejects overflow before projection. Projection
validates exact source references and deduplicates in first-occurrence order.
The budget therefore counts backend documents before deduplication, not unique
references. Metadata cloning and reference callbacks are bounded by freshness
checks. The adapter retains separate capability, filter and publication admissions
at its candidate and tensor targets; exact scoring never adds hidden full-index
fallback. Threshold and Graph options are rejected; Vector may carry the candidate
provider's vector intent and is cleared before the tensor target call.

See [persistent storage](persistent/README.md) for retained catalogs and limits.
