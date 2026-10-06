# pgvector store profile

`Config.Space` is mandatory and names the host-owned model revision, preprocessing configuration, vector space, dimension and metric. Both query vectors and records must match it exactly, including when dimensions happen to agree. Validation runs before database I/O. `Space()` returns the configured declaration; it does not inspect or certify existing rows.

This adapter implements cosine only. It queries `vector <=> $1` in ascending distance order and returns `1 - distance` with `dense.cosine` semantics. Negative cosine scores remain negative; vectors are not normalized by the library. Other metrics return `ErrUnsupported` at construction. This follows the [pgvector operator contract](https://github.com/pgvector/pgvector#distances), verified 2026-10-06.

Before adopting the breaking contract, inventory the old table's embedding model, revision, preprocessing and vector dimension. Re-embed into a separate table if any identity component changed or cannot be established; provision its column dimension and cosine operator class explicitly. Configure the new profile only after those host checks. A new declaration cannot make unknown old vectors compatible. No migration, data deletion or service configuration is performed by the adapter. Metadata and SQL transport remain host-provided.
