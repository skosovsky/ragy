# In-memory BM25

`BM25Index` keeps documents, term postings, per-document lengths and aggregate
corpus token length in process memory. `Index` stages a replacement under the
writer lock and publishes it atomically; failed rebuilds preserve the old corpus.
`Upsert` updates aggregate length without scanning all document lengths. Neither
operation persists records or restores them after restart.

A query captures the postings for its expanded terms and only their union of
candidate documents/lengths under one read lock. Owned snapshot maps preserve
term frequencies and global `docCount`/average length while writers replace
records. Copy cost scales with the selected postings and candidates; a common
term can still select the whole corpus. Ranking sorts all matching candidates
before taking the result limit, so `TopK` is not a full memory or CPU bound.

Raw `BM25Index` uses corpus-wide scoring statistics, then applies filters to scored
candidates. `managed.Adapter` first admits metadata under the complete prepared
predicate and constructs corpus statistics from admitted documents only. These
profiles deliberately retain their existing score semantics.

Constructors own `SearchFields` and synonym slices. Host tokenizer, codec and
identity callbacks must be stable and safe for concurrent use. Mutable metadata
ownership remains the host's responsibility for the raw index. Use readonly
`BM25Snapshot`/the managed adapter with a host `CloneMeta` function when owned
metadata and pinned binding enforcement are required.

Read paths check context and binding freshness immediately before and after every
metadata codec/clone callback, including failed callbacks and denied metadata.
Cancellation/revocation stops before the next codec/clone and suppresses all output;
protection takes precedence over simultaneous ordinary codec errors. Snapshot
construction also gates metadata search-field encoding and discards its temporary
construction context before publishing the index. Cache hit/miss paths never cache
authority decisions. These are cooperative boundaries, with no retries or forced
interruption of host code. Raw BM25 metadata remains borrowed under the host's
stability/concurrency contract; snapshot/managed CloneMeta retains owned output.

Protected lexical errors retain callback identities/public classifications through
`errors.Is` along with cancellation/authority causes. Callback text and arbitrary
payload-bearing callback error objects are excluded from error text/`errors.As`
traversal. Protection/gate errors remain inspectable; this scoped boundary does not
change global `access.Protect` sanitization or raw metadata ownership.
