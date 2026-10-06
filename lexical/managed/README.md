# Managed BM25 target

`managed.Adapter[TMeta]` implements lexical retrieval, lifecycle `StagePort` and
`CleanupPort`. Register it in `Executor` and `Cleaner`, supplying a finalized
metadata schema, host metadata clone function, required positive `MaxCachedSnapshots`, durable manifest store and validated
payload fingerprint. Stage the full `[]managed.Record[TMeta]` inventory.

Use `lifecycle.CapturePublication` before planning/fan-out and construct the trusted
binding with that publication. This adapter requires pinned reads and rejects live
admission before I/O. Complete capture requires every requested target ready. Explicit partial capture
freezes excluded targets; this adapter rejects a pin excluding its own target. Tombstoned sources are
excluded; an empty capture is pinned complete-empty, never a live fallback.

Records are keyed by exact namespace/source/revision/transformation/access identity.
Returned document IDs are canonical artifact-reference hashes; original artifact IDs
remain in each record's reference and host metadata. Stage preserves supplied source locators and adds the canonical artifact reference.
Supply original mappings/supports on the staged document; an artifact reference alone
does not establish original quotation precision. Query scope metadata is checked before payload cloning. Readonly BM25 snapshots
retain the original binding fingerprint, gating callbacks and final delivery.

Staging and exact-version cleanup share the adapter lock. New source publications
cannot make another version's records disappear through cleanup of an old manifest.
Every read confirms exact ledger inventory before accessing the cache and again after scoring/output cloning. A physically removed or retired snapshot returns unavailable with no latest substitution or partial sibling results.

This reference adapter stores documents **in process memory**. Durable manifest state
does not imply durable lexical records. A fresh adapter without restored records
returns snapshot unavailable; host record persistence/restoration is not supplied by
this adapter. Persistent dense/tensor acceptance remains separate.

`Adapter.InventoryObserver(maxEntries, maxRecords)` provides a bounded observation
port for `lifecycle.NewFencedInventoryVerifier`. It holds the actual mutation mutex
through the nested observation callback. Complete inventory accounts for retained
versions by `manifest:<operation-id>` opaque keys; missing memory is unavailable even
when a durable ledger retains a ready checkpoint. The observer performs no indexing,
source adoption, host metadata callback or deletion.

Scoped BM25 snapshots are retained in an LRU cache of at most `MaxCachedSnapshots`
entries; zero/negative capacity is invalid. Each adapter owns its configuration and
cache. Keys include the complete binding fingerprint (publication and policy), actual
prepared predicate fingerprint, exact staged manifest inventory digest, and mutation
generation. Stage and cleanup invalidate every entry. A cache hit still checks
binding freshness, inventory availability and output clone callbacks; authority is
never cached. Host codec, resolver and clone callbacks must remain stable and safe
for concurrent calls for the adapter lifetime. `CloneMeta` must return owned metadata.

Misses scan pinned record metadata before cloning admitted payload and build scoring
statistics solely from the admitted corpus. Hits avoid corpus clone/tokenize/build,
but retain ledger/inventory validation scans. This is an entry bound, not a byte or
full-request memory bound: admitted document sizes, concurrent misses and retained
versions are host workload limits. Eviction discards derived indexes only, never
retained records or lifecycle history. Volatile records still need explicit cleanup.

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
