# Canonical documents and scoped hydration

Use Hydrator/Reader with an explicit Binding, authoritative retained Catalog,
admitted Loader and deep ClonePayload for scoped source reads. The complete batch
is admitted before payload I/O and rechecked after capture/delivery. No latest
fallback is permitted. RawStore is explicit unscoped administrative storage, not
a scoped hydration path. Host controls source revision retention and permissions.

[Current source-to-index-to-resolution guide](../source/README.md) links the runnable
[retained chunk projection test](chunking_integration_test.go) with real BM25 and
exact historical resolution after newer revision/deletion/revocation. Projection
is all-or-nothing; metadata ownership follows explicit host callback contracts.
