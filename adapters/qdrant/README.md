# Qdrant store profile

`Config.Space` is mandatory and names the host-owned model revision, preprocessing configuration, vector space, dimension and metric. Both query vectors and records must match it exactly before client I/O. `Space()` returns the configured declaration; the client abstraction does not inspect or certify collection configuration or stored vectors.

Supported metrics are cosine, dot, and normalized dot. The host must provision the collection with Qdrant Cosine for the cosine profile, and Dot for dot/normalized-dot profiles. `Client.Search` must return the native similarity score unchanged (larger is better). Documents carry the corresponding `dense.cosine`, `dense.dot` or `dense.normalized-dot` semantics. The library never normalizes vectors. Negative squared L2 is rejected at construction: this client contract does not convert service distance scores into that core similarity metric. See [Qdrant similarity search](https://qdrant.tech/documentation/search/search/), verified 2026-10-06. Qdrant's own [cosine normalization](https://qdrant.tech/documentation/manage-data/collections/) belongs to service configuration, not a library guarantee.

Before adopting the breaking contract, inventory the existing collection's model, revision, preprocessing, dimension and distance configuration. Re-embed into a separate correctly configured collection if identity changed or is unknown. Configure the new profile only after those host checks. A new declaration cannot make unknown old vectors compatible. No migration, data deletion or service configuration is performed by the adapter. Domain metadata and client implementation remain host-owned.

Custom Decode receives schema-canonical attributes; omitted whole attributes
normalize to nil, and codecs accept nil/empty equivalently. Present null/wrong-kind
values are rejected. The host Client owns native transport, retries and remote
predicate enforcement; local wire tests do not certify tenant enforcement on a
real service. Raw Get/Delete are explicit administration outside read bindings.

Deletion Client methods must report exact nonnegative affected counts. Negative
counts fail protocol; unknown/asynchronous counts must be observed by the host or
rejected (ErrUnsupported for an unavailable exact-count profile), never replaced
with submitted ID length or guessed zero. This exact-only profile does not claim
that an async acknowledgment identifies an affected count.
