# Persistent tensor profile

This local Darwin/Linux adapter retains exact tensor payloads and catalog inventories
for pinned publication reads. Exact-within-candidates describes scoring only.
Finding candidates can enumerate all relevant pinned catalogs and entries, even
when CandidateBudget or TopK is small. Payload loading is restricted to admitted
references; catalog discovery work is not bounded by TopK.

`Config.MaxRecords` is a positive shared configuration ceiling applied separately
to these units:

- Records supplied to one Stage call.
- Record entries decoded from one catalog.
- Maximum query CandidateBudget (also advertised as CandidateLimit).
- Directory keys considered by a complete inventory observation.
- Total retained records verified across that inventory observation.

These checks do not turn record count into a byte, token, dimension or CPU ceiling.
MaxCatalogBytes and MaxPayloadBytes bound individual serialized inputs; the host
also bounds embedding shape and aggregate retained storage. Inventory can reject
an aggregate exceeding MaxRecords even if each stage independently fits.

Published, ready manifests must match every pinned identity field. Retired,
tombstoned and unpublished manifests are excluded explicitly; ambiguous matching
live manifests fail unavailable. Catalog artifacts must agree with the retained
manifest inventory before payloads are read. Keep admission, cloning, inventory
and final freshness checks as separate ownership/publication boundaries. Their
repetition is intentional and is not a claim of a clone-cost optimization.

Query failures redact host path/storage details. Cancellation and deadline errors
remain identifiable, protocol violations remain protocol errors, other payload
failures become unavailable; late failures do not publish partial query results.

The host preprovisions a trusted root and durable ancestors. File and immediate
root-directory fsync support the local process-crash/restart profile; MkdirAll
does not establish ancestor creation durability and these tests do not prove
hardware power-loss safety. This is not a sandbox for attacker-controlled roots,
symlinks or mounts. Local nonblocking flock reports contention immediately; callers
close unduplicated lock handles. Linux flock is associated with the open file
description, rather than an abstract process lifetime. No distributed lock or
network-filesystem support is promised.

Dense and tensor retain type-specific catalogs and query gates. Their shared
[durablefs](../../internal/durablefs/README.md) layer handles narrow filesystem
mechanics; a generic storage engine would hide type/publication checks without a
measured benefit. Both selectors explicitly reject retired manifests.
