# Managed embedded graph

This volatile graph target admits mandatory/query node and edge scope before traversal and retains original source supports. Facts belong to exact published lifecycle versions or an explicitly selected immutable host basis. It cannot recover its fact corpus automatically from durable lifecycle manifests after restart.

Configure both positive capacities when constructing `Config[TMeta]`:

- `MaxRecords`: maximum node+edge records in one staged payload or host basis.
- `MaxAdmissionRecords`: maximum total selected records across publications and the optional host basis, before scope filtering, conflict removal or deduplication.

Neither capacity has an implicit default. Existing consumers must set `MaxAdmissionRecords` explicitly. Shared identical facts in two versions count twice because both versions require admission. Exceeding admission capacity returns an error wrapping `ragy.ErrInvalidArgument` with `graph admission record capacity exceeded`; no partial payload or delivery clone callback is returned. `AdmitTraversal` is a shape/capability preflight and does not load inventory or predict this capacity failure.

`Request.MaxNodes` and `MaxEdges` are result limits. Per-read admission still scans selected facts and target support inventory. The resulting outbound/inbound adjacency indexes contain only edges whose two endpoints survived scope and conflict admission; expansion examines incident edges. Cycles and self-loops are deduplicated, and conflicting facts cannot create a bridge.

Lifecycle store load/validation and retained-version selection still scan metadata independently of `MaxAdmissionRecords`. Support counts, identifiers and metadata byte sizes remain host controlled. Hosts need separate namespace retention, byte quotas and context deadlines; this adapter does not provide a complete memory-byte or CPU bound, graph ANN, a scheduler or distributed transactions.

The admission and traversal contract (local task archive) describes ownership, failure boundaries and actual cost. Reproducible before/after workloads are in TASK18 results (local task archive).
