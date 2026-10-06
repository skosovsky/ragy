# Bounded local graph expansion

This optional, model-free recipe executes one managed breadth-first traversal with
explicit host seeds, direction and node/edge filters. The target's visited set cuts
cycles. Configure finite depth, distinct visited-node and edge limits, a positive
attempt duration, clock, BYOT metadata cloner and fixed cost per dispatched attempt.
The reference profile is depth 2, 50 nodes, 100 edges and 5 seconds. A shared recipe
ledger with at most 4 graph/retrieval calls reserves the single dispatch before I/O;
the recipe itself makes exactly one call and spends no model calls/tokens.

For the declared Team A→Service→Database fixture, use Team A's canonical graph ID
as seed, depth 2 and undirected traversal through allowed `owned_by`/`depends_on`
relations. Seeds, ontology, identity resolution and allowed relation filters belong
to the host. The recipe does not interpret a natural-language question, choose an
ontology, create aliases, perform actions or generate a final answer. Apply relation
filters through the host graph metadata schema when restricting relation types.

`managed.Adapter.AdmitTraversal` validates the complete scope/pinned-publication and
traversal profile before pricing, budget reservation or target I/O. The same binding
is retained throughout execution. The managed target admits nodes and edges before
expansion/payload loading; inaccessible bridges cannot connect allowed endpoints.
Paging is unsupported. Invalid depth/seeds/capability/filter profiles fail before
pricing or dispatch. Pinned reads and explicitly selected host foundations use the
existing managed graph retention/support contracts.

Known pricing is a flat host-defined cost per dispatched graph attempt, including
failure. Input/output token reservations must be zero. Unknown price is denied before
dispatch when the shared ledger requires it, or conservatively retained in explicit
advisory mode. Budget refusal returns insufficient with a fixed stop reason and zero
graph calls; no fallback/retry/model call is performed. Target failure suppresses
evidence while the attempt remains charged. Parent cancellation, deadline, fake-clock
attempt expiry and revocation suppress output, including revocation during pricing
or metadata snapshot callbacks.

Results own graph metadata, labels, support and conflict slices. `SourceReferences`
exports unique observed original source references. It does not convert graph IDs or
explicit host foundations into fabricated citations. Conflicting source facts remain
explicit in managed evidence. A complete outcome describes a successful declared
bounded expansion; it does not attest that a natural-language answer is sufficient.
An expansion with no edges reports insufficient. The host evaluates domain evidence
and decides the next application step.

Integration tests exercise actual managed traversal, scoped cycle/private-bridge
admission, shared ledger stops, depth/node/edge limits, cancellation/revocation and
the durable lifecycle stage/publication path with original-source support export.
The comparative hybrid baseline experiment is a separate acceptance requirement.
