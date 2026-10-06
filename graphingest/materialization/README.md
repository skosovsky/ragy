# Resolved graph materialization

`Materializer.Build` converts a typed resolver result into a source-bound planned
lifecycle manifest and owned `managed.Payload`. It does not stage, publish, delete
or schedule work. Host supplies graph kind/attribute projectors, metadata/attribute
cloners, graph schema, finite fact/support budgets and original support admission.
The request supplies source identity, target, expected publication, idempotency key
and declared content/payload fingerprints. Host must derive those fingerprints
from its actual source/extraction/config payload; changing assertions under an
unchanged declaration is not a valid idempotent request.

Only assertions supported by the requested namespace/source/revision/access tuple
are materialized. The complete selected inventory is structurally validated and
admitted before projector callbacks. Ingestion support admission can authorize a
revision not yet published; it still checks retained source access under the original
read policy. Cancellation/freshness checks gate every projector/clone and delivery.
Original representation and transform remain in lifecycle artifact supports.

The materialized transformation fingerprint binds the declared extraction transform
plus ontology and identity policy names. Canonical fact IDs remain stable across
policy configuration changes that keep the same explicit identity decisions.
Ontology/policy names must match the supplied resolver result. Preserve the typed
resolver result separately as a decision record when durable decision history is
required; a transformation digest alone is not a complete history of assertions.

Multiple conflicting variants within one source revision are rejected; no arbitrary
winner is selected. Conflicts across sources can materialize as separate source
versions and remain explicit managed read conflicts. Relations require both endpoint
nodes to have actual supports from that source; missing closure is rejected rather
than attributed to invented source evidence. Unresolved mentions from the requested
source prevent complete materialization. Graph labels/types/metadata are schema
validated before the plan is returned. Empty managed payloads are not tombstones;
source deletion remains an explicit lifecycle tombstone operation.

Pass the result to lifecycle Prepare/Stage/Publish. Its complete original support
inventory enables shared-fact cleanup: removing one source drops only its support,
and removing the last source removes the derived fact. Host scheduler, policy
history storage, real model extractor and summary recipes remain separate.
