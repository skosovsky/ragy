# Library and host responsibilities

Ragy is a capability-specific retrieval library. Business types remain host-defined; adapters translate wire/storage representations into those types. A declared capability is an executable integration contract, not a Go sandbox or distributed IAM service.

| Ragy | Host or adjacent library |
|---|---|
| Typed retrieval, filter IR, scoring, explicit fusion, bounded recipes | Agent goal/loop, tool execution, workflow scheduling, product UI |
| Exact/derived source mappings, chunking and layout normalization | Original blob storage, revision retention, access policy, crop/render UI |
| Graph extraction, resolution/materialization protocols, supports | Ontology/alias truth, prompts/models and semantic grading |
| Generations, publications, pins, finite cleanup/reconcile steps | Poll/backoff scheduling, business retention and distributed authority |
| Reservations, observed usage and deadlines | Global quota, prices/billing, metrics collection and evaluation policy |
| HTTP/wire/storage ports | Native driver deployment, credentials, service/schema migration and retry policy |

Use [integration](integration.md) for composition, [ownership](ownership.md) for BYOT captures and callbacks, [errors and recovery](errors-and-recovery.md) for delivery precedence, and [capabilities](capabilities.md) for the distinction between implemented contracts and verified profiles.

`filter.RawAttributes` and private provider/storage JSON maps are allowed at the wire boundary. Public business payload remains `TMeta`; build filters through a finalized typed schema. Copy/clone policies are explicit, not a reflection-based universal deep copy.

An exact source mapping identifies one retained revision/representation. It is not an authentication proof. Resolve the original admitted source; there is no fallback to a newer revision when the captured source is absent. Derived artifact dependencies and selected citations serve different purposes. Metadata pins and crash-safe local journals do not preserve external payload or certify hardware power-loss durability.
