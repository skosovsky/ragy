# TASK-13: retrieval composition contract

Baseline: `d541ca710bdc7cb876467b6ae181af11ee11873c`. Status: implementation in progress.

## Decisions

- Keep independent `RequestBackend`, `ReadCapabilityProvider` and `PublicationAdmission` capabilities. A decorator forwards actual guarantees and reports unsupported for an absent guarantee; implementing a forwarding method does not grant the wrapped leaf a capability.
- Request-aware `RequestReadAdmission` is supplementary leaf preflight and its coverage survives node/decorator composition. Cross-type projection uses an explicit pure `AdmissionProject` for every cross-type scoped/pinned projection, independent of wrapper order; ordinary payload `Project` is never run during admission. Tensor candidate/scoring composition negotiates both targets.
- Cache/projection/tracing preserve the original binding. Scoped and pinned admission precedes callbacks and target I/O; delivery suppresses protected payloads. Ordinary partial failures remain ordinary partial failures. Cache storage is used only for current complete reads; retained pins and partial coverage always reach the leaf because an entry cannot prove physical retention.
- RRF contributes once per merge key per list, at the first (best) original list position. Duplicate observations/supports are retained; payload conflict remains an error. A different list contributes independently.
- Host basis IDs are immutable for the lifetime of a managed adapter. Release irreversibly retires an existing ID, including identical re-registration; new facts need a new ID. Retirement stores only the ID, releases payload, and does not claim persistence across adapter restart. Releasing an unknown ID is an idempotent no-op.
- Graph supports are looked up by `(target, artifact reference)` in the captured manifest, never by reference alone.
- Retrieval request intent/meta, plan intent and execution metadata are host-owned immutable inputs while execution is active, including parallel branches. Value passing is not a deep copy. A host that needs mutable per-branch state must construct/clone it inside its custom node. No implicit cloning or reflection is provided.

## Requirement matrix

Each row is mandatory. Completeness is confirmed rows / 18; a row is confirmed only after its whole stated behavior and test evidence are inspected.

| ID | Requirement | Evidence required |
|---|---|---|
| C01 | Minimal independent capability contracts, unsupported/denial distinction | contracts and admission tests |
| C02 | Cache forwards publication admission without inventing guarantees | complete/partial pin and unsupported regression |
| C03 | OTel forwards schema/read/publication guarantees | actual pipeline and decorator permutation tests |
| C04 | Projection and all supplied backend decorators preserve binding and protected delivery | code inventory and adversarial tests |
| C05 | Plain/decorated scoped current reads agree | external typed consumer |
| C06 | Plain/decorated complete pins agree | external typed consumer |
| C07 | Plain/decorated partial pins agree and retain partial coverage | external typed consumer |
| C08 | Excluded target/scope conflict/unsupported/revocation prevent payload I/O; cache hit checks freshness | direct and pipeline negative tests |
| C09 | Missing retained revision never substitutes latest | actual managed/persistent conformance |
| C10 | RRF one vote per key/list, best original position | duplicate-list regression |
| C11 | RRF cross-list consensus still contributes independently | numeric/order regression |
| C12 | RRF duplicates retain supports/observations and conflicts remain errors | source/score/conflict regression |
| C13 | Retired basis cannot be rebound; old readers remain unavailable | release/re-register/read regression |
| C14 | Basis identity guarantee honestly limited to adapter lifetime | contracts and restart profile |
| C15 | Same artifact ref in different targets never mixes supports | actual multi-target manifest regression, reversed target order |
| C16 | Request/parallel BYOT ownership explicit; no false deep-copy promise | Go docs and concurrent custom struct consumer |
| C17 | Public conformance extended and external module exercised; races checked | GOWORK=off and -race results |
| C18 | Updated docs/consumers, no compatibility layer, module checks | diff review and test/lint results |

No existing signatures need replacement for these repairs. `AdmissionProject` expresses a new pure preflight capability, not a compatibility overload. Existing independent capability interfaces are meaningful capabilities, not compatibility aliases. Future observer fields and embedding APIs belong to subsequent tasks.
