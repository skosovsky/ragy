# TASK-17 observed execution contract

Baseline: `924b3c2`. Original TASK-17 scope remains mandatory. Spec-first decisions below define provider-neutral boundaries.

- Introduce optional core `observation` contract carried by context: fixed stage/outcome/error-class enums, attempt-local numeric operation/parent/query/branch ordinals, bounded counts/known usage and elapsed time. No host metadata, query, document ID, error text, credential, or mutable object enters a diagnostic event. Disabled observation does not invoke exporter/serializer. Enabled sessions have explicit finite event capacity; callbacks are serialized, synchronous and cooperative, failures/panics are diagnostic-only, never retry/change retrieval. No background queue, exporter service or arbitrary string labels.
- Instrument actual retrieval pipeline/branches/cache, text recipe/model/encoding/delivery and lifecycle execution boundaries. Start and completion distinguish successful empty, partial, failed, canceled, unsupported and resource exhaustion. Attempt-local correlation does not assert remote cancellation or billing. Counts/usage unavailable remain explicitly unknown; a local duration is observed, remote usage only supplied by actual port accounting.
- Evidence v2 adds a privacy-controlled structured decision envelope: executed query ordinals and explicitly allowed texts, contributors through selected/delivered states, per-query coverage before/after packing, sufficiency signal, stop reason, and separately host-supplied model/prompt/config/recipe revisions. Absent host revisions are unavailable, default raw text/IDs omitted. No model rationale/chain-of-thought. Required evidence sink behavior and protection suppression remain intact.
- Diagnostics default-deny all sensitive payload, including IDs/hashes; no text hashing workaround. Explicit evidence export policies grant only named fields. Limits protect decoder/record cardinality and byte size. Core imports no OTel or provider modules.
- OTel consumes the same safe events and updates existing wrapper spans with sanitized status/errors/counts/actual usage. No second capability decorator API. Select and document current official GenAI convention version, use only applicable documented attributes; ragy-owned data lives under ragy.*, operation ordinals are span attributes, never metric labels.
- Independent JSON Schema v2 fixtures must agree with strict Go decoder, including absent/unknown/unexecuted distinctions, malformed/old versions and privacy. Preserve historical task12 records/schema as historical; new current schema resides in docs/task17/schemas. Record supports audit of captured input, not deterministic replay.

| ID | Mandatory requirement |
|---|---|
| O01 | Bounded optional immutable observer contract, disabled no-work, cooperative serialized concurrency/error policy |
| O02 | Real success/partial/fallback/rescue/cache/budget/cancel/unsupported events, numeric correlation and explicit known usage |
| O03 | Actual retrieval, recipe/encoding and lifecycle instrumentation, no replay or extra dispatch |
| O04 | Structured decision record links executed query → retrieval → fusion → actual delivered contributors/coverage and stop/sufficiency |
| O05 | Explicit host model/prompt/config/recipe identities; absent unavailable, no reasoning/rationale capture |
| O06 | Default-deny text/meta/locations/IDs/hashes/auth/raw errors and opt-in negative tests |
| O07 | OTel sanitized status/errors/counts/correlation and selected documented GenAI convention version |
| O08 | Diagnostic failure cannot change/retry operation; required recording errors and protection suppression preserved |
| O09 | Versioned strict Go/JSON Schema v2 agreement via independent fixtures; old format explicit rejection, history preserved |
| O10 | Affected consumers/docs/race/lint pass, no OTel/provider dependencies in core |
