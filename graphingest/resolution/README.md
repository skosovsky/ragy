# Typed identity resolution

`Resolver` consumes source-supported typed entity/relation mentions and produces
canonical groups under explicit host ontology and identity policies. Domain kinds,
relations and attributes are BYOT. It has no model dependency, source storage,
authorization engine, implicit alias table or universal ontology.

Configure ontology/policy identities, entity/relation/support bounds, validators,
identity/alias decisions, relation keys, attribute cloning/equality and original
support admission. Support admission must authorize each locator against the
captured binding and retained source catalog. The complete structural batch is
validated and all supports admitted before identity/attribute callbacks. Publication
membership alone never substitutes for source admission. Cancellation/freshness
failure suppresses the complete result, including prior identities and metadata.

Identity keys are explicit namespace/key pairs. Core does not infer namespace or
merge display names. The reference policy returns ambiguous for absent namespace,
merges Billing/Pay only in production, and keeps staging Billing independent.
Canonical IDs hash framed namespace/key tuples; ontology/policy identity is retained
separately as decision provenance. A policy identity change does not invent a new
entity identity when its explicit canonical decision remains unchanged.

Each group has independently owned attribute variants and original supports.
Equivalent variants union their supports. Conflicting kinds/attributes stay as
multiple variants; no first/latest/max-score winner is selected. Canonical groups
are ordered by ID, while variants/supports retain input encounter order. Relations
with ambiguous endpoints remain explicitly unresolved and still pass ontology
validation. Ontology callbacks validate relation endpoint kinds and attributes.
Callbacks are model-free policy functions; cloning isolates their BYOT values but
is not a sandbox for arbitrary host Go code.

Consumer chooses an explicit conflict/materialization policy before constructing
canonical graph payloads; lifecycle publication remains a separate explicit stage.
Keep ontology/policy identities with transformation fingerprints and retained
source decisions. This package does not yet supply a source extractor, model
adapter, summary recipes or a complete materialization/history integration.

Identity strings use valid UTF-8 without implicit Unicode/case/whitespace normalization.
Entity IDs/names, relation IDs/endpoints, configuration ontology/policy identifiers,
resolved namespace/key/name and relation keys must be nonempty. An input entity
namespace may be absent; ambiguity remains the host's decision. A resolved decision
must fill all three fields; an ambiguous decision must leave them empty.
Malformed configuration/direct input returns `ragy.ErrInvalidArgument` before input
support/identity callbacks. Malformed host decisions/keys return `ragy.ErrProtocol`
before hashing/grouping; all errors suppress the entire result. Earlier admitted
callbacks may already have run. Valid IDs retain the existing JSON tuple + SHA256
framing. U+FFFD is valid; malformed ff/fe bytes are rejected, so they cannot collapse
into a replacement-character key. Valid records need no ID migration.
