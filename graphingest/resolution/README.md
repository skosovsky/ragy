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
