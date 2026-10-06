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
source decisions. Extraction belongs to [extraction](../extraction/README.md), source-bound plans to
[materialization](../materialization/README.md), optional declared decision records
to [history](history/README.md), and summaries to [graphsummary](../../recipe/graphsummary/README.md).
[Pipeline composition and explicit lifecycle handoff](../composition.md) connects
these packages without moving transport or policy into the resolver.

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

Decision.Name is the canonical identity name, while Entity.Name is a local mention
label. A host alias policy may resolve “Pay” and “Billing” to the same key and
canonical Name “Billing”. Returning the same namespace/key with different canonical
names fails with ErrProtocol; core never chooses the first/latest label. Public
names remain unchanged to preserve the decision-record JSON shape.

Unresolved.Kind is a fact category: the literals `entity` and `relation`, distinct
from the generic ontology Kind. It is retained as a string with these documented
values rather than adding a second meaning or renaming persisted fields.

Generic kinds must have stable, reflexive equality over the lifetime of decisions.
Use named strings/integers or immutable comparable value enums. Pointer identity
and NaN are unsuitable identity kinds; faithful deterministic JSON round trips
are additionally required for history. The host enforces its domain; core does
not add reflection restrictions or a universal ontology.

Variant matching invokes host Equivalent and clones its arguments; support unions
scan in stable encounter order. A group with V distinct same-kind variants may
require O(V²) comparisons, and U unique supports can require O(U²) comparisons.
Counts are finite under configured bounds, not a linear-time guarantee. See
[scaling measurements](scaling.md) before selecting larger host capacities.
