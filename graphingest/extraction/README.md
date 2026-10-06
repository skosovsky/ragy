# Bounded source extraction adapter

`Adapter.Extract` connects one injected typed model client to source-supported graph
mentions. It requires a shared budget ledger, host pricing/input-token counting,
model-free ontology validation, BYOT attribute/access cloners, metadata schema and
original snippet admission. The client must enforce its supplied input/output limits
and execute exactly one call without hidden retries. Actual provider transport is
separate; the contract tests use a scripted client and do not prove model quality.

Input snippets contain immutable original/derived mapping, host ontology namespace
and typed access metadata. Scope/schema compatibility is checked first. The complete
mapping/byte/support bounds are validated; mandatory metadata scope and source/quote
admission run before model dispatch. Host AdmitSnippet must authorize every mapping
support and verify retained text/representation. Publication alone is not snippet
authorization. When constructing input snippets, load retained text through the
scoped source reader; this adapter cannot control earlier application payload loads.

Model input contains only ontology/config identity, ordinal/text pairs and reserved
token limits. It contains no read binding, access metadata, source references or
credentials. Token counter and model receive independent input slices. Model output
contains local mentions and snippet ordinals; core validates kind/attributes,
endpoint references, bounds and evidence indices, then derives supports and
namespace from admitted snippets. Mixed/unknown namespaces remain ambiguous for
host identity resolution. The model cannot choose canonical source revision,
permissions or entity merge policy. Structural citation association is not proof
that a model assertion is true; external quality evaluation remains required.

Pricing and exact provider input-token counting run before atomic reservation.
Required unknown pricing refuses dispatch; advisory unknown accounting conservatively
retains full reservations. Known token overrun is rejected even with unknown cost.
Actual usage settles on client failure; errors do not cause retry. Earlier parent/
attempt deadline, injected clock expiry and host revocation suppress all payloads.
The host-owned ledger retains accounting even when the result is suppressed.

Pass returned typed mentions through namespace/alias resolution and explicit graph
materialization before lifecycle staging/publication. No model/source engine,
provider credential store, background worker or universal ontology is introduced.

Model and settlement causes remain inspectable when the post-call deadline/freshness gate also fails. Accounting settles once after expiry; no failed or protected model output is projected or retried.
