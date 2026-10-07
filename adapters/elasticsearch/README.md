# Elasticsearch lexical bridge

The host supplies Client, metadata codec and optional tokenizer. This adapter emits
search DSL; it does not provide a native driver, retries, DDL or certify remote
scope enforcement. Host callbacks must remain stable and support concurrent calls.

Construction owns SearchFields and a deep copy of SynonymMap, including nested
slices. Each successful query tokenizes exactly once; one synonym expansion drives
both empty-query admission and the wire query. Subsequent caller edits to the
configuration do not change store behavior. There is no hidden tokenization rerun.

Stored fields are projected through the declared schema before custom Decode.
Present values are canonical string, bool, int64 and finite float64. Omitted whole
attributes normalize to nil; custom codecs accept nil and empty maps equivalently.
Unknown service fields/content are separated from declared metadata. Present
null/wrong-kind values are rejected rather than treated as omission. Local DSL and
callback tests certify this bridge contract, not every host driver's server-side
predicate enforcement. Hosts separately certify tenant pairs, omission predicates,
exact integer identities above 2^53 and unsupported publication profiles.
