# Metadata transport acceptance

Library-owned metadata boundaries were inventoried from production adapter imports/types,
not inferred from provider names. The contracts below describe the shipped ports. A custom
Client/DB/Runner owns its external protocol implementation; these tests do not assert live
service, database engine, driver or network behavior.

| Boundary | Actual path exercised | Evidence / limit |
|---|---|---|
| Elasticsearch Hit.Source | raw JSON into RawAttributes → Store.Retrieve → schema projection → JSONCodec.Decode | four adjacent/boundary integers, four invalid numbers, exact serialized terms membership; injected Client |
| Qdrant Point.Attributes | raw JSON into RawAttributes → Retrieve projection; typed metadata → Upsert Encode/canonicalization → captured attributes JSON → Decode | same read cases, four write roundtrips, seven malformed custom codec cases with zero writes, exact InCondition values; injected Client |
| pgvector stored attributes JSON | Rows.Scan bytes → decodeStoredMeta → JSONCodec.Decode; Upsert arguments contain canonical JSON bytes | same read cases, four write roundtrips, seven malformed custom codec cases with zero writes, exact SQL membership arguments; injected Rows/DB |
| Neo4j Runner[TMeta] | typed graph snapshots; library graph metadata normalization preserves typed integers | typed BYOT boundary, no library-owned JSON driver decode; graph/meta_integer_test.go and module suite |
| Root persistent dense/tensor catalogs | actual durable Stage/Publish/files/new adapter → retained schema normalization → matching/typed metadata | external integer-storage suite; separate actual filesystem evidence |

The other shipped adapters provide embeddings, reranking, models, PDF parsing or telemetry;
they do not introduce another RawAttributes/MetadataCodec storage decoder. Structured model
outputs and usage envelopes are typed/protocol-validated separately. Optional external
clients must decode metadata JSON directly into RawAttributes or typed metadata, preserving
numbers before schema normalization. A float64-decoded identifier cannot be reconstructed
by this library; historical damage requires source reindex.

Declared metadata schema is scalar. Membership fixtures exercise arrays of integer filter
values transported without a float intermediary; this does not add array-valued mandatory
attributes or broaden the Eq/In/And authorization profile. Explicit float schema fields
retain their floating-point semantics. Invalid codec output is rejected before write I/O,
regardless of the custom codec's internal normalization choices. Normalized output is an
owned snapshot; a transport must not gain authority to adopt sources or publish snapshots.

49 new wire cases comprise 9 Elasticsearch, 20 Qdrant and 20 pgvector cases. Test names and
actual command output are retained in results/*-integer-wire-test.txt. Before-fix injected
transport reproductions are separate from final passing logs. Full make lint/test evidence
is retained in results/integer-wire-full-*.txt. No live external service is claimed.
