# pgvector store profile

`Config.Space` is mandatory and names the host-owned model revision, preprocessing configuration, vector space, dimension and metric. Both query vectors and records must match it exactly, including when dimensions happen to agree. Validation runs before database I/O. `Space()` returns the configured declaration; it does not inspect or certify existing rows.

This adapter implements cosine only. It queries `vector <=> $1` in ascending distance order and returns `1 - distance` with `dense.cosine` semantics. Negative cosine scores remain negative; vectors are not normalized by the library. Other metrics return `ErrUnsupported` at construction. This follows the [pgvector operator contract](https://github.com/pgvector/pgvector#distances), verified 2026-10-06.

Before adopting the breaking contract, inventory the old table's embedding model, revision, preprocessing and vector dimension. Re-embed into a separate table if any identity component changed or cannot be established; provision its column dimension and cosine operator class explicitly. Configure the new profile only after those host checks. A new declaration cannot make unknown old vectors compatible. No migration, data deletion or service configuration is performed by the adapter. Metadata and SQL transport remain host-provided.

## Portable filter semantics

Optional absent fields follow the core two-valued truth table: Eq/In/numeric orders
are false, Neq is true, and NOT/AND/OR compose those booleans. SQL normalizes each
positive leaf with `COALESCE(..., FALSE)`; Neq negates normalized Eq. Query and
`DeleteByFilter` share the renderer. Empty read conditions match all rows, while
empty deletion conditions are rejected. Strings/bools support Eq/Neq/In; order
supports int/float only. Integers retain exact bigint comparison, including adjacent
values above 2^53. Values are bound parameters; table/field identifiers are validated.

A host codec/store must admit normalized metadata: valid UTF-8 string, bool, int64,
finite float64 or omitted fields. Present null, malformed/wrong-kind values are
rejected rather than interpreted as missing. A whole nil/empty attribute map is
lawful. Remote rows must satisfy this schema; no predicate authorizes malformed
storage records by treating a failed cast as omission.

`contracttest.PortableFilterParity` supplies an exhaustive 81-row absent/equal/unequal
corpus and 56 predicates across all four kinds, numeric orders and nested NOT/AND/OR.
Default tests retain exact integer and injection checks. The `integration` profile
executes actual Upsert/Retrieve/DeleteByFilter on a process-unique table via a host
psql bridge. PostgreSQL native PREPARE/EXECUTE binds the unchanged adapter predicates;
RETURNING instrumentation observes actual deleted IDs. It compares queries/deletions
with core matcher IDs and verifies remaining rows and metadata roundtrips. This
certifies that SQL/profile, not every host driver or malformed external corpus.

Each test starts its own Docker PostgreSQL container from a pinned pgvector image,
waits for TCP readiness, installs the extension and registers cleanup with t.Cleanup.
No external container, label or environment variable is required.

```sh
cd adapters/pgvector
GOWORK=off go test -race -tags=integration -run '^TestIntegration' -count=1 -timeout=10m ./...
```

A missing Docker executable, unavailable daemon or failed startup fails the profile;
it never skips. Cleanup removes only the test's uniquely named container and volume.
Production transport remains the host's responsibility; no SQL
driver dependency or automatic server startup was added to this module.

## Host bridge outcomes and identifiers

The existing single ASCII identifier grammar is retained; table names are always
quoted, preserving case and keywords. Schema-qualified names are unsupported.
Values remain parameters. Hosts provision tables and bound Upsert batch sizes for
server/driver parameter and statement capacity. Upsert is one SQL statement; no
implicit chunking or partial batch commits are introduced.

Distance-only ordering leaves equal-distance ties unspecified. Ranks reflect actual
backend order; callers must not assume reproducible ID tie order or ANN recall.
This retains the query shape usable by a host's chosen ANN planner/index profile.

Custom Decode receives schema-canonical scalars (including exact int64) and is
called even on empty wire attributes. Omitted whole attributes normalize to nil;
codecs accept nil/empty equivalently. Present null/wrong-kind values fail admission.
JSONCodec may also accept raw wire attributes and normalize them defensively.

Rows.Err reports query/iteration outcome before delivery. Rows.Close belongs to
cleanup only in this bridge profile; its returned diagnostic is not interpreted
as a late query outcome. Host drivers must surface actual query failures through
Err and observe cleanup diagnostics themselves. Raw FindByIDs/Delete are explicit
administration without bound retrieval scope/publication guarantees.

RowsAffected must be exact, nonnegative and representable as a Go int. Invalid
counts fail protocol. Unknown/asynchronous outcomes require host observation or an
explicit unsupported/error result; never infer affected count from submitted IDs.
No transport, retry, migration or remote profile inspection is supplied here.

The additional `TestIntegrationPostgresCanonicalCodecAndQuotedTable` profile creates a
process-unique mixed-case table and asserts actual Upsert/Retrieve/FindByIDs/delete,
custom-codec canonical int64 >2^53, empty attributes and exact deleted count.
Use the `integration` tag; select these profiles with
`-run '^TestIntegrationPostgres(Portable|Canonical|Scoped)'`. Each profile owns an isolated
container, including when run directly through Go.

`TestIntegrationPostgresScopedTenantPairsAndPinnedAdmission` uses actual host-scoped
Bindings and mandatory tenant predicates on adjacent int64 identities above2^53.
It checks both tenants, omitted metadata, a conflicting optional predicate, and
unsupported pinned publication rejection before DB invocation. This certifies the
explicit local bridge/server test profile, not arbitrary host authority policies.
