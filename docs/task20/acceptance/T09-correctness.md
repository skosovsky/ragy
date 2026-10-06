# T09 correctness acceptance

Verdict: **PASS**. No actionable correctness findings in the reviewed T09 scope. Independent acceptance, no implementation edits and no other reviewer report read.

Baseline HEAD: `528164b711ef219b510a6056ae0335ba1ff0df7f`.

Reviewed the F08 master finding, storage S1, T09.C01–C04 and the approved portable truth contract. Positive comparison/membership leaves now normalize SQL NULL to false before grouping; Neq negates normalized Eq. Shared query/delete rendering, parameter ordering and validated identifier policy remain intact. Integer values retain int64/bigint and JSON UseNumber transport. Null, wrong-kind, nonfinite and malformed UTF-8 values are rejected through schema admission; arbitrary malformed remote records remain outside the admitted corpus. The UTF-8 change enforces the existing normative domain without repairing invalid bytes. Empty deletion still fails before DB I/O.

## Independent verification

- Fresh `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./filter/... ./contracttest/... ./adapters/pgvector/...`: PASS all three packages.
- Fresh `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run --allow-serial-runners --build-tags=integration_pg ./filter/... ./contracttest/... ./adapters/pgvector/...`: 0 issues.
- Actual independent required profile: `RAGY_PG_TEST_CONTAINER=ragy-task20-t09-528164b GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -tags=integration_pg -run '^TestRealPostgresPortable' -count=1 -v ./adapters/pgvector/...`: PASS, package 160.489s. PostgreSQL 17.11 / pgvector 0.8.7. Corpus 81, predicates 56, malformed samples 12; exact query IDs, 55 actual delete ID sets and their remaining complements match core. Empty delete intentionally returns ErrInvalidArgument. No SKIP. Independent full output: `/private/tmp/ragy-t09-correctness-pg.log`.
- Independent overlay-only adversarial actual PG test: `RAGY_PG_TEST_CONTAINER=ragy-task20-t09-528164b GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -overlay=/private/tmp/ragy-t09-correctness-overlay.json -tags=integration_pg -run '^TestReviewT09' -count=1 -v ./adapters/pgvector/...`: PASS five added query/delete cases, package 23.349s. Separate process-unique table and a schema-normalized three-row missing/MinInt64/MaxInt64 corpus. Cases exercise double NOT, Neq at MaxInt64, nested NOT(OR(In(min,max), Gt(negative float))), NOT(Lt(max)) and an injection-shaped string (`'); SELECT pg_sleep(60); --`). Exact metadata roundtrip and actual returned/deleted/remaining IDs were checked. Both review tables were dropped by cleanup; repository implementation unchanged.
- Required profile without environment: `env -u RAGY_PG_TEST_CONTAINER GOCACHE=/private/tmp/ragy-task20-go-cache go test -tags=integration_pg -run '^TestRealPostgresPortable' -count=1 ./adapters/pgvector/...`: expected exit 1 with `integration_pg requires RAGY_PG_TEST_CONTAINER; no SKIP`, confirming unavailable runtime cannot be counted as PASS.
- `git diff --check`: PASS. Existing integer boundary, malicious identifier and parameter binding tests inspected and retained.

The test bridge submits actual adapter SQL through native PREPARE/EXECUTE and observes affected IDs through RETURNING. It verifies PostgreSQL semantics on the admitted corpus; it does not certify every production transport, malformed external database row or other backend. Parent-reported broad root documentation blacklist failure concerns an unchanged T06 history README and is assigned T21.C01; this report makes no all-repository PASS claim.

## Reviewed substantive file SHA256

| File | SHA256 |
|---|---|
| `docs/contracts/remediation.md` | `2c11991ebda8865f3a69c5636d3a54aac637008c0654d28b95bf191d4f9664f0` |
| `adapters/pgvector/store.go` | `efb39b98da91c848b2c34a08a4f4f2fdbd44133b84f311d5b77974ecc7374a47` |
| `adapters/pgvector/store_test.go` | `b07a5c7cd9e189a5f130d1b0dcab2149b7e9058c0441ed957069d2848832bd30` |
| `adapters/pgvector/filter_pg_test.go` | `4d9a83d5521f0b22a91f4c598041d90a014bb52a13c6ed68c9d3d0491e0a6fc8` |
| `adapters/pgvector/README.md` | `5fd851de6270b9e5fa705cf0d281bef267a27b2c10b5c348777aeef0321816e0` |
| `filter/filter.go` | `02e610e7f9ec3bb9bfa3002e3f86313ddca69a41564878534d5c0c83cf05c407` |
| `filter/rawattributes.go` | `dadbf117ddd574afbd9167ca99b3839e41502e5dce785fc3e7b7faa1f52b980a` |
| `filter/match.go` | `ca33153f14abb918b282cee0ad2bce51f08eeaae812f2cf000dca4c75dcbd5c1` |
| `contracttest/filter_parity.go` | `e9d840603ee41c00af0c790ef5018671bfb249c9f74035e17ae29f677901a277` |
| `contracttest/filter_parity_test.go` | `f0b6de164ef980dbe655e1171b1272287f792009f89396058171e0c9bfa148b4` |
