# T09 completeness — independent acceptance

Baseline HEAD: `528164b711ef219b510a6056ae0335ba1ff0df7f`. Reviewer did not implement this change and did not read the correctness report. Verdict: **ACCEPTED — 4/4 criteria = 100%**. Assigned source coverage: **2/2 = 100%**. No unresolved T09 completeness finding. Acceptance is scoped to T09, not the whole remaining backlog.

| Criterion | Status | Evidence |
|---|---|---|
| T09.C01 | Complete | Eq/In/numeric orders wrap each atomic comparison with COALESCE(..., FALSE), Neq negates normalized Eq; shared SQL walker combines normalized children under NOT/AND/OR. All four kinds exercise Eq/Neq/In and their negations; int/float additionally exercise all four orders and negations. Seven cross-field compounds exercise omission combinations. String/bool ordering remains rejected by condition validation. |
| T09.C02 | Complete | Shared fixture has every 3^4 absent/equal/unequal combination (81 records), 56 predicates, and 12 distinct malformed admission samples. Independent fixed truth oracle asserts every core result. Null/wrong-kind/invalid UTF-8/nonfinite values reject before DB callbacks; whole nil/empty maps are lawful. Exact bigint neighbors 9007199254740992 and 9007199254740993 survive predicates and roundtrip. Existing integer bounds/transport and identifier-validation tests are retained. SQL-shaped string with quote/comment/drop syntax is passed as a membership parameter in the real profile without altering SQL/table state. |
| T09.C03 | Complete | Independently executed actual PostgreSQL adapter Upsert/Retrieve/DeleteByFilter profile. Native server PREPARE executes the adapter SQL, not a fake evaluator. RETURNING observes actual deleted IDs. All 56 query ID sets and 55 actual deletion ID sets match core; remainder is verified after each accepted deletion. Empty deletion is the explicitly rejected API exception. Table ragy_t09_15985 was process-specific and dropped by cleanup. |
| T09.C04 | Complete | Independent fresh normal race tests PASS three packages, integration-tagged lint reports 0 issues, real profile PASS (160.244s), version/corpus/command evidence below. Missing environment test fails immediately rather than SKIP. |

| Assigned source | Status | Coverage |
|---|---|---|
| Master F08 | Complete | Original category!=x, NOT(Eq), NOT(In) absence disagreement is fixed at positive leaves before composition, for query and delete. Contract and actual service corpus establish the replacement truth table. |
| reviews/storage.md S1 | Complete | Same defect and full AAA expectation covered by four scalar kinds, all legal scalar operators, nested composition, admitted omissions, exact integers and parameterization; real PG evidence replaces previous SQL-only diagnostic. |

Independent verification on 2026-10-06:

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./filter/... ./contracttest/... ./adapters/pgvector/...` — PASS filter 1.234s, contracttest 1.250s, pgvector 1.236s.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run --allow-serial-runners --build-tags=integration_pg ./filter/... ./contracttest/... ./adapters/pgvector/...` — 0 issues; only existing deprecated exhaustruct warning.
- `RAGY_PG_TEST_CONTAINER=ragy-task20-t09-528164b GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -tags=integration_pg -run '^TestRealPostgresPortable' -count=1 -v ./adapters/pgvector/...` — PASS PostgreSQL 17.11 (Debian 17.11-1.pgdg12+2), pgvector 0.8.7. Corpus=81, predicates=56, malformed=12. Independent full output retained at `docs/task20/acceptance/T09-completeness-pg.log`.
- Independent direct PostgreSQL truth probe against `{}` JSON: negated normalized string Eq = true, negated bigint In = true, float Gt = false, NOT(AND(string Eq,bool Eq)) = true, exact bigint neighbor inequality = true (`t,t,f,t,t`). This corroborates the missing-field and integer mechanics separately from adapter tests.
- `env -u RAGY_PG_TEST_CONTAINER GOCACHE=/private/tmp/ragy-task20-go-cache go test -tags=integration_pg -run '^TestRealPostgresPortable' -count=1 ./adapters/pgvector/...` — expected FAIL with explicit required environment message, no SKIP.
- `git diff --check` — PASS.

Scope limitations: psql is a test-only host transport exercising actual server SQL and pgvector, not certification of every production driver. Malformed remote rows are outside the admitted corpus. No live tenant enforcement, optimization, provider or full backlog completion claim. Parent reported a separate pre-existing root wording-blacklist test failure on unchanged T06 history README; its replacement is assigned T21.C01, and this report does not claim global suite PASS. Owned PostgreSQL container remains available for the other independent acceptance reviewer.

Reviewed SHA256 (T09.md captured before later acceptance/cleanup bookkeeping):

| File | SHA256 |
|---|---|
| `docs/task20/T09.md` | `9f3f1331f54c2341abb11c7b5000652df5bdfac5a607b1aca7fd569be4ff8ca4` |
| `docs/contracts/remediation.md` | `2c11991ebda8865f3a69c5636d3a54aac637008c0654d28b95bf191d4f9664f0` |
| `adapters/pgvector/store.go` | `efb39b98da91c848b2c34a08a4f4f2fdbd44133b84f311d5b77974ecc7374a47` |
| `adapters/pgvector/store_test.go` | `b07a5c7cd9e189a5f130d1b0dcab2149b7e9058c0441ed957069d2848832bd30` |
| `adapters/pgvector/filter_pg_test.go` | `4d9a83d5521f0b22a91f4c598041d90a014bb52a13c6ed68c9d3d0491e0a6fc8` |
| `adapters/pgvector/integer_transport_test.go` | `f5cda3b269eee75d42f8b5bbdffdf07b4355a2fc813b3251b83d1c255be5132d` |
| `adapters/pgvector/integer_metadata_test.go` | `eac5d2157c5dac9081726aebf43fa974e1e6ab326614b8a7535091da3f4f777b` |
| `adapters/pgvector/README.md` | `5fd851de6270b9e5fa705cf0d281bef267a27b2c10b5c348777aeef0321816e0` |
| `contracttest/filter_parity.go` | `e9d840603ee41c00af0c790ef5018671bfb249c9f74035e17ae29f677901a277` |
| `contracttest/filter_parity_test.go` | `f0b6de164ef980dbe655e1171b1272287f792009f89396058171e0c9bfa148b4` |
| `filter/filter.go` | `02e610e7f9ec3bb9bfa3002e3f86313ddca69a41564878534d5c0c83cf05c407` |
| `filter/match.go` | `ca33153f14abb918b282cee0ad2bce51f08eeaae812f2cf000dca4c75dcbd5c1` |
| `filter/rawattributes.go` | `dadbf117ddd574afbd9167ca99b3839e41502e5dce785fc3e7b7faa1f52b980a` |
