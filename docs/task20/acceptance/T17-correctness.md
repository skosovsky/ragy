# T17 independent correctness acceptance

Baseline `e6578d18eeab6e8030727025f5cd57e974afa619`. Nonimplementing reviewer; only acceptance report and uniquely prefixed logs written. **PASS: no unresolved correctness findings on the final revised candidate.** All five criteria and 21 assigned decisions inspected. Earlier candidate was revised to cover planned Neo4j predicates and native scoped PG bindings; final tests below prove that revision.

## Audit

| Criterion | Correctness evidence |
|---|---|
| T17.C01 | PG `decodeStoredMeta` always canonicalizes schema attributes before custom Decode, including empty wire. RawAttributes UseNumber is preserved until schema normalization; present null is rejected before callback. Filter nil/zero/unfinalized builders classify ErrInvalidArgument. Public scalar constraints match constructor kinds; private parity helpers migrated. |
| T17.C02 | ES deep clones each synonym slice and SearchFields. One queryTokens invocation supplies both admission and render; fixture mutates nested inputs and changes second tokenizer result. No second tokenization/framework retained. |
| T17.C03 | Validated single ASCII table identity is quoted centrally and used by all SELECT/INSERT/DELETE paths. Bound values remain separate. Batch capacity is host responsibility; one-statement write retained. Distance ties explicitly unspecified. Rows.Err observed for query/find; deferred Close documented cleanup-only, tested separately from ordinary prefix failures. |
| T17.C04 | Neo4j rejects Options.Filters, Plan.Filters and Plan.Ranges before Runner, keeps graph-specific conditions, validates document projection separately from full administrative snapshots. Final delivery takes nil ordinary callback error, then joins private callback through errors.Is only on protected delivery. Existing T10 success/empty/partial/private-runner cancellation fixtures passed independently. PG negative/unrepresentable and Qdrant negative affected counts fail protocol, exact zero preserved. |
| T17.C05 | Independent root four-package race and each changed adapter module race passed. Independent native PG canonical/mixed-case and exhaustive portable query/delete profiles PASS. Additional actual scoped Binding tenant pairs, omission, conflicting optional predicates and pinned-before-I/O native profile PASS. Certification remains limited to the supplied bridge/profile. No ES/Qdrant/Neo4j native certification or SKIP=PASS. |

## Assigned source decisions

D51 and storage:01/05/17: canonical/empty codec contract, admitted lookup and builder classification inspected. D52 and storage:02/03: owned configuration and token-once dispatch inspected. D53 and storage:04/09/10/13: quoted single identifier, explicit host capacity, unspecified ties and cleanup-only Close with authoritative Err inspected. D54 and storage:06/07/08/11/12/14/15: Runner honesty, rejected generic filters, projection boundary, raw administration and exact affected count policies inspected. storage:16: exact built-in constraints and two compiled parity helper consumers inspected. All 21 decisions have technically justified outcomes; no speculative native transport, automatic migration or alias framework introduced.

Retained T12 contracttest fixtures now reject nil predicates at Build and inspect FusionFailureError observations; production composition is unchanged. The owned PG runtime is never stopped by this reviewer. Existing production-doc blacklist issue belongs to T21.C01; unrelated task18 iCloud duplicate untouched.

## Independent commands

- `go test -race -count=1 ./filter ./retrieval ./contracttest ./documents`: PASS, T17-correctness-root.log.
- In each ES/Neo4j/PG/Qdrant module, `go test -race -count=1 ./...`: PASS, T17-correctness-<module>.log.
- `git diff --check`: PASS.
- First native attempt failed before tests because default Homebrew Go1.27.1 reported missing encoding/csv. Second Go1.26.5 attempt failed on sandbox access to shared Go cache. Neither is a product failure or counted PASS. Final attempt uses local Go1.26.5 and a fresh owned /tmp cache, same isolated container, both native profiles; both final profiles passed (81 records,56 predicates, malformed admission; PostgreSQL17.11/pgvector0.8.7). Preserved logs: T17-correctness-live-pg.log, T17-correctness-live-pg-126.log, T17-correctness-live-pg-final.log.

Final revised checks:

- Neo4j entire module `go test -race -count=1 ./...` under Go1.26.5 and owned /tmp cache: PASS, T17-correctness-neo4j-revised.log. Revised Options.Filters/Plan.Filters/Plan.Ranges pre-Runner AAA fixtures and all T10 private delivery regression cases passed. One attempted repeat with shared default cache failed before testing (T17-correctness-neo4j-final.log); not counted.
- Actual native `TestRealPostgresScopedTenantPairsAndPinnedAdmission`: PASS, T17-correctness-live-pg-scoped.log. Bound adjacent exact tenant identities above2^53 each return only their own document; omitted tenant excluded, contradictory optional tenant intersects to empty, unsupported pin produces protected ErrUnsupported before DB call.
- Independent full native portable/canonical profile: PASS, T17-correctness-live-pg-final.log, exit0. Earlier unrestricted fixture evidence is correctly distinguished from actual Binding scope profile in final T17.md and PG README.

No performance optimization or unsupported native certification is asserted. Parent owns runtime cleanup. Current conformance migration is consistent with accepted T12 semantics and passed independent contracttest race. Parent root expanded race/lint evidence is supplementary; no all-root success claimed.

## Final SHA256 manifest

Implementation, tests and current documents include all changed task-owned source/test/docs; execution bookkeeping and generated logs excluded. Relevant unchanged gate/normalization/native-parity evidence also frozen below.

```text
82a47260be4f59d725cea9e8877b9809682dc0ed11ab5f87ab929aafa10f5419  adapters/elasticsearch/README.md
02e8399178e822ec096341f6ce85278b19d2419147ffcf63cd472c8a6fac2d41  adapters/elasticsearch/config_contract_test.go
4fc3346749008049f0f47a2f30e5e52515e56a71951f2019ae8900db6b97a9df  adapters/elasticsearch/elasticsearch.go
f06e95deddfd45c49625e1e8070889e5803135be4534e45a0e4ceb80b48f023d  adapters/neo4j/README.md
86637d3587cee2c9de33ec16115656177533d0e3c50618876125b7220d6f1bb6  adapters/neo4j/delivery_test.go
33891515b20f9784a3eef7e06bb3f59a924a16bbd399531f3f476041747b02cd  adapters/neo4j/filter_contract_test.go
3d7f961983063cde8cf9c1c21c93df40e92e9b515716bcc2ef59a77ec6635cf0  adapters/neo4j/neo4j.go
46fdd66dcf2fa317b52b3c7213adcdbe712d80b7d62e3e349aaab7bb59b264b7  adapters/pgvector/README.md
9edd22bc55becc84ce517e33f0fb38fd2bf5106ed90bfa401294fb01a92cc651  adapters/pgvector/bridge_pg_test.go
91786b57173102186d3165537a682dbf1082a43170d62b837f0952f1f87d1168  adapters/pgvector/codec_contract_test.go
b07e4a2a3394d05fa9859a9a6b6737b65264ee2ff250696fef39433d36e22f2f  adapters/pgvector/count_contract_test.go
4d9a83d5521f0b22a91f4c598041d90a014bb52a13c6ed68c9d3d0491e0a6fc8  adapters/pgvector/filter_pg_test.go
a5754f3f48145fa5bd2b66a9ec80e675bc245f7216537e868a82af82faca8a74  adapters/pgvector/rows_contract_test.go
2b06ae120f3c5fe02c4b4aa5a66a4e3df9540282c55f68d22d0075753c2f45ed  adapters/pgvector/scope_pg_test.go
a317625239d67be2a67ec1d17137d1792bed1199bb08503d2a2545d2d2be231b  adapters/pgvector/store.go
28eb5c8dfd0fda469fa7f4e06a66f9304406a24df7f01b675c6f720635204a89  adapters/pgvector/store_test.go
d78cedddd68920c2f9a378b2ca582c0f7a2f500fff782b8e5cdb26790f763b18  adapters/qdrant/README.md
9679f98a5d1b4044eb53995606d5e66314e2e46fb56638c82b082256a3cbab44  adapters/qdrant/count_contract_test.go
6dffc0b2fb9593169010c6d9bad95d2794f3a3af8e9546f4e4bc531c656b9e3e  adapters/qdrant/store.go
6ad4459cd80f5d621ecb0d3872b08372a92f07805bb4d5370ffc987f0dc1f84f  contracttest/filter_parity.go
268db0cb98bff4ab03205916933d31a4c98cf497d6a18ebea426223c14e59850  contracttest/pipeline.go
1722028d652725463ee9436f8c507bff4edff6235aa12a8f6fbb11ef9b98184b  contracttest/pipeline_test.go
cf0e9bdcd74ef00ae286ba7a10ededc8c9c3c7364fed40ad86f736376daea594  docs/task19/capabilities.md
7cb36505075ddc41bf1a588e17b9e9cb0848b65df261a9520b0e3ad2a69625b4  docs/task20/T17.md
22d18fec34041d396761dfc360dae05ec2c40077568090b20315df6121d5c728  documents/README.md
8a2e652af6ef2ed7588184bfe8533240151438aebb9225f839fbc61564902225  documents/documents.go
a6755b651d552618a1a5374fdbc39b834bddf240a4cbfce1dd069fbead979436  filter/README.md
b28092bf28a886a6427d2585b20c19d0c365dcef531042e374029ac24761d2e9  filter/builder.go
23446b78d4474ceb88bb45b9473eaa54c4399f48985773c6ac98f9849411341b  filter/builder_contract_test.go
6c2cf761385a73de54728fb64c469aea16caf3bdaa293740e0cbb125a13c42d8  filter/filter.go
ca33153f14abb918b282cee0ad2bce51f08eeaae812f2cf000dca4c75dcbd5c1  filter/match.go
dadbf117ddd574afbd9167ca99b3839e41502e5dce785fc3e7b7faa1f52b980a  filter/rawattributes.go
0ffbb31ad96c96548e5c6840b5eda9e9c974d25ff246bd452781b956a8693913  internal/readfailure/callback.go
feab1dd9a9da0e15ff3bc96e4a264ba069ad6b36917ccf3122d4aa67de6cae19  retrieval/access.go
27634ea00548d793818f8639d8a923a2e4fd9af80bd7830f06f3a7ef1916e745  retrieval/codec.go
```
