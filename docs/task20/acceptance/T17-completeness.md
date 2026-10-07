# T17 independent completeness acceptance

Baseline `e6578d18eeab6e8030727025f5cd57e974afa619`. Nonimplementing reviewer; no implementation edits and no counterpart report used. **PASS: 5/5 criteria = 100%; 21/21 assigned source decisions = 100%.** Final revised candidate independently checked after the planned-predicate and bound-native coverage corrections; old candidate acceptance is superseded. No missing required scoped check, blocked criterion or SKIP counted as PASS. Reviewed actual current diff, normative T17.md, original master D51–D54, original storage role items01–17, and code/tests/current bridge guides independently.

| Criterion | Result | Evidence |
|---|---|---|
| T17.C01 | выполнено | All21 independently reviewed dispositions below. PG NormalizeAttributes precedes Decode even on empty wire; canonical custom-codec >2^53 and empty tests PASS. Existing ES/Qdrant normalize before Decode. MetadataCodec and guides agree on present canonical kinds and nil/empty equivalence. Filter admitted-lookup/omission contracts, exact constraints and builder sentinel AAA tests agree. |
| T17.C02 | выполнено | ES clones SynonymMap and nested slices plus fields. One queryTokens expansion feeds both empty admission and render. Caller mutation/stateful fixture asserts one call and original query/fields. Independent ES race PASS. |
| T17.C03 | выполнено | Validated single ASCII table is double-quoted in store state and used by every SQL path; wire keyword/case tests and actual mixed-case native operations PASS. Upsert remains one bound-value atomic statement with host capacity responsibility. Distance-only ties and cleanup-only Close versus authoritative Rows.Err documented; fixture checks cleanup and ordinary-prefix failure. |
| T17.C04 | выполнено | Neo4j Options.Filters, Plan.Filters and Plan.Ranges rejected before Runner; graph-domain options remain. Projection versus complete snapshot fixture PASS. Retained T10 final-delivery cancellation/private-cause tests PASS after local error-merge correction. Exact nonnegative count contract and negative-report checks; raw admin distinct. |
| T17.C05 | выполнено | Independent root scoped and all four changed-module GOWORK=off race PASS; current nested conformance consumer PASS. Actual isolated PG canonical/mixed-case, full portable query/delete parity, and scoped adjacent-int64 tenant pair/pinned-before-I/O PASS, process-unique tables cleaned by test. Host wire tests do not certify live ES/Qdrant/Neo4j and docs make no such claim. No optimization planned or claimed. |

## Independent checks

- Root `GOWORK=off go test -race -count=1 ./filter ./retrieval ./documents ./contracttest`: exit0, [root log](T17-completeness-root.log).
- ES/Neo4j `GOCACHE=/tmp/ragy-t17-completeness-cache GOWORK=off go test -race -count=1 ./...`: exit0, [ES](T17-completeness-es-fresh.log), [Neo4j](T17-completeness-neo-fresh.log).
- Qdrant `GOWORK=off go test -race -count=1 ./...`: exit0, [Qdrant](T17-completeness-qdrant.log).
- PG full wire `GOCACHE=/tmp/ragy-t17-completeness-cache GOWORK=off go test -race -count=1 ./...`: exit0, [wire](T17-completeness-pg-wire.log).
- PG actual `GOCACHE=/tmp/ragy-t17-completeness-cache GOWORK=off RAGY_PG_TEST_CONTAINER=ragy-task20-t17-pg go test -race -tags=integration_pg -count=1 -run '^TestRealPostgres(Portable|Canonical)' -v ./...`: exit0, [native profile](T17-completeness-pg-fresh.log). PostgreSQL17.11/pgvector0.8.7. Mixed-case table RagyT17_13065 with canonical custom-codec calls4/delete1; separate table ragy_t09_13065 for 81records/56predicates/12malformed attributes. Query and actual DeleteByFilter sets match core, including omission and exact adjacent int64 >2^53. This portable corpus alone does not certify bound tenant intersection; final independent actual scoped profile below separately proves mandatory tenant binding. Container retained for root cleanup; no stop operation from this reviewer.
- `examples/conformance`, `GOCACHE=/tmp/ragy-t17-completeness-cache GOWORK=off go test -count=1 ./...`: exit0, [current consumers](T17-completeness-conformance.log); graph/recipe/tensor and remaining packages PASS.
- `git diff --check`: exit0. Independently read root and four changed-module final lint logs: all0issues. Earlier failed lint/race attempts remain recorded; those failures are not counted as success. No counterpart acceptance report used.

Initial independent ES/Neo4j default-cache attempts hit sandbox permission failures; initial PG attempt reported stdlib encoding/csv unavailable. These failed attempts remain in T17-completeness-{es,neo,pg}.log. Dedicated isolated /tmp cache resolved all three without implementation changes. Fresh successful runs above are authoritative. Root production-doc blacklist failure remains deferred T21.C01; this acceptance does not claim all-root PASS. No live ES/Qdrant/Neo4j certification or final T22 validation claimed.

Current conformance migration matches accepted T12 behavior: nil predicates rejected at Build, FusionFailureError keeps original observations and no fabricated fallback result; only harness fixtures changed, no production composition algorithm. Removed obsolete fields have no remaining current consumer. Independent root harness and nested consumer tests compile and PASS.

## Assigned source decisions

| Source | Result | Evidence / rationale |
|---|---|---|
| D51 | выполнено (change) | PG canonical int64/empty Decode; scalar/null admission and two-valued omission contract; builder sentinel probes PASS. |
| D52 | выполнено (change) | Deep-cloned synonyms/slices/fields and exact single tokenizer/expansion wire reuse; mutation/stateful fixture PASS. |
| D53 | выполнено (change) | Validated single identifier always quoted; one atomic host-bounded SQL statement; unspecified ties and cleanup-only Close explicit. |
| D54 | выполнено (change) | Runner-only capability, all generic options/planned filters and planned ranges rejected, projection distinct from full snapshot; honest exact-only counts and certification boundaries. |
| storage:01 | выполнено (change) | decodeStoredMeta normalizes before custom codec, including empty wire; real mixed-case/custom-codec profile PASS. |
| storage:02 | выполнено (change) | Deep copy includes nested synonym slices and SearchFields; mutation test proves constructor ownership. |
| storage:03 | выполнено (change) | One queryTokens call and exact expanded tokens passed into render; stateful tokenizer calls=1 assertion. |
| storage:04 | выполнено (change) | Single ASCII identifier validation retained, table quoted for all SQL operations; keyword wire and actual mixed-case fixtures PASS. |
| storage:05 | выполнено (retain) | MatchIR admitted-lookup precondition explicit; malformed present values rejected; 81-row/56-predicate real query/delete parity PASS. |
| storage:06 | выполнено (retain) | Runner and unrestricted-current-only contract retained; scoped/pinned denial tests remain; no native driver claim. |
| storage:07 | выполнено (change) | Options.Filters, Plan.Filters and Plan.Ranges return ErrUnsupported before Runner; three-channel fixture PASS, graph NodeFilter/EdgeFilter retained explicitly. |
| storage:08 | выполнено (change) | Retrieve projects ID/content only; invalid graph-label projection succeeds while Traverse rejects full snapshot; fixture PASS. |
| storage:09 | выполнено (change) | Upsert still builds one four-parameters-per-record statement and calls Exec once; host capacity responsibility documented without chunking. |
| storage:10 | выполнено (change) | Distance-only ORDER BY preserved; backend order/rank and unspecified equal-distance ties documented; ANN recall not promised. |
| storage:11 | выполнено (retain) | FindByIDs/Get/Delete remain explicit raw administration, separate from read bindings; no service IAM retrofit. |
| storage:12 | выполнено (change) | Exact-only count contract; PG negative/unrepresentable and Qdrant negative reject protocol; zero/five retained, no inferred input length. |
| storage:13 | выполнено (change) | Visible defer Close retained; Rows.Err authoritative; independent rows fixture checks cleanup-only error and ordinary prefix outcomes. |
| storage:14 | выполнено (retain) | Host owns transport/retry/migration/provisioning; constructors perform no automatic remote DDL/re-embedding/profile inspection. |
| storage:15 | выполнено (retain) | Wire versus live certification explicitly separated. Actual PG SQL omission/exact-integer parity plus separately bound tenant-pair, conflicting optional filter and pinned-before-I/O profiles PASS; no live ES/Qdrant/Neo4j claim. |
| storage:16 | выполнено (change) | Underlying-type constraints removed; two private conformance generic helpers migrated; GOWORK=off current consumer tests PASS. |
| storage:17 | выполнено (change) | Nil/zero/unfinalized builder errors wrap ErrInvalidArgument; AAA sentinel fixture PASS; schema name/kind validation remains. |


## Final candidate revision verification

Direct planned metadata channels are explicit unsupported inputs to Neo4j, including Plan.Ranges. Inspected hasGenericPredicates against Request/PlannedQuery declarations: Options.Filters, Plan.Filters and any Plan.Ranges all reject before Runner. All three fixtures assert ErrUnsupported, zero results and zero callbacks; no graph-domain reinterpretation. Revised full Neo4j `GOCACHE=/tmp/ragy-t17-completeness-cache GOWORK=off go test -race -count=1 ./...`: exit0, [final log](T17-completeness-neo-final.log).

Native portable predicate parity uses unrestricted binding and is not itself a scoped-tenant certificate. Final additional `GOCACHE=/tmp/ragy-t17-completeness-cache GOWORK=off RAGY_PG_TEST_CONTAINER=ragy-task20-t17-pg go test -race -tags=integration_pg -count=1 -run '^TestRealPostgresScoped' -v ./...`: exit0, [final scoped log](T17-completeness-pg-scoped-final.log). Actual DB rows have adjacent tenant identities9007199254740992/9007199254740993 and omitted tenant. Both independently constructed access.Scoped mandatory conditions retrieve only their corresponding record; omitted tenant never leaks. Conflicting optional filter yields empty result, demonstrating intersection rather than replacement. Pinned request yields protected ErrUnsupported and db.calls unchanged before actual DB invocation. This revised fixture has actual native DB transport and cleans only its process-specific test table. Reviewed revised PG/Neo4j lint0issues and updated guides/trace rows; no broader live certification claim.

## Current implementation/test/docs SHA256

127 current changed implementation/tests/guides and retained code/profile/nested consumer evidence files are fingerprinted below, including T17.md, scope_pg_test.go and conformance migration. Mutable backlog/plan/traceability journals, acceptance logs/reports and unrelated `docs/task18/correctness 2.md` are excluded. Criteria and all21 trace rows were inspected directly above. Any implementation/test/guide change requires renewed acceptance.

| File | SHA256 |
|---|---|
| adapters/elasticsearch/README.md | `82a47260be4f59d725cea9e8877b9809682dc0ed11ab5f87ab929aafa10f5419` |
| adapters/elasticsearch/config_contract_test.go | `02e8399178e822ec096341f6ce85278b19d2419147ffcf63cd472c8a6fac2d41` |
| adapters/elasticsearch/elasticsearch.go | `4fc3346749008049f0f47a2f30e5e52515e56a71951f2019ae8900db6b97a9df` |
| adapters/elasticsearch/go.mod | `facc262e2110b59b94260c121da211b1dc1a5f41ee70fcb8e7693033ac1099a9` |
| adapters/neo4j/README.md | `f06e95deddfd45c49625e1e8070889e5803135be4534e45a0e4ceb80b48f023d` |
| adapters/neo4j/delivery_test.go | `86637d3587cee2c9de33ec16115656177533d0e3c50618876125b7220d6f1bb6` |
| adapters/neo4j/filter_contract_test.go | `33891515b20f9784a3eef7e06bb3f59a924a16bbd399531f3f476041747b02cd` |
| adapters/neo4j/go.mod | `cc141d479b959392c6343df301702dd05d6d9e5bccf35c1fd864d62cb3423975` |
| adapters/neo4j/neo4j.go | `3d7f961983063cde8cf9c1c21c93df40e92e9b515716bcc2ef59a77ec6635cf0` |
| adapters/pgvector/README.md | `46fdd66dcf2fa317b52b3c7213adcdbe712d80b7d62e3e349aaab7bb59b264b7` |
| adapters/pgvector/bridge_pg_test.go | `9edd22bc55becc84ce517e33f0fb38fd2bf5106ed90bfa401294fb01a92cc651` |
| adapters/pgvector/codec_contract_test.go | `91786b57173102186d3165537a682dbf1082a43170d62b837f0952f1f87d1168` |
| adapters/pgvector/count_contract_test.go | `b07e4a2a3394d05fa9859a9a6b6737b65264ee2ff250696fef39433d36e22f2f` |
| adapters/pgvector/filter_pg_test.go | `4d9a83d5521f0b22a91f4c598041d90a014bb52a13c6ed68c9d3d0491e0a6fc8` |
| adapters/pgvector/go.mod | `399a2c963deace92fda49e9862d63bd9a9bbf10eb1c8fbe42a04399e0bf29bdd` |
| adapters/pgvector/rows_contract_test.go | `a5754f3f48145fa5bd2b66a9ec80e675bc245f7216537e868a82af82faca8a74` |
| adapters/pgvector/scope_pg_test.go | `2b06ae120f3c5fe02c4b4aa5a66a4e3df9540282c55f68d22d0075753c2f45ed` |
| adapters/pgvector/store.go | `a317625239d67be2a67ec1d17137d1792bed1199bb08503d2a2545d2d2be231b` |
| adapters/pgvector/store_test.go | `28eb5c8dfd0fda469fa7f4e06a66f9304406a24df7f01b675c6f720635204a89` |
| adapters/qdrant/README.md | `d78cedddd68920c2f9a378b2ca582c0f7a2f500fff782b8e5cdb26790f763b18` |
| adapters/qdrant/count_contract_test.go | `9679f98a5d1b4044eb53995606d5e66314e2e46fb56638c82b082256a3cbab44` |
| adapters/qdrant/go.mod | `daffbd96d3a44d399608e71978607f405b082e61713a2253f2655c019f36a968` |
| adapters/qdrant/store.go | `6dffc0b2fb9593169010c6d9bad95d2794f3a3af8e9546f4e4bc531c656b9e3e` |
| contracttest/filter_parity.go | `6ad4459cd80f5d621ecb0d3872b08372a92f07805bb4d5370ffc987f0dc1f84f` |
| contracttest/pipeline.go | `268db0cb98bff4ab03205916933d31a4c98cf497d6a18ebea426223c14e59850` |
| contracttest/pipeline_test.go | `1722028d652725463ee9436f8c507bff4edff6235aa12a8f6fbb11ef9b98184b` |
| docs/task19/capabilities.md | `cf0e9bdcd74ef00ae286ba7a10ededc8c9c3c7364fed40ad86f736376daea594` |
| docs/task20/T17.md | `7cb36505075ddc41bf1a588e17b9e9cb0848b65df261a9520b0e3ad2a69625b4` |
| documents/README.md | `22d18fec34041d396761dfc360dae05ec2c40077568090b20315df6121d5c728` |
| documents/documents.go | `8a2e652af6ef2ed7588184bfe8533240151438aebb9225f839fbc61564902225` |
| examples/conformance/decorators_test.go | `5c6df7b91e6a2120732a199a35819bfb1b2b1c9ec2e826d8e1f711ef9f54d75c` |
| examples/conformance/final_contract/cache_test.go | `0556c7fbed3d1db9086ea33a10ecaa570654c82321f149b4af8e9716cdba9497` |
| examples/conformance/final_contract/context_test.go | `191165934a017510bd1d34b68656efa5fb2fb80a56151d1580d4902952c22976` |
| examples/conformance/final_contract/lifecycle_unix_test.go | `8475ea1277ea02d04a16260df05b5dbcc6ec2593e8c06a9cb32431e287d8d271` |
| examples/conformance/final_contract/recipe_test.go | `58d25d6e9468361301f87cf01c6eeb0706c62b56c09777e42e6a987c91b094fb` |
| examples/conformance/final_contract/space_unix_test.go | `73cb47b001320f8bfbb0f06c01c43dd647244952c358e0e07ed532d6998fa20a` |
| examples/conformance/go.mod | `a02fcc9051a038f627d383d89d1ba61ad4ba22e202a8370c1edd13522112e8cd` |
| examples/conformance/graph_comparison/baseline_unix.go | `92936c5469a7780347a3bf42ee7e1f51355ca9b43da9201b85140180bf2a9a5a` |
| examples/conformance/graph_comparison/baseline_unix_test.go | `f6b3827453a50e563b978b196b974cc3cd7cb9b9862e6781ac5c05732731774d` |
| examples/conformance/graph_comparison/capture_unix.go | `bb475aa45013cb92380e0ce5519f59baefd2da871e384fafacfc38817ebae4e5` |
| examples/conformance/graph_comparison/capture_unix_test.go | `8422f4b25d497bf5a5c64d4545c36cd0343993fbc4dfa7f4a73195ac2a1a9a4a` |
| examples/conformance/graph_comparison/codex_capture_unix.go | `a74b4ec28acfe473c5bcf6294c7f6b3deb8aca7a2d60a59bb14978bdef729ca7` |
| examples/conformance/graph_comparison/codex_ports_unix.go | `5f86c882d97f02bfe07c4a8befc79fdfa65a34d5359d7d3812214afde31cc378` |
| examples/conformance/graph_comparison/codex_profile.go | `acdc872b96525f8d482b394689c2f8f971f0d6e12b883623cb2b1fc30c6629ca` |
| examples/conformance/graph_comparison/codex_profile_test.go | `782d8e9d6d662da1c1d1260a363ef86fe8c0343d4e4b2c8c57eb71243fb174a2` |
| examples/conformance/graph_comparison/configuration.go | `22273ff2a9a4bd7e08ae83a72624f71440dc29b9e0dd04e7983140fe26532f45` |
| examples/conformance/graph_comparison/evidence_unix.go | `197e554da81d267b08b9819b3fe80e5603c3b3810244f247046df811bcbabd57` |
| examples/conformance/graph_comparison/evidence_unix_test.go | `33e1f905dc58d232bc2f06baf4b3fda01621ef835414d399ea1628e1258a725a` |
| examples/conformance/graph_comparison/evidence_validation.go | `ad0cb6194328219133c0693ab5ba41b4942dff4e2c779cbcc72f6707a268b790` |
| examples/conformance/graph_comparison/extraction_model_unix.go | `4f04182caafd378792961121e4c5c0faaa513ff4ba169adf176afa18c7ee9f81` |
| examples/conformance/graph_comparison/extraction_unix.go | `b6168227822b00ab522c17a0e75e3e909ef5f18a85b2b9462b7cda635ebe5fe6` |
| examples/conformance/graph_comparison/extraction_unix_test.go | `61b66eb23ca02e7ccaa39e1c28b9aff28c6c31ab65bb7aaba8ad5cf3752365fa` |
| examples/conformance/graph_comparison/graph_stages_unix.go | `be1511797f227445b5d57cccdbcfcaef872b9b9ba5e895d8aadebae064c62e2e` |
| examples/conformance/graph_comparison/graph_unix.go | `20fe0a43f303c9744d1d6289c723e18b26bde0ef7996377c1bd537212a2fd1db` |
| examples/conformance/graph_comparison/graph_unix_test.go | `3628d79aab6c48dbc1831e7bb47d21c716e34ec6899bfb33979da214ec120497` |
| examples/conformance/graph_comparison/history_unix.go | `55552a613e9bad17517cb1385dedf5703fd48608433fd82094b3f20bb0f3e07c` |
| examples/conformance/graph_comparison/live_unix.go | `4b4d84560dfc240a25543bcfe91b6962ca0b258fa1241287ad0bf0f07f58fcba` |
| examples/conformance/graph_comparison/live_unix_test.go | `8d3e3d84e38acd6526e513b8f3c61cb7c213b449b1a9d519648a1aa265ba7a5f` |
| examples/conformance/graph_comparison/live_unsupported.go | `b88cec3f8cbe12a14de789ef2825801d0c60c689bc4f79924b8a44de155d4657` |
| examples/conformance/graph_comparison/local_unix.go | `e46ee4090030f48851f354e8d81c8c2ad1c10629b2d71baf55395499e8352a83` |
| examples/conformance/graph_comparison/local_unix_test.go | `c64fc34757da3469c61d26ce3770752c074dd672d80b2558ec8989176a25a1a2` |
| examples/conformance/graph_comparison/main.go | `9f053220e36e2214df64230bd5c9fbaf41b1e2ed87aa1f114f31fec200dd9146` |
| examples/conformance/graph_comparison/main_test.go | `0257e7209952dff5297a0e6768cc39481fd9d02b9f2a59a07fcd99b9aed392df` |
| examples/conformance/graph_comparison/model_transport_unix.go | `40c8e7610289eabee7a0ee683107b57de911b0c07451f778ec4152c87d27efeb` |
| examples/conformance/graph_comparison/preparation.go | `46de00ddc04d3edb3786430ab725393125ba42067d88007dc24b287aff437337` |
| examples/conformance/graph_comparison/preparation_test.go | `18c3564866b095449a48969971eedaabbc9db324c74e6a17554c95cdfcb741d0` |
| examples/conformance/graph_comparison/provenance_unix_test.go | `518bde4a812e6b8d0cefe882b7ae9e23e0866224fdc30c39166df443528e813d` |
| examples/conformance/graph_comparison/summary_model_unix.go | `8659dcdaba9f814e8595778163df47746aa78be5853bfe02819cd0c62b22979c` |
| examples/conformance/graph_comparison/summary_model_unix_test.go | `03dba3398e53501f5ac6a2a9467dd0dbbc3881133f44a376072117bdfc4cf90f` |
| examples/conformance/graph_comparison/summary_sources_unix.go | `7f0661ed4d65070e12d0ef0172dcf5fb3c73b80c3a65840f7f821a77f5bc5325` |
| examples/conformance/graph_comparison/summary_sources_unix_test.go | `4b203627fef417b5a76e63685f4060417424bd9942e37f8403b9c55b5756c3b3` |
| examples/conformance/graph_comparison/summary_unix.go | `97a0227af65dcdf63c6d399d37ae7ac8d03e140ebf832e613b566df33e42f674` |
| examples/conformance/graph_comparison/summary_unix_test.go | `a84a35881336a304777f0fb4aa14aafdd98e12744eee6eaf751dd1060f167ad0` |
| examples/conformance/graph_comparison/task19_graph_unix.go | `b3fb1135a390bc72d1860b3fc5da75cbdcc817fce80b1f93e05d5478209692a0` |
| examples/conformance/graph_comparison/task19_unix.go | `a879b5d80038553fa6caab7860b6223e6b092cf50a131b3c17098791405a8d5e` |
| examples/conformance/graph_comparison/task19_unix_test.go | `f058362939589f5a654f8a9cc205a1bde5ea1fa414b0d6f2d350c0b275718d65` |
| examples/conformance/graph_read_unix_test.go | `30cb5d0594a4947b45cc4e44623c90df188b30ac391a266da9c9878d71ac3851` |
| examples/conformance/integer_stage_unix_test.go | `9917e30824bf24699e958e336cbfc2119dcb7595b57781230cf7fb90226d401a` |
| examples/conformance/integer_storage_unix_test.go | `543d71b7ffd58357493e006106205b034013e03631c113ea47b3c390ecfbc383` |
| examples/conformance/internal/codexcall/audit_regression_test.go | `07f45368bd9797aacc5dd096845b13128b31619850d4c2e1eb9ffd25d8756002` |
| examples/conformance/internal/codexcall/call.go | `f5071e7b236a25b1ebf65ea9f64a4bb1ac88269f19f52e66514d5da6088e124b` |
| examples/conformance/internal/codexcall/call_test.go | `923fe54d0c3076aa030d1cfd48b39f2305615b215bdde4ab93d8c7b29415f5ce` |
| examples/conformance/internal/codexcall/json.go | `4171941f8f92d6ec396f0dfc1c33f4805013b7223610bcf927dd90a8d14b5c3d` |
| examples/conformance/internal/modelcounter/counter.go | `1d7facd0c319b6995c2f50bdf2cb9eacb06cc9ba7eb0ad653f07b4987851062f` |
| examples/conformance/internal/task19/data.go | `72b4158b5e1def5ec98c1f76d49c646eb16dbea343f5391903a8df193ac1c018` |
| examples/conformance/internal/task19/dense.go | `19f844df4c9067478b1ac6e84e09d7e65ed4eba21ea23715d96a90939bc57992` |
| examples/conformance/internal/task19/evaluation.go | `3bf0b17dece14f91c13caefb57b2ecc44f199f99fbc66b480d04486b53d0db69` |
| examples/conformance/internal/task19/evaluation_test.go | `370208e9d37fbef5ba7c1179176168357a786d2d38f81ed0777c0119c9be43e2` |
| examples/conformance/internal/task19/freeze.go | `aa75ef617c75d99da03cfd80f28b2903b4df316b1156bfe4931a06d0b20efd9e` |
| examples/conformance/internal/task19/freeze_test.go | `d89c2c52f4c4dd3baacadf10ea2273736d71eac67a6b821d77527b713a1b259a` |
| examples/conformance/joint_read/composition_unix_test.go | `c9ad5487877899f5bb8c912d6aa8328d197741b853ca837aa27eaaf3e5f14917` |
| examples/conformance/joint_read/decorators_unix_test.go | `4a2e9b73da29d871153e422531b4e3160c11a60e921153bb5fa54bde9f493cd3` |
| examples/conformance/joint_read/fixture_unix_test.go | `1d8c77e45c24bb5300d6997888ef183a4f9e712118d0945913ee2eb2a64f903b` |
| examples/conformance/joint_read/recording_envelope_unix_test.go | `24be46397edac5e5a11c01074a1ff991a3a1f5bead5f9ed479a8fe9be7f3a6a3` |
| examples/conformance/joint_read/recording_unix_test.go | `82c0ab3e6f1df932925ef4966011c587948e04cb74dbd503bf426df166f760a8` |
| examples/conformance/observation_contract/observation_test.go | `bb556c2e35459687e4e372dda56f488b393c4127b4a038ee95dd726adf5aa494` |
| examples/conformance/persistent_read_unix_test.go | `d660d5b1dda94f111a2bda5e3d06b15cc0b6614395afc4f70877682b755690a2` |
| examples/conformance/read_test.go | `799e93087a4014fce1a25dd3e494e2464dce661627915c59bcdc58ac49789cda` |
| examples/conformance/recipe_comparison/capture.go | `52d21ae65854c897e8c3b663b1c4a6011f82877b1e45cfc1f468c71c49e321d1` |
| examples/conformance/recipe_comparison/capture_test.go | `d36695461c48d46c0dcb2d6009a786132c6bf25e28927e081506270f2edca8f7` |
| examples/conformance/recipe_comparison/codex_calibration.go | `d3d949e64fd78a1587b980044213d7fa0f5fc191f1443892eee366f7d3df5b34` |
| examples/conformance/recipe_comparison/codex_capture.go | `38826fb583d0ef22f48661063fbc4a1e80015a04bc332e5e3fef382bafadf26e` |
| examples/conformance/recipe_comparison/codex_ports.go | `31fedf2e4cdee6b5998e0cd3f6a896e0d188cb17a654310d8152dc49a07f217b` |
| examples/conformance/recipe_comparison/configuration.go | `6a407ecb761c0e0ddeef3802f6ead839e510e4cdd7e212d3996d1217a8d62951` |
| examples/conformance/recipe_comparison/configuration_test.go | `650d55b4ed622f023ca4d02fd40a9f2f6dc42d7e1377b8c7b7305f0e96f16351` |
| examples/conformance/recipe_comparison/live.go | `401a16380706b881e6c923c9205d847255fcaa32a93bf2d2dfafe8ffe790bf8f` |
| examples/conformance/recipe_comparison/main.go | `0ed4c867d51d322ec0eb97f8275fd8833da929f31c6489d2e4570d52596012d7` |
| examples/conformance/recipe_comparison/main_test.go | `63a6fa4754ccfc2ee99d039aef7f77d9044706871b1baf1d8eabbf4b914adf46` |
| examples/conformance/recipe_comparison/model_ports.go | `7de6c31d5972dbfc471564e462ac715fbec5b057744d87ac8f8f1eef9b1f2c20` |
| examples/conformance/recipe_comparison/model_ports_test.go | `4919b0cd0c094a6104461b926f0b283d67fb3378e334c5a7a5bfafa3b87e8313` |
| examples/conformance/recipe_comparison/task19.go | `8d191f0fcf1de69050f8ed7c77692761085ad080540616cfdffabf132b1f6ae7` |
| examples/conformance/recipe_comparison/task19_test.go | `65798d8ad6c886a5582e873b281814f34465c7053c657a90f7e6b2adbfa930e4` |
| examples/conformance/recipe_comparison/tokenizer.go | `543cd6427d0413db5dfdd0a36a244bc3f93bf40df80a69d68357e131d8c63275` |
| examples/conformance/recipe_comparison/tokenizer_test.go | `c08f1983b3e15031349b484b29474bbcfe8b72d5c8a130e68058bddd0b5bc02f` |
| examples/conformance/tensor_comparison/configuration.go | `64d9e08756407e34512d940acd34f512c7d3f6e20737d1b12937f492082fd5ca` |
| examples/conformance/tensor_comparison/main.go | `2f858683230bd69da40146480f18aacddb0ed4efc6eb0da94a82842d882c385a` |
| examples/conformance/tensor_comparison/main_test.go | `cd9f76b3335e1f46ae56cc0bcdf7059612d37cac712cab09211ef60094c3003a` |
| examples/conformance/tensor_comparison/profile.go | `9f0629614159db92ee444c14a46a7f64efc109fdb2c35c8965ea6d7b8e0db066` |
| examples/conformance/tensor_comparison/task19.go | `0189ac16a468f7954706d3b4c101942e639d0b0c26db7ba6ac99cf5a5691a52e` |
| examples/conformance/tensor_comparison/task19_dev_test.go | `ffb5150159682960d72610620d0b40508452328e2e7dd9a195554e918c80c414` |
| examples/conformance/tensor_comparison/task19_test.go | `0b148f052e0cdf18f0e5d9c6eaedcd2c36e01deaf5fd860ca1768e1b64286dcd` |
| filter/README.md | `a6755b651d552618a1a5374fdbc39b834bddf240a4cbfce1dd069fbead979436` |
| filter/builder.go | `b28092bf28a886a6427d2585b20c19d0c365dcef531042e374029ac24761d2e9` |
| filter/builder_contract_test.go | `23446b78d4474ceb88bb45b9473eaa54c4399f48985773c6ac98f9849411341b` |
| filter/filter.go | `6c2cf761385a73de54728fb64c469aea16caf3bdaa293740e0cbb125a13c42d8` |
| filter/match.go | `ca33153f14abb918b282cee0ad2bce51f08eeaae812f2cf000dca4c75dcbd5c1` |
| retrieval/codec.go | `27634ea00548d793818f8639d8a923a2e4fd9af80bd7830f06f3a7ef1916e745` |
