# T15 independent correctness acceptance

Verdict: **PASS**, no unresolved correctness findings. Baseline `e5e9d1e18407b1c02a5a4df6be4e3df5cb1f23e5`. Independent nonimplementing audit; did not read counterpart acceptance. User AGENTS instructions were provided in conversation; no repository-root AGENTS.md exists. Reviewed master task, original ingestion report, all six backlog criteria and all 38 assigned source rows.

| Criterion | Verdict | Evidence and limits |
|---|---|---|
| T15.C01 | PASS | Explicit fallback overlap, conditional delimiter loss, hash-line/fenced-code/Setext and punctuation/BYO grammar retained honestly in current chunk guide; permanent readable grammar tests. |
| T15.C02 | PASS | Whole contextual batch validation before any generator; ordered index/source/Total0 or exact N; standalone UTF-8 and exact supplied span; whitespace-only sentence gaps. Invalid generated context fail-zero; existing cancel/join tests pass. Independent multibyte input-span and Unicode-whitespace adversarial cases PASS. |
| T15.C03 | PASS | Each projection failure path returns nil; pre-callback Content/Context UTF-8; default identity rejects actual source disagreement, custom policy explicit; URI/StorageID/fallback and borrowed metadata/preauthorized layout input documented. Late identity/metadata/index/document tests PASS. |
| T15.C04 | PASS | Ordinary mapping JSON decision explicitly documents duplicate/case/missing-zero/Unicode repair and source authenticity limit, with executable policy cases. Catalog/Loader no-latest and raw administration unchanged; Slice broad supports retained; media resolution unchanged. |
| T15.C05 | PASS | Partial OCR/empty ImageText override tested. Literal inactive multimodal fields and UTF-8 centralized without URL mutation/network policy. Original-source membership remains explicit. Capture rejects above MaxExactRank before hit identity; independent callback-count test PASS. Decoder validates exact ordinal domain; host nested-input work bounds stated. |
| T15.C06 | PASS | Current guides link runnable actual BM25/retained historical/denied resolution and layout/OCR profiles. Fresh independent nine-package race and primary lint PASS. Before/after benchmark logs match retained baseline benchmark and current implementation; limited 100ms measured claim, no RSS/parser speedup claim. Independent current benchmark PASS. |

Correctness scope: 6/6 criteria PASS and 38/38 assigned source rows addressed. Required checks have no SKIP on this darwin/arm64 64-bit host. Evidence rank test's 32-bit branch was not used. This acceptance does not claim final all-module/live parser gates assigned T22.

Independent checks:

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./source ./chunking ./layout ./documents ./multimodal ./evidence ./recipe ./graphingest/materialization ./graphingest/resolution` exit0, `T15-correctness-race.log`.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run --allow-serial-runners ./source ./chunking ./layout ./documents ./multimodal ./evidence` exit0, 0 issues, `T15-correctness-lint.log`.
- Private Go overlay with three independent boundary tests under fresh race, exit0; `T15-correctness-adversarial.log`: invalid multibyte supplied span and Unicode whitespace coverage; oversized rank zero hit-identity calls/zero record; malformed mapping rendered rune boundary rejected. Overlay sources retained at `/private/tmp/ragy-t15-correctness/` and not written into implementation tree.
- Independent `go test -run '^$' -bench BenchmarkLongPageWordMappings -benchmem -benchtime=100ms -count=1 ./source` exit0, `T15-correctness-benchmark.log`.
- `git diff --check` PASS; baseline HEAD unchanged.

An initial expanded command mistakenly referenced nonexistent ./artifact and exited setup failure although real packages passed. That invocation is not counted as PASS; corrected full command above reran fresh and exited0. No implementation changes were made by this reviewer.

Optimization audit: OriginalTexts performs snapshot UTF-8 once then preserves per-locator shape/bounds/rune starts and exact mapping construction; MappedText.Validate checks UTF-8 before private boundary helper. regionText only changes per-word construction to batch and retains encounter order, empty return and JoinMapped separator/support behavior. Permanent exhaustive interval/late-error/ownership tests plus independent malformed rendered-boundary overlay PASS. Retained before benchmark constructs identical multibyte word spans using independently validating OriginalText; after uses admitted batch. Baseline before logs are execution evidence supplied with task, not independently rerun by this reviewer. Allocation differences and residual support construction cost are disclosed.

## Source decisions

| Source | Verdict | Decision evidence |
|---|---|---|
| D34 | PASS | contract; `chunking/README.md`, `chunking/remediation_contract_test.go`, `chunking/ranges_fuzz_test.go` |
| D35 | PASS | contract; `chunking/README.md`, `chunking/remediation_contract_test.go` |
| D36 | PASS | change; `chunking/contextual_admission.go`, `chunking/chunk.go`, `chunking/ranges.go`, `chunking/remediation_contract_test.go` |
| D37 | PASS | change; `chunking/project.go`, `chunking/remediation_contract_test.go`, `chunking/README.md` |
| D38 | PASS | retain; `source/README.md`, `documents/README.md`, `layout/project.go`, `documents/chunking_integration_test.go` |
| D39 | PASS | contract; `source/README.md`, `source/mapping_profile_test.go`, `source/mapping_json.go` |
| D40 | PASS | change; `source/scaling.md`, `source/mapping.go`, `source/locator.go`, `source/original_batch_test.go`, `layout/resolve.go` |
| D41 | PASS | contract; `layout/README.md`, `layout/project.go`, `layout/project_test.go`, `layout/ocr_test.go` |
| D42 | PASS | change; `multimodal/multimodal.go`, `multimodal/multimodal_test.go`, `multimodal/README.md` |
| D43 | PASS | change; `evidence/README.md`, `evidence/capture.go`, `evidence/record.go`, `evidence/rank_contract_test.go` |
| ingestion:01 | PASS | contract; `chunking/README.md`, `chunking/remediation_contract_test.go`, `chunking/ranges_fuzz_test.go` |
| ingestion:02 | PASS | contract; `chunking/README.md`, `chunking/remediation_contract_test.go` |
| ingestion:03 | PASS | contract; `chunking/README.md`, `chunking/remediation_contract_test.go`, `chunking/ranges_fuzz_test.go` |
| ingestion:04 | PASS | change; `chunking/contextual_admission.go`, `chunking/chunk.go`, `chunking/ranges.go`, `chunking/remediation_contract_test.go` |
| ingestion:05 | PASS | contract; `chunking/README.md`, `chunking/remediation_contract_test.go` |
| ingestion:06 | PASS | change; `chunking/contextual_admission.go`, `chunking/chunk.go`, `chunking/ranges.go`, `chunking/remediation_contract_test.go` |
| ingestion:07 | PASS | change; `chunking/contextual_admission.go`, `chunking/chunk.go`, `chunking/ranges.go`, `chunking/remediation_contract_test.go` |
| ingestion:08 | PASS | change; `chunking/contextual_admission.go`, `chunking/chunk.go`, `chunking/ranges.go`, `chunking/remediation_contract_test.go` |
| ingestion:09 | PASS | change; `chunking/project.go`, `chunking/remediation_contract_test.go`, `chunking/README.md` |
| ingestion:10 | PASS | change; `chunking/project.go`, `chunking/remediation_contract_test.go`, `chunking/README.md` |
| ingestion:11 | PASS | change; `chunking/project.go`, `chunking/remediation_contract_test.go`, `chunking/README.md` |
| ingestion:12 | PASS | retain; `source/README.md`, `documents/README.md`, `layout/project.go`, `documents/chunking_integration_test.go` |
| ingestion:13 | PASS | retain; `source/README.md`, `documents/README.md`, `layout/project.go`, `documents/chunking_integration_test.go` |
| ingestion:14 | PASS | retain; `source/README.md`, `documents/README.md`, `layout/project.go`, `documents/chunking_integration_test.go` |
| ingestion:15 | PASS | retain; `source/README.md`, `documents/README.md`, `layout/project.go`, `documents/chunking_integration_test.go` |
| ingestion:16 | PASS | contract; `source/README.md`, `source/mapping_profile_test.go`, `source/mapping_json.go` |
| ingestion:17 | PASS | contract; `source/README.md`, `source/mapping_profile_test.go`, `source/mapping_json.go` |
| ingestion:18 | PASS | change; `source/scaling.md`, `source/mapping.go`, `source/locator.go`, `source/original_batch_test.go`, `layout/resolve.go` |
| ingestion:19 | PASS | change; `source/scaling.md`, `source/mapping.go`, `source/locator.go`, `source/original_batch_test.go`, `layout/resolve.go` |
| ingestion:20 | PASS | change; `source/scaling.md`, `source/mapping.go`, `source/locator.go`, `source/original_batch_test.go`, `layout/resolve.go` |
| ingestion:21 | PASS | contract; `layout/README.md`, `layout/project.go`, `layout/project_test.go`, `layout/ocr_test.go` |
| ingestion:22 | PASS | contract; `layout/README.md`, `layout/project.go`, `layout/project_test.go`, `layout/ocr_test.go` |
| ingestion:23 | PASS | change; `multimodal/multimodal.go`, `multimodal/multimodal_test.go`, `multimodal/README.md` |
| ingestion:24 | PASS | change; `evidence/README.md`, `evidence/capture.go`, `evidence/record.go`, `evidence/rank_contract_test.go` |
| ingestion:25 | PASS | change; `evidence/README.md`, `evidence/capture.go`, `evidence/record.go`, `evidence/rank_contract_test.go` |
| ingestion:26 | PASS | change; `evidence/README.md`, `evidence/capture.go`, `evidence/record.go`, `evidence/rank_contract_test.go` |
| ingestion:27 | PASS | change; `evidence/README.md`, `evidence/capture.go`, `evidence/record.go`, `evidence/rank_contract_test.go` |
| ingestion:28 | PASS | change; `source/README.md`, `documents/README.md`, `layout/project.go`, `documents/chunking_integration_test.go` |

## Accepted source fingerprints

Only implementation/tests/current contracts/guides and retained baseline benchmark; mutable journals/bookkeeping/acceptance reports and unrelated iCloud duplicate excluded. These hashes define this acceptance's exact candidate.

```text
43fab1494113c1d00ddf307058f95c297e69cd252fc29ec17bdb03d71a44f4e9  chunking/README.md
fa9f9a903bcc731fa4c53b785407c020d66c32cb71f7fce539f8267507687d27  chunking/chunk.go
99976bc1a4651645150bfb9d56bc6d916f2afbf538f64ab079d6cc248c35b152  chunking/chunking.go
a05cc174e1012964666f14a1ce2334d14e11cbbd18a76482bc11f5c990339709  chunking/contextual_admission.go
a4d1552c7e6441eaaa677fa255c0fd70d4330b7622ea3d7d8b923b2db1c5d97e  chunking/project.go
efa7fd4564d687f3071682769f0bb7db83b227cb66d92acdc31baa71c277c7d5  chunking/ranges.go
f7deea757b31ffbd6c6f24a5b202b4ee6ec89a21b76233e6b97c9e008ae3c9bc  chunking/remediation_contract_test.go
777e3d1911e2dd4d22d1f1ac23f951f17b81a12dd3682a57c78647774ff4c99c  docs/contracts/remediation.md
cc6536b7fabc92b1ac27e9c1250885a6c62e3d5f1499c67539e1375ceca4cb00  docs/task20/T15-before-benchmark.go.txt
22d18fec34041d396761dfc360dae05ec2c40077568090b20315df6121d5c728  documents/README.md
519a4090353d04bcab8485a08c75f35d9339bffdea4bc93bc971a3f6add0e919  evidence/README.md
c2846509d72f9e7b54411767c30af03a763c7e1fbcd72d9f5355fa9bb697e3c6  evidence/capture.go
ac76f244ce390930c8426a55b855ae7e84ed5a5a15690bff1bfcdb7658de6783  evidence/contracts.go
c335420b33fdaeb8b7b019ee2895bd6cc8ad6a3a9aeca30103ccdd03a9ad6b26  evidence/rank_contract_test.go
27388e7fc7ebe3fab54d5bea52d90a2661fbd3f58dfb4df7d7f49115a22e0230  evidence/record.go
57cb4a5aad495ff6113782ca3d9df2d5305e4c90bb3313801974054d0ee2cd25  layout/README.md
4c2bec9ab24cc45911c5801df0ba4263538738e1e198508c165c50b919e93bef  layout/project.go
ee4dba04e4193f9898ae6e0d1ae685a63fb5ac1deeeca9b4cd3ff25993b4a7fd  layout/project_test.go
3025dffb4e8bad2ff66b2a1b971862a57d170a0c4596c667112ce55cb6c84099  layout/resolve.go
2ade27b9599f4abb5bd8d1ad43322d18c6f38eb4f9b7bc65b9ec87b7990fa13a  multimodal/README.md
b8f9e9a067af26c1a080f8d2eb7c537cde1c26d8442bb4a4bc6a4c6c5aecfd43  multimodal/multimodal.go
76014a64fcfc4f412e2f231371b70a3b2cfa4c4da7e1ee2585b7c66530467e34  multimodal/multimodal_test.go
7fadfd09fce54887ed63836462781c7009ddebe6075ada77274fa876f5bcaa0f  source/README.md
91ed0c264033bd5a60c145ed038f4b071e0f2a05da718acfb0ab5593fa2358f9  source/locator.go
3b147d40b2a7a7b7f82042a0fcad05fed846dafab82c5a302099412637dbaefc  source/mapping.go
8abb06f144a4143327e8a17a9546bb5ecf92379f9bbdd1cc27db586332717677  source/mapping_profile_test.go
1dc51cdf7b504b22861915c6c621d0a589512d00db839efb5329c5b6ffac2274  source/original_batch_test.go
a2d11b1df42ef20e43067721ca8266867f59f39a100e4bf3ff0a292129137f9e  source/range_scaling_benchmark_test.go
49cf7f7f57037f1363e6a5e90dd26b8fd3ec3b7dbf543e411635d6cee45ca799  source/scaling.md
```
