# T18 independent correctness acceptance

Verdict: **PASS** on the stable revised candidate. No unresolved findings.
Baseline: `98ff5c013c756724fdb286035fa72777dda30137`.
I did not implement these changes or use the completeness report as evidence.

## Scope and review

Reviewed every changed production source, test, current guide and T18 contract below against the five backlog criteria and original provider review. Source scope is D55–D59 and providers:02–18 (22 assigned items; providers:01 belongs to already accepted T11).

- C01 / D55–D56 / providers:02–05: shared credential admission is applied before dispatch across all provider constructors; blank, invalid UTF-8, ASCII control/DEL reject without changing accepted bytes. Structured ordinary Doer failure is unavailable, body failure protocol, context sentinels retain precedence/privacy. Required structured index is nonnull zero and rejects with admitted usage. Shared last-duplicate/case/surrogate normalization and structured exact decoded duplicate/depth rejection remain separate explicit policies. Domain validation remains a host callback.
- C02 / D57 / providers:06–10,13: Space remains authoritative, optional Model equality assertion and host-attested revision/configuration remain explicit. MaxVectorRows replaces only the matrix-row limit; compiled rename inventory has no stale embedding limit references. Remaining MaxOutputTokens fields are distinct graph/summary/extraction completion budgets. Finite defaults, explicit structured/PDF bounds, purpose/dimension/metric profiles and Cohere query-plus-documents units agree with code. Input/output bounds do not promise peak RSS/CPU.
- C03 / D58 / providers:11–12: raw model and full data/embedding fields postpone typed admission until independently valid usage is observed. Rejected vector types, float32 overflow, model/cardinality/index/shape/row failures yield no embeddings and retained valid usage. Malformed/negative/absent usage and unreadable/truncated/trailing whole JSON never become known. Post-model/data decoding gates and final materialization gates preserve context priority. Cohere retained input/prefix with error is not claimed as reranked success; final protected delivery remains active.
- C04 / D59 / providers:14–15: only private declared geometry/limit exceptions map to unsupported/invalid; recognized parser read/syntax/EOF map to invalid input; unexpected dependency/programming exceptions map to sanitized unavailable. No raw exception/source text or partial document is returned. Runtime/dependency/fingerprint descriptions match execution.
- C05 / providers:16–18: examples compile under independent module races; current ownership/concurrency documentation agrees with per-call transport state and host ports. Real PDF extraction/geometry/limits plus projection/scoped resolution/durable publication run successfully. Metric implementation is unchanged; no optimization claim, before/after N/A. No new mandatory orchestration, routing, schema, billing or credential-store layer.

## Findings resolved before final acceptance

1. P2: initial PDF main caught arbitrary OverflowError/NotImplementedError as declared limits/geometry. Independent fault injection confirmed the contradiction in `T18-correctness-unexpected-exceptions.log`. Private engine exceptions now isolate those admission decisions; revised 11-class fault injection passes.
2. Raw model/vector envelope decoding now preserves valid accounting across typed decode rejection (root identified initial typed-vector loss; independent 24-case public API probe confirms the revised result).
3. New model decode initially lacked a post-decode context gate. The revised candidate checks context before returning model decode/admission failures, consistently with the data gate.

Initial passing tests do not certify the changed revised sources; revised runs below supersede them there.

## Independent executed checks

All commands below exited 0. Go executable `/opt/homebrew/Cellar/go/1.26.5/bin/go`, `GOTOOLCHAIN=local GOWORK=off GOCACHE=/tmp/ragy-t18-correctness-cache`.

| Check | Evidence |
|---|---|
| Root `test -race -count=1 -v ./embedding ./internal/providerhttp ./dense ./tensor ./recipe` | `T18-correctness-root-revised-race.log` |
| OpenAI full module race, including structured envelope/error regressions/examples | `T18-correctness-openai-revised-race.log` |
| Jina full module race, dense/tensor bounds/accounting/examples | `T18-correctness-jina-revised-race.log` |
| Gemini full module race, dense/shared wire/multimodal | `T18-correctness-gemini-revised-race.log` |
| Cohere full module race, unchanged after initial independent run | `T18-correctness-cohere-race.log` |
| PDF full module race with actual interpreter: eight actual-parser parent tests PASS, no PDF SKIP | `T18-correctness-pdf-revised-race.log` |
| Independent temporary consumer `go run -mod=mod -race .`: 24 public API adversarial accounting cases, no network | `T18-correctness-public-probe.log` (includes reproducer source/module) |
| Real-import exception injection: 11 sanitized classes PASS | `T18-correctness-engine-errors-revised.log` |
| Independent actual PDF fixture inspection | `T18-correctness-fixture-revised.log` |
| Actual runtime/dependency introspection | `T18-correctness-runtime.log` |
| Whitespace patch admission | `git diff --check`, exit 0 |

Actual Python `/Users/skosovsky/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`: Python3.12.14, macOS27.0.1 arm64, pdfplumber0.11.9, pypdf6.10.0, pdfminer.six20251230, reportlab4.4.9. Ran both `adapters/pdf/testdata/verify_engine_errors.py` and `verify_fixture.py` with this interpreter and `RAGY_PDF_PYTHON` for Go tests.

Paid provider smoke profiles explicitly SKIP without opt-in (OpenAI, Jina dense/tensor, Gemini dense/multimodal, Cohere); these are not PASS and are not required by T18.C05. No paid calls or remote model-quality/revision claims. This report certifies the changed local wire/contract/PDF profile, not all-repository final acceptance or arbitrary parser platforms. The previously known production-doc wording blacklist remains assigned to T21. Reviewed root lint evidence is separate from the independent race runs listed here; no claim of an independent all-module lint run.

## Stable candidate SHA256 manifest

All 39 changed production/test/current-doc paths are covered. Mutable backlog/plan/traceability, acceptance reports/logs and unrelated iCloud duplicate are excluded. Hashes describe the final candidate, not the initial rejected revision.

```text
af99f8c83b2e5c5cdb59d3ff354ac00503ab48d9be7cda664ec1d05157fe2153  adapters/cohere/README.md
87094fb5e94b3b6494baf6f0af2f662a0be7c8e8b0e30e08c92e9abec90f8933  adapters/cohere/rerank/client.go
dbee717a7b1865e4143a912b82423c311a268045820bd29fe8b38d4ec6b7e1d7  adapters/gemini/README.md
b7113caea4abac1978ac2832a607f69a60e8535812265451b3cc1c1897ef5f30  adapters/gemini/dense/client.go
1a9cd65e7ac92a359463665d1cd143b1662d2795e50c7c7815cccdcad300c79c  adapters/gemini/dense/example_test.go
97d34622e74b3f031feaaa339b88c81a8b1876eb26fb561796135d7fe4387d47  adapters/gemini/dense/usage_contract_test.go
0a08ec254065b4c26dccc44699a905fe0907650b6d063657dd3509402724a2c9  adapters/gemini/internal/wire/client.go
15f98efe16be3294ebc3a647f875a2bd0b45cc339fbdffde7cf47874ef19987a  adapters/gemini/internal/wire/usage_contract_test.go
6faf6b70a41c9b1cd1b3c764879f1ffa17c2d3671eba8ef6b0d55c0a7b81ecc9  adapters/gemini/multimodal/client.go
5384a5328f08c91c5abe6dffe8475b08a42bdf9bdd12b82d14a3bb3486d81f28  adapters/jina/README.md
68734a463b967ac0d0c67acdeccb6789287f3396eb0e244b648a5b5d3f8e31d3  adapters/jina/dense/client.go
ba621ee6aec4c7886a1d5db1da52bb7102a2395f42d48bd3e086c306021a75db  adapters/jina/dense/example_test.go
943bf49934870c5a21d9a46c978deb4e9364409aa1955799de1ddcd67c38f110  adapters/jina/dense/usage_contract_test.go
23a031f24ad286ce9adcb04da6b6930f3d331c1dbb92c765ab0474c9ef33b889  adapters/jina/tensor/client.go
8718505b0f66e1e35841a2b37cb43afdc8e44106305942131c8547e60549ba68  adapters/jina/tensor/client_test.go
c85183924ec17d041342717e88c25eeaa80d307cce587810a24d03957bb605ea  adapters/jina/tensor/example_test.go
372000cae3211873218b58dc266cdc8dbf398c47847ff05f2f93ddccd23f3793  adapters/jina/tensor/usage_contract_test.go
41ee5f379f138e83bf01cc8cbce615bf5f2b5e50038e8d10c7a0507232ee4133  adapters/openai/dense/README.md
3a6a1cb66f7afbf627d25406bb6b57b358cfbd450518473780bccf988fc2feb3  adapters/openai/dense/client.go
1a90ee487f70eeff975660227b11968e771b939e39473fded00686818f22bfde  adapters/openai/dense/example_test.go
4b4a37275e7a0895b769467e577da12d724f24d1e517a4d355a4a84e17accbcb  adapters/openai/dense/usage_contract_test.go
c95fef3b0c44da09cb72d112a1c3050cdbfcdecb08c12ba36f0a0b9bb2f497e0  adapters/openai/structured/README.md
83925bb3643aac9c0eda0698a31e10a683b003d6125546cfe413db83e6e3d51a  adapters/openai/structured/client.go
50dd98beb238cd4c23e1eaed0a033da9e20e847d5cf7399b01951e2d5eb32424  adapters/openai/structured/client_test.go
36031e99d979d02a47a955371e66a1a8b2ecf67a91cf058b7e3affb25d627e4c  adapters/openai/structured/decode.go
fc1e62edd0acceda3dacde9b1152a10ab687ea302858a33416c15132b458169e  adapters/openai/structured/transport_regression_test.go
9c3bd9b18c78003c8d82d7457b0566892f6611a482114d89ebe165a6e2d57310  adapters/pdf/README.md
4fb8ed59713ea14a95c0c43a73cf5f904c3d78fcbc5a475c1e0b486132fb1079  adapters/pdf/engine.py
06c9212a8f8e12e88d250891e3154eb0f59908b2f88a6c0dc889e70d877d895b  adapters/pdf/engine_contract_test.go
09ba406be77c05fa725e9402568dfe992a2b384645c56b79ec1e787f4a36b3d0  adapters/pdf/normalize.go
10b2fabcc3b058dd160959ebbd71397982e9d9b3f9c6d6289befe9089624daeb  adapters/pdf/testdata/verify_engine_errors.py
bf14d5d3d2cd6dcf7945a69a5d0863738f5def19c165f7334f3aab009be613c8  docs/task20/T18.md
97284ca0bc51986f3322eb0513e21376e758d5b4c0d299bf7c88b91ecaacfcfc  embedding/README.md
5a2f58d7134b523f4ad3e8c6f1ce130b202c1a3906687e93481f467a8bb91661  embedding/embedding.go
bed96459e9a4703eaf60022ba56d00fa519627c419b6c37b88fd6affc12e59ce  internal/providerhttp/client_test.go
c2b619d047f1a61d6828374ab3af461526c4022a2f034b90844430c1b271054d  internal/providerhttp/credential.go
68467f708b2f0d7498085e47d25f6dc79975708cde4ec3d4ffcc20493731fbd5  internal/providerhttp/credential_test.go
a5a6d71ebc58336d5f99d640d14503a53b6e831f4731518c951c481851b8e114  internal/providerhttp/json_policy_test.go
6e03363e842445d744055311111baa5740797f7aa817c02c514c6dd55c01025f  internal/providerhttp/model.go
```
