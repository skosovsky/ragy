# T18 independent completeness acceptance

Baseline: `98ff5c013c756724fdb286035fa72777dda30137`. Reviewer did not implement changes and did not read counterpart report. This verdict applies to the final revised candidate identified by SHA256 below; initial runs do not accept superseded source.

## Verdict

**PASS: completeness 5/5 × 100 = 100%; assigned source coverage 22/22 × 100 = 100%.** No incomplete or blocked criteria and no unresolved completeness finding.

## Criteria

| Criterion | Status | Evidence |
|---|---|---|
| T18.C01 | выполнено | D55–D59 и providers02–18 dispositions inspected against original master and providers role text; providers01 remains separately accepted T11. Shared credential constructors, transport ErrUnavailable/body ErrProtocol and cancellation gates; JSON policies separately stated; structured required index tested missing/null with usage. |
| T18.C02 | выполнено | embedding/README.md identity/default/explicit-limit table; optional Model equality retained, revision/configuration host attestation, purpose/metric profiles unchanged. MaxVectorRows clean rename and compiled inventory verified; Cohere query+documents units explicit. |
| T18.C03 | выполнено | Raw Model/Data/Embeddings decode occurs after valid usage; model type, vector type/float32 overflow/cardinality/index/shape/row failures return zero embeddings + observed usage. Whole malformed/truncated envelope and invalid/absent counters are unknown. Independent module race covers regression tests; Cohere error/prefix distinction and final delivery retained. |
| T18.C04 | выполнено | Recognized parser syntax/read exceptions map invalid input; private geometry/limit exceptions avoid classifying dependency OverflowError/NotImplementedError as user faults. Real-import11classes and actual parser PASS; runtime versions independently match README; fingerprint inputs and RSS/CPU limits documented. |
| T18.C05 | выполнено | Four compiled runnable client examples, public ownership/concurrency GoDoc and guides reviewed. All five modules GOWORK=off race/examples PASS; revised affected modules rerun; actual PDF full profile without SKIP PASS and verifier PASS. Dense metric algorithm unchanged, no optimization or speedup claim; before/after N/A. |

## Original-source coverage

Master D55–D59 and all seventeen assigned role decisions providers02–18 are counted separately. providers01 is not counted again: it is assigned/accepted under T11 (URL/cancellation shared helpers); original P-01/P-02 remediation remains intact and structured module regressions pass.

| Source | Status | Disposition and inspected evidence |
|---|---|---|
| D55 | выполнено | change: Shared T11 URL/context helpers retained; ordinary structured Doer failures align with ErrUnavailable, body failures stay ErrProtocol. Shared UTF-8/control/blank credential admission runs in all constructors before dispatch. Evidence: `internal/providerhttp/credential.go`, `internal/providerhttp/credential_test.go`, `adapters/openai/structured/transport_regression_test.go`. |
| D56 | выполнено | change: Structured choice index requires nonnull zero with known usage preserved. Unknown-field evolution, exact decoded duplicate rejection, case aliases and surrogate normalization have separate explicit policies; host executable schema retained. Evidence: `adapters/openai/structured/decode.go`, `adapters/openai/structured/client_test.go`, `embedding/README.md`. |
| D57 | выполнено | change: Space remains authoritative; Model convenience equality assertion retained explicitly. Host attestation is separate from optional model echo. MaxVectorRows clean rename completed after inventory; shared defaults/explicit units table published. Evidence: `embedding/embedding.go`, `embedding/README.md`, `docs/task20/T18.md`, `adapters/jina/tensor/client.go`. |
| D58 | выполнено | change: Independent valid usage precedes delayed raw model/vector/shape/index JSON decode and semantic admission; known usage survives malformed payload members/float32 overflow, rejected model/cardinality/vectors and postdecode cancellation without embeddings. Invalid/absent counters and malformed/truncated whole envelopes remain unknown. Shared optional model echo decoder has post-decode context gates. Evidence: `adapters/openai/dense/client.go`, `adapters/jina/dense/client.go`, `adapters/jina/tensor/client.go`, `adapters/gemini/internal/wire/client.go`, `adapters/openai/dense/usage_contract_test.go`, `adapters/jina/tensor/usage_contract_test.go`, `internal/providerhttp/model.go`, `adapters/gemini/internal/wire/usage_contract_test.go`. |
| D59 | выполнено | change: Recognized parser read/syntax errors are invalid input; only private declared limit/geometry exceptions select invalid/unsupported. Unexpected dependency OverflowError/NotImplementedError and implementation errors are internal unavailable without raw text. Actual runtime/dependency/fingerprint and 11-class fault profile published; no RSS/CPU claim. Evidence: `adapters/pdf/engine.py`, `adapters/pdf/normalize.go`, `adapters/pdf/engine_contract_test.go`, `adapters/pdf/testdata/verify_engine_errors.py`, `adapters/pdf/README.md`. |
| providers:02 | выполнено | change: Structured choice index requires nonnull zero with known usage preserved. Unknown-field evolution, exact decoded duplicate rejection, case aliases and surrogate normalization have separate explicit policies; host executable schema retained. Evidence: `adapters/openai/structured/decode.go`, `adapters/openai/structured/client_test.go`, `embedding/README.md`. |
| providers:03 | выполнено | change: Structured choice index requires nonnull zero with known usage preserved. Unknown-field evolution, exact decoded duplicate rejection, case aliases and surrogate normalization have separate explicit policies; host executable schema retained. Evidence: `adapters/openai/structured/decode.go`, `adapters/openai/structured/client_test.go`, `embedding/README.md`. |
| providers:04 | выполнено | change: Shared T11 URL/context helpers retained; ordinary structured Doer failures align with ErrUnavailable, body failures stay ErrProtocol. Shared UTF-8/control/blank credential admission runs in all constructors before dispatch. Evidence: `internal/providerhttp/credential.go`, `internal/providerhttp/credential_test.go`, `adapters/openai/structured/transport_regression_test.go`. |
| providers:05 | выполнено | change: Shared T11 URL/context helpers retained; ordinary structured Doer failures align with ErrUnavailable, body failures stay ErrProtocol. Shared UTF-8/control/blank credential admission runs in all constructors before dispatch. Evidence: `internal/providerhttp/credential.go`, `internal/providerhttp/credential_test.go`, `adapters/openai/structured/transport_regression_test.go`. |
| providers:06 | выполнено | change: Space remains authoritative; Model convenience equality assertion retained explicitly. Host attestation is separate from optional model echo. MaxVectorRows clean rename completed after inventory; shared defaults/explicit units table published. Evidence: `embedding/embedding.go`, `embedding/README.md`, `docs/task20/T18.md`, `adapters/jina/tensor/client.go`. |
| providers:07 | выполнено | change: Space remains authoritative; Model convenience equality assertion retained explicitly. Host attestation is separate from optional model echo. MaxVectorRows clean rename completed after inventory; shared defaults/explicit units table published. Evidence: `embedding/embedding.go`, `embedding/README.md`, `docs/task20/T18.md`, `adapters/jina/tensor/client.go`. |
| providers:08 | выполнено | change: Space remains authoritative; Model convenience equality assertion retained explicitly. Host attestation is separate from optional model echo. MaxVectorRows clean rename completed after inventory; shared defaults/explicit units table published. Evidence: `embedding/embedding.go`, `embedding/README.md`, `docs/task20/T18.md`, `adapters/jina/tensor/client.go`. |
| providers:09 | выполнено | retain: Wire byte/output admission follows allocation/extraction; no peak-memory/CPU guarantee. Host process isolation remains outside library, explicitly documented. Evidence: `embedding/README.md`, `adapters/pdf/README.md`. |
| providers:10 | выполнено | retain: MaxInputs includes query (default maximum127 documents); original input/validated prefix plus error is not successful reranking. Existing final access delivery and observed search-unit policy retained without helper/retry/fallback framework. Evidence: `adapters/cohere/rerank/client.go`, `adapters/cohere/README.md`, `embedding/README.md`. |
| providers:11 | выполнено | retain: MaxInputs includes query (default maximum127 documents); original input/validated prefix plus error is not successful reranking. Existing final access delivery and observed search-unit policy retained without helper/retry/fallback framework. Evidence: `adapters/cohere/rerank/client.go`, `adapters/cohere/README.md`, `embedding/README.md`. |
| providers:12 | выполнено | change: Independent valid usage precedes delayed raw model/vector/shape/index JSON decode and semantic admission; known usage survives malformed payload members/float32 overflow, rejected model/cardinality/vectors and postdecode cancellation without embeddings. Invalid/absent counters and malformed/truncated whole envelopes remain unknown. Shared optional model echo decoder has post-decode context gates. Evidence: `adapters/openai/dense/client.go`, `adapters/jina/dense/client.go`, `adapters/jina/tensor/client.go`, `adapters/gemini/internal/wire/client.go`, `adapters/openai/dense/usage_contract_test.go`, `adapters/jina/tensor/usage_contract_test.go`, `internal/providerhttp/model.go`, `adapters/gemini/internal/wire/usage_contract_test.go`. |
| providers:13 | выполнено | retain: Explicit purpose/identity/metric profiles, validation and float64 scoring retained. No speedup/optimization claim (before/after N/A); model routing, retries, pricing, agent modes and universal schemas stay outside adapters. Evidence: `embedding/README.md`, `dense/embedding.go`, `docs/task20/T18.md`. |
| providers:14 | выполнено | change: Recognized parser read/syntax errors are invalid input; only private declared limit/geometry exceptions select invalid/unsupported. Unexpected dependency OverflowError/NotImplementedError and implementation errors are internal unavailable without raw text. Actual runtime/dependency/fingerprint and 11-class fault profile published; no RSS/CPU claim. Evidence: `adapters/pdf/engine.py`, `adapters/pdf/normalize.go`, `adapters/pdf/engine_contract_test.go`, `adapters/pdf/testdata/verify_engine_errors.py`, `adapters/pdf/README.md`. |
| providers:15 | выполнено | change: Recognized parser read/syntax errors are invalid input; only private declared limit/geometry exceptions select invalid/unsupported. Unexpected dependency OverflowError/NotImplementedError and implementation errors are internal unavailable without raw text. Actual runtime/dependency/fingerprint and 11-class fault profile published; no RSS/CPU claim. Evidence: `adapters/pdf/engine.py`, `adapters/pdf/normalize.go`, `adapters/pdf/engine_contract_test.go`, `adapters/pdf/testdata/verify_engine_errors.py`, `adapters/pdf/README.md`. |
| providers:16 | выполнено | change: Current GoDoc and compilable client examples describe ownership, host identity, finite limits, context and known/unknown accounting. Deterministic examples dispatch no paid provider calls. Evidence: `embedding/README.md`, `adapters/openai/dense/example_test.go`, `adapters/jina/dense/example_test.go`, `adapters/jina/tensor/example_test.go`, `adapters/gemini/dense/example_test.go`. |
| providers:17 | выполнено | retain: Explicit purpose/identity/metric profiles, validation and float64 scoring retained. No speedup/optimization claim (before/after N/A); model routing, retries, pricing, agent modes and universal schemas stay outside adapters. Evidence: `embedding/README.md`, `dense/embedding.go`, `docs/task20/T18.md`. |
| providers:18 | выполнено | retain: Explicit purpose/identity/metric profiles, validation and float64 scoring retained. No speedup/optimization claim (before/after N/A); model routing, retries, pricing, agent modes and universal schemas stay outside adapters. Evidence: `embedding/README.md`, `dense/embedding.go`, `docs/task20/T18.md`. |

## Independent verification

Go `/opt/homebrew/Cellar/go/1.26.5/bin/go`, `GOTOOLCHAIN=local`; tests use `-race -count=1`. All commands completed exit0.

| Check | Current evidence |
|---|---|
| Root embedding/dense/tensor/providerhttp | `T18-completeness-revised-root-race.log` |
| OpenAI GOWORK=off including structured and example | `T18-completeness-revised-openai-race.log` |
| Jina GOWORK=off dense/tensor including examples | `T18-completeness-revised-jina-race.log` |
| Gemini GOWORK=off dense/shared wire/multimodal including example | `T18-completeness-revised-gemini-race.log` |
| Cohere GOWORK=off (unchanged after first acceptance run) | `T18-completeness-cohere-off-race.log` |
| Actual PDF complete GOWORK=off `-v` profile with explicit runtime | `T18-completeness-revised-pdf-actual.log`; PASS, no SKIP |
| Real imported PDF engine11exception classes | `T18-completeness-revised-pdf-errors.log` |
| Existing PDF/PNG independent fixture verifier | `T18-completeness-revised-pdf-fixture.log` |
| Actual runtime/dependency versions | `T18-completeness-pdf-runtime.log` |

PDF interpreter: `/Users/skosovsky/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`. Verified macOS27.0.1 arm64, Python3.12.14, pdfplumber0.11.9, pypdf6.10.0, pdfminer.six20251230, reportlab4.4.9. Actual malformed/geometry/limits, parse/chunk/project/BM25/scoped resolve and durable publication/reopen/typed modality paths ran, rather than treating fake executables as layout evidence.

Initial checks and logs retained. Final changed scope rerun after two review revisions: typed vector/model errors losing valid usage, and generic PDF OverflowError/NotImplementedError classified as user/unsupported input. Revised code and tests address both. Independent runtime/version and stale embedding symbol inventory agree with current contracts. `git diff --check` PASS.

## Limits

No billable live LLM calls, remote revision/model quality attestation or all-root/all-module final acceptance claimed. Optional paid profiles without opt-in are not counted as PASS or required here. Known unchanged production documentation blacklist failure is assigned T21.C01. Output/byte admission is not RSS/CPU isolation. Host schema, pricing, tokenizer, identity/protection and process isolation remain external ports. Dense metrics retained; no optimization, before/after measurements N/A. Unrelated iCloud duplicate untouched.

## Final candidate SHA256 manifest

39 changed source/test/current documentation files. Mutable plan/backlog/traceability, acceptance artifacts/logs and unrelated `docs/task18/correctness 2.md` excluded.

| Path | SHA256 |
|---|---|
| `adapters/cohere/README.md` | `af99f8c83b2e5c5cdb59d3ff354ac00503ab48d9be7cda664ec1d05157fe2153` |
| `adapters/cohere/rerank/client.go` | `87094fb5e94b3b6494baf6f0af2f662a0be7c8e8b0e30e08c92e9abec90f8933` |
| `adapters/gemini/README.md` | `dbee717a7b1865e4143a912b82423c311a268045820bd29fe8b38d4ec6b7e1d7` |
| `adapters/gemini/dense/client.go` | `b7113caea4abac1978ac2832a607f69a60e8535812265451b3cc1c1897ef5f30` |
| `adapters/gemini/dense/example_test.go` | `1a9cd65e7ac92a359463665d1cd143b1662d2795e50c7c7815cccdcad300c79c` |
| `adapters/gemini/dense/usage_contract_test.go` | `97d34622e74b3f031feaaa339b88c81a8b1876eb26fb561796135d7fe4387d47` |
| `adapters/gemini/internal/wire/client.go` | `0a08ec254065b4c26dccc44699a905fe0907650b6d063657dd3509402724a2c9` |
| `adapters/gemini/internal/wire/usage_contract_test.go` | `15f98efe16be3294ebc3a647f875a2bd0b45cc339fbdffde7cf47874ef19987a` |
| `adapters/gemini/multimodal/client.go` | `6faf6b70a41c9b1cd1b3c764879f1ffa17c2d3671eba8ef6b0d55c0a7b81ecc9` |
| `adapters/jina/README.md` | `5384a5328f08c91c5abe6dffe8475b08a42bdf9bdd12b82d14a3bb3486d81f28` |
| `adapters/jina/dense/client.go` | `68734a463b967ac0d0c67acdeccb6789287f3396eb0e244b648a5b5d3f8e31d3` |
| `adapters/jina/dense/example_test.go` | `ba621ee6aec4c7886a1d5db1da52bb7102a2395f42d48bd3e086c306021a75db` |
| `adapters/jina/dense/usage_contract_test.go` | `943bf49934870c5a21d9a46c978deb4e9364409aa1955799de1ddcd67c38f110` |
| `adapters/jina/tensor/client.go` | `23a031f24ad286ce9adcb04da6b6930f3d331c1dbb92c765ab0474c9ef33b889` |
| `adapters/jina/tensor/client_test.go` | `8718505b0f66e1e35841a2b37cb43afdc8e44106305942131c8547e60549ba68` |
| `adapters/jina/tensor/example_test.go` | `c85183924ec17d041342717e88c25eeaa80d307cce587810a24d03957bb605ea` |
| `adapters/jina/tensor/usage_contract_test.go` | `372000cae3211873218b58dc266cdc8dbf398c47847ff05f2f93ddccd23f3793` |
| `adapters/openai/dense/README.md` | `41ee5f379f138e83bf01cc8cbce615bf5f2b5e50038e8d10c7a0507232ee4133` |
| `adapters/openai/dense/client.go` | `3a6a1cb66f7afbf627d25406bb6b57b358cfbd450518473780bccf988fc2feb3` |
| `adapters/openai/dense/example_test.go` | `1a90ee487f70eeff975660227b11968e771b939e39473fded00686818f22bfde` |
| `adapters/openai/dense/usage_contract_test.go` | `4b4a37275e7a0895b769467e577da12d724f24d1e517a4d355a4a84e17accbcb` |
| `adapters/openai/structured/README.md` | `c95fef3b0c44da09cb72d112a1c3050cdbfcdecb08c12ba36f0a0b9bb2f497e0` |
| `adapters/openai/structured/client.go` | `83925bb3643aac9c0eda0698a31e10a683b003d6125546cfe413db83e6e3d51a` |
| `adapters/openai/structured/client_test.go` | `50dd98beb238cd4c23e1eaed0a033da9e20e847d5cf7399b01951e2d5eb32424` |
| `adapters/openai/structured/decode.go` | `36031e99d979d02a47a955371e66a1a8b2ecf67a91cf058b7e3affb25d627e4c` |
| `adapters/openai/structured/transport_regression_test.go` | `fc1e62edd0acceda3dacde9b1152a10ab687ea302858a33416c15132b458169e` |
| `adapters/pdf/README.md` | `9c3bd9b18c78003c8d82d7457b0566892f6611a482114d89ebe165a6e2d57310` |
| `adapters/pdf/engine.py` | `4fb8ed59713ea14a95c0c43a73cf5f904c3d78fcbc5a475c1e0b486132fb1079` |
| `adapters/pdf/engine_contract_test.go` | `06c9212a8f8e12e88d250891e3154eb0f59908b2f88a6c0dc889e70d877d895b` |
| `adapters/pdf/normalize.go` | `09ba406be77c05fa725e9402568dfe992a2b384645c56b79ec1e787f4a36b3d0` |
| `adapters/pdf/testdata/verify_engine_errors.py` | `10b2fabcc3b058dd160959ebbd71397982e9d9b3f9c6d6289befe9089624daeb` |
| `docs/task20/T18.md` | `bf14d5d3d2cd6dcf7945a69a5d0863738f5def19c165f7334f3aab009be613c8` |
| `embedding/README.md` | `97284ca0bc51986f3322eb0513e21376e758d5b4c0d299bf7c88b91ecaacfcfc` |
| `embedding/embedding.go` | `5a2f58d7134b523f4ad3e8c6f1ce130b202c1a3906687e93481f467a8bb91661` |
| `internal/providerhttp/client_test.go` | `bed96459e9a4703eaf60022ba56d00fa519627c419b6c37b88fd6affc12e59ce` |
| `internal/providerhttp/credential.go` | `c2b619d047f1a61d6828374ab3af461526c4022a2f034b90844430c1b271054d` |
| `internal/providerhttp/credential_test.go` | `68467f708b2f0d7498085e47d25f6dc79975708cde4ec3d4ffcc20493731fbd5` |
| `internal/providerhttp/json_policy_test.go` | `a5a6d71ebc58336d5f99d640d14503a53b6e831f4731518c951c481851b8e114` |
| `internal/providerhttp/model.go` | `6e03363e842445d744055311111baa5740797f7aa817c02c514c6dd55c01025f` |
