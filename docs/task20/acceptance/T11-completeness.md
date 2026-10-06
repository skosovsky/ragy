# T11 independent completeness acceptance

Baseline: `e960f9e645433e1b8b93d95b18797a8e5156fa47`. Independent read-only acceptance; no implementation participation. Reviewed current diff, immutable master F10/F11, providers review P-01/P-02 and providers:01, all four backlog criteria, normative remediation contract and package documentation.

| Criterion | Status | Evidence |
|---|---|---|
| T11.C01 | выполнено | Structured exchange closes response body, prioritizes current request context and wrapped cancellation/deadline at transport/body failures, gates received headers before status, returns zero output/unknown usage on incomplete transport, performs one dispatch. Eleven permanent phase regressions plus independent canceled non-2xx headers and partial valid-envelope body probes PASS. |
| T11.C02 | выполнено | Shared ParseBaseURL rejects ForceQuery, query, fragment (also bare fragment), credentials, opaque/missing host/unsupported schemes before dispatch. Parsed Endpoint preserves custom path, escaping and separator. Nine invalid and five valid permanent cases plus actual shared escaped-path dispatch with zero/one trailing slash PASS. |
| T11.C03 | выполнено | Both structured and shared provider clients consume the same internal Endpoint/ParseBaseURL/SanitizedError policy; no public transport framework. Distinct response-domain logic and existing ordinary transport fallback classes remain explicit. Common-policy portion of D55 is covered; broader D55/provider decisions remain assigned T18. |
| T11.C04 | выполнено | Fresh required race and lint passed; permanent AAA cases cover body local deadline, parent cancellation, ordinary private I/O, valid usage and URL matrix. Existing complete-envelope refusal/schema/overrun, bounds and redirect controls passed. README and T11 contract align with implementation. |

Criteria completeness: **4 / 4 × 100 = 100%**.

| Assigned source item | Status | Evidence |
|---|---|---|
| F10, immutable master | выполнено | Context-aware body deadline and parent cancellation preserve public sentinels, close body and suppress incomplete bytes/usage with no retry. Ordinary private I/O remains sanitized ErrProtocol. |
| F11, immutable master | выполнено | Bare query rejected as ErrInvalidArgument before network; parsed path construction preserves /v1/chat/completions and custom paths. |
| providers:01, providers review line 36 | выполнено | Small shared internal URL/error helpers prevent actual policy drift without forcing structured usage/schema semantics into embedding transport. |
| P-01, providers review | выполнено | Headers cancellation checked before HTTP status; wrapped context and incomplete body bytes tested independently with zero output and unknown usage. |
| P-02, providers review | выполнено | ForceQuery and invalid-base matrix plus normalized parsed endpoints meet original admission/path requirements. |

Assigned source coverage: **5 / 5 × 100 = 100%**. No unfulfilled or blocked criterion, and no acceptance finding in assigned scope.

Independent verification:

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./internal/providerhttp/... ./adapters/openai/... ./adapters/gemini/... ./adapters/cohere/... ./adapters/jina/...`: exit 0, eight tested packages PASS; Gemini internal/wire has no test files.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run --allow-serial-runners ./internal/providerhttp/... ./adapters/openai/... ./adapters/gemini/... ./adapters/cohere/... ./adapters/jina/...`: exit 0, 0 issues; existing exhaustruct deprecation warning only.
- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 -overlay=/private/tmp/t11-completeness-probes/overlay.json -run '^TestIndependentT11' -v ./adapters/openai/structured/...`: exit 0, all three independent tests and nested cases PASS. Probes cover cancellation at non-2xx headers before body read; partial valid bytes with cancel, wrapped deadline or private ordinary error; actual shared escaped-path request construction; invalid endpoint query/fragment/authority. Overlay source exists only in /private/tmp and changes no implementation.
- `git diff --check`: exit 0.

Scope limits: paid provider smoke tests are existing explicit opt-ins and were excluded; their SKIP is not counted as acceptance PASS or live certification. The original F10/F11 profile is deterministic HTTP lifecycle/configuration semantics, requiring no paid model call. All mandatory T11 deterministic profiles executed. The known root documentation blacklist issue is assigned T21; this report makes no all-module PASS claim. Final providerhttp package GoDoc clarification was reread; executable files remain the independently tested versions.

## Reviewed SHA256

| File | SHA256 |
|---|---|
| `adapters/openai/structured/client.go` | `fac5c0ea3d3ea31fe98a23103da0c6eccb25926c5d7bb5b52451085dc2a2f6ef` |
| `adapters/openai/structured/transport_regression_test.go` | `e1941dab198dc457cf9e223ae31d9988d0920f0a4ca9a2035637d40a9540c9d1` |
| `adapters/openai/structured/README.md` | `42879443ab112945fe742a106ed2b26b842c044465507c415f01df13559aa868` |
| `internal/providerhttp/client.go` | `069368175d0e25b3444c1f5f7b0ee0e91c11b1e218906d08f5f87125c7d65175` |
| `internal/providerhttp/url.go` | `10c1ad743d150a0c6140bca5745871ef57944b12cb0766e656508dd2da88d898` |
| `internal/providerhttp/client_test.go` | `b8e13c918cf5d4909f29d19562a2d72fb96dfcfef74471b3892cdf5f7304e9e5` |
| `adapters/openai/structured/client_test.go` | `18779a96eb486722710ccce7715b4145cdd31fdf316896bdbc034ac82f1da20f` |
| `docs/contracts/remediation.md` | `2c11991ebda8865f3a69c5636d3a54aac637008c0654d28b95bf191d4f9664f0` |
| `docs/task20/T11.md` | `f85abd9fa1220f48a0c36a487d2ddda57d43097dd65f397c4ed23efb531f0cc9` |
