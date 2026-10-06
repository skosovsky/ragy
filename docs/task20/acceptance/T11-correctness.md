# T11 correctness acceptance

Baseline: `e960f9e645433e1b8b93d95b18797a8e5156fa47`. Independent read-only review; no implementation participation. Verdict: **PASS**, no remaining actionable correctness findings in T11 scope.

Reviewed current diff and target contract against master F10/F11 and provider P-01/P-02/providers:01. Shared parsed URL admission rejects ForceQuery, query, credentials, opaque and fragment before dispatch. Parsed escaped path construction preserves host and encoded segments with one separator; embedding providers retain their envelope semantics. Shared SanitizedError prefers current request context and wrapped public context sentinels without exposing private transport text. Structured exchange closes owned response bodies, checks cancellation at headers and body boundaries, returns zero output/unknown usage for incomplete transport and performs exactly one dispatch. Existing complete-envelope usage/refusal/schema/redirect controls remain covered. Retained ordinary transport classes differ deliberately by adapter; no public transport framework or hidden retry introduced.

Independent fresh checks:

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./internal/providerhttp/... ./adapters/openai/... ./adapters/gemini/... ./adapters/cohere/... ./adapters/jina/...` — PASS, eight tested packages; Gemini internal wire has no tests and is not represented as test coverage.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run --allow-serial-runners ./internal/providerhttp/... ./adapters/openai/... ./adapters/gemini/... ./adapters/cohere/... ./adapters/jina/...` — exit 0, 0 issues (exhaustruct deprecation warning only).
- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 -overlay=/private/tmp/t11_correctness_overlay.json -run '^TestT11Independent' -v ./adapters/openai/structured` — PASS, four independent adversarial cases. Overlay source `/private/tmp/t11_correctness_probe.go` (no workspace implementation edits): canceled HTTP 500 headers must win before reading body; cancellation accompanying complete JSON/EOF must return unknown usage rather than decode; cancellation accompanying oversized bytes must win over protocol size classification; shared Post dispatch preserves encoded slash/hash/question-mark in base and encoded slash in endpoint. Each relevant case asserts one dispatch, zero typed output/usage and closed body. Shared Post asserts decoded valid control and exact host/path/no query.
- `git diff --check` — PASS.

These deterministic HTTP semantics require no paid live provider calls. No SKIP is counted as PASS. Scope is T11, not all-module acceptance; deferred global documentation-wording check belongs to T21. Package GoDoc wording was refreshed during acceptance without executable changes; inspected current digest below.

Reviewed SHA256 (before final administrative acceptance updates):

| File | SHA256 |
| --- | --- |
| `docs/contracts/remediation.md` | `2c11991ebda8865f3a69c5636d3a54aac637008c0654d28b95bf191d4f9664f0` |
| `docs/task20/T11.md` | `f85abd9fa1220f48a0c36a487d2ddda57d43097dd65f397c4ed23efb531f0cc9` |
| `internal/providerhttp/client.go` | `069368175d0e25b3444c1f5f7b0ee0e91c11b1e218906d08f5f87125c7d65175` |
| `internal/providerhttp/url.go` | `10c1ad743d150a0c6140bca5745871ef57944b12cb0766e656508dd2da88d898` |
| `adapters/openai/structured/client.go` | `fac5c0ea3d3ea31fe98a23103da0c6eccb25926c5d7bb5b52451085dc2a2f6ef` |
| `adapters/openai/structured/transport_regression_test.go` | `e1941dab198dc457cf9e223ae31d9988d0920f0a4ca9a2035637d40a9540c9d1` |
| `adapters/openai/structured/README.md` | `42879443ab112945fe742a106ed2b26b842c044465507c415f01df13559aa868` |
