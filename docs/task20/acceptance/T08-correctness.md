# T08 — independent correctness acceptance

Verdict: **PASS — no unresolved errors found**. Baseline: `dd81b21abe6e49f1f52b4c03c09c5496a6f25c83`.

I did not participate in implementation and did not read the completeness report. Repeated the full lexical implementation/test/doc review against F07 / T-F01, T08.C01–C03 and the normative callback contract after the initial P2 finding was fixed.

## Resolution of the initial P2

The initial code erased an ordinary callback cause when a post-callback read gate failed, contradicting the normative retention promise. The scoped `internal/readfailure` boundary now wraps gate + callback under fixed-text protection, preserves callback identity/classification through errors.Is, and exposes only gate errors through Unwrap/errors.As. Global access.Protect is unchanged. Every changed Encode/clone completion uses this boundary; final raw/snapshot/managed delivery gates preserve an already observed cause. Contract/README explicitly describe the scoped errors.Is retention and private sibling sanitization.

Repeated and expanded the independent failing probe using typed private callbacks wrapping ErrProtocol. Six actual cases: snapshot capture, metadata-field indexing, query filtering, output cloning, managed cache miss, managed cache hit. Each asserts protected cancellation, errors.Is(original typed error) and errors.Is(ErrProtocol), no typed private errors.As or private text, zero output, and exact callback counts stopping before the next payload callback. All six PASS on current code. Probe files `/private/tmp/ragy-t08-correctness/probe_test.go` and `managed_probe_test.go`, overlay `/private/tmp/ragy-t08-correctness/overlay.json`; command:

`GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -overlay=/private/tmp/ragy-t08-correctness/overlay.json -run '^TestT08Independent' -count=1 -v ./lexical ./lexical/managed`

PASS lexical 1.526s, managed 1.921s. This directly closes the previous FAIL and additionally checks privacy at public adapter boundaries.

## Full scope review and checks

- F07/T-F01/C01: scoring has current context/Binding gates before and immediately after each MatchDocument Encode; failed or denied matching still reaches the completion gate. Protection discards partial candidates before ranking/output cloning. Capture matching and every metadata-field build Encode are gated. The temporary builder codec is removed before publication and no authority decision is cached. Existing clone gates plus corrected cause retention stop further payload work.
- C02: current permanent AAA matrices cover snapshot capture/index/retrieve cancellation/revocation with optional ordinary ErrProtocol, managed miss/hit cancellation/revocation with optional ErrProtocol, ordinary clone failure plus revocation, explicit protected second-candidate error, and valid ranking controls. Independent private-cause probes cover both cache paths and all snapshot callback phases. Existing ordinary filter-prefix error policy remains unchanged; deterministic rank/score ownership controls pass.
- C03: raw BM25 metadata remains borrowed under host stability/concurrency contract; owning snapshots retain explicit CloneMeta and owned output. Construction context is not retained, confirmed by valid retrieval after cancellation of the build context. No retries, background workers, backend/parser changes or score/cache policy changes were introduced.

Fresh independent `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./internal/readfailure ./lexical/... ./retrieval/...`: PASS (readfailure 1.252s, lexical 1.293s, managed 4.537s, retrieval 1.640s). Fresh independent targeted golangci-lint: 0 issues. `git diff --check`: PASS. No required check was skipped. Live backend/parser profile is inapplicable to these in-memory callback changes; no performance optimization is claimed.

## Reviewed SHA256

- `docs/contracts/remediation.md`: `2c11991ebda8865f3a69c5636d3a54aac637008c0654d28b95bf191d4f9664f0`
- `internal/readfailure/callback.go`: `0ffbb31ad96c96548e5c6840b5eda9e9c974d25ff246bd452781b956a8693913`
- `internal/readfailure/callback_test.go`: `c40f3e98f29a0c70cb59eb80bfabbdebebe73b0c5509a9553976097f11da2297`
- `lexical/README.md`: `8fa6f7567dd7329c14c9cc33e8a9b99f10f7b8d0c11886722469ffed874dd9e2`
- `lexical/bm25.go`: `3f85b325e080a4230d4ad29bbad30f2b8ee849c0ccd5722e7b1059c4f26a600e`
- `lexical/bm25_test.go`: `73d0b8c399a170f22050fe1ee77877751d5d742be3f1e1d5f98011be5986e5fd`
- `lexical/snapshot.go`: `d09c40acee698f4fbf9c09d6493ac84e5b504d69d9ad2141cc083fde1cb7868c`
- `lexical/snapshot_codec.go`: `45331225102232b1f135968d9db9446adeab31f03e632b6fdeba41898328d478`
- `lexical/callback_gate_test.go`: `35c911b0734e01be036c62491ed4b8e8a786214157963d8e128694fd9db5460f`
- `lexical/managed/README.md`: `b15d62b93f255a7d1f5cca61a8bc4f092a50b43179682ba622251f118b39142b`
- `lexical/managed/managed_test.go`: `abb2a3acffd23ff4a9efc201c6b02b8aea04cf3224b35472c46732b32c92891b`
- `lexical/managed/read.go`: `a717f564615338593d5f8d3efed0c9fb5ec3ea0d3b1a731a3ec4d1e6bcfc39b9`
- `lexical/managed/callback_gate_test.go`: `46bffafcc9552f0109457f213983142df9e055e310f759e08b72fff96de244cf`
