# T10 correctness acceptance

Verdict: **PASS**. No actionable correctness findings in the reviewed T10 scope. Independent acceptance; no implementation edits and no other reviewer report read.

Baseline HEAD: `a2fc0a48ed24112cf3a1579e153ce2936454effb`.

Reviewed F09, storage S2, T10.C01–C03, normative delivery contract and the public/private split. Every private return passes DeliverRead; cancellation suppresses successful, empty and ordinary partial results, without a second traversal. Ordinary valid-context projection prefixes and native traversal rank/ScoreAbsent remain intact. When context cancellation coincides with ordinary Runner/projection failure, scoped readfailure.Join preserves the observed cause through errors.Is while the private callback is excluded from errors.As and unwrap traversal. Independently supplied ProtectionError values retain the global access.Protect cause contract; their own host-supplied causes are not newly sanitized. A private sibling joined outside that protected branch remains hidden by the scoped join. The final README/T10 clarification accurately states this boundary. Global access.Protect behavior is unchanged. Scoped/pinned rejection still precedes traversal. Traverse/Upsert administrative behavior is unchanged.

## Independent verification

- Fresh `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./adapters/neo4j/... ./retrieval/... ./internal/readfailure/...`: PASS all three packages (1.285s, 1.380s, 1.221s).
- Fresh `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run --allow-serial-runners ./adapters/neo4j/... ./retrieval/... ./internal/readfailure/...`: 0 issues.
- Independent overlay-only probes: `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -overlay=/private/tmp/ragy-t10-correctness-overlay.json -run '^TestReviewT10' -count=1 -v ./adapters/neo4j/...`: PASS three adversarial probes (2.391s). Actual parent deadline expires while Runner waits on ctx.Done; a protocol error returned simultaneously retains both DeadlineExceeded and ErrProtocol with empty protected output and one traversal. A valid-context Runner independently joins ProtectionError(ErrUnavailable) with a private typed protocol sibling: empty output, both classifications and sibling identity preserved through errors.Is, private type/text hidden. A complete-empty pinned publication rejects before Runner dispatch with protected ErrUnsupported. The initial two overlay probes omitted required TopK and correctly failed admission; after correcting the review fixture, all actual intended paths passed. Repository implementation unchanged.
- Existing success/empty/partial/cancel and fetch-limit tests inspected; `git diff --check`: PASS.

No SKIP used. Live Neo4j profile is not applicable to this change: the package is a typed host Runner bridge, this diff adds local final delivery only and changes no Cypher, driver operation or remote authorization enforcement. The source review expressly identifies a local cancellation inconsistency and excludes tenant revocation enforcement. These checks do not certify arbitrary host Runners or production Neo4j. The separately reported broad root documentation blacklist failure is assigned T21; this report makes no all-repository PASS claim.

## Reviewed substantive file SHA256

| File | SHA256 |
|---|---|
| `adapters/neo4j/neo4j.go` | `a1d2ead44fcd93114e36956cc73724e27fa5eb5bf77a5f24bb1a527f888af407` |
| `adapters/neo4j/neo4j_test.go` | `e89349e8efc1f4a30d3bc3cfd14018c75671e4bc2872c8be824442473bdd226c` |
| `adapters/neo4j/delivery_test.go` | `86637d3587cee2c9de33ec16115656177533d0e3c50618876125b7220d6f1bb6` |
| `adapters/neo4j/README.md` | `221c5a6bc7b5f4c6c1eed746ad402d05369d061ecfc4f2bbe98bf0a93204bfdd` |
| `docs/contracts/remediation.md` | `2c11991ebda8865f3a69c5636d3a54aac637008c0654d28b95bf191d4f9664f0` |
| `docs/task20/T10.md` | `a4a21c18956bd7cd21d157d392b3da7f05197811008764e8ad02d43a0ed6e62b` |
| `internal/readfailure/callback.go` | `0ffbb31ad96c96548e5c6840b5eda9e9c974d25ff246bd452781b956a8693913` |
| `retrieval/access.go` | `c1d6f830b8a27955d3070c86eeeeee70d55faf2c2c885cb7bec8f7b4329b5d6a` |
