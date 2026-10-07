# T20 verification attempts

- Initial local `go run` using the default shared Go build cache failed with `operation not permitted`; rerun with task-owned `/tmp/ragy-t20-go-cache`, GOWORK=off and intact Go1.26.5 passed. No successful run inferred from the cache error.
- Initial new-example lint found one mnd literal (TopK=3). Named resultLimit added; final same scoped lint passed, 0 issues. The exhaustruct deprecation warning is a tooling warning, not a source failure.
- Naive Markdown link regex initially matched Go syntax inside the README code fence. Verification excludes code fences and validates actual links.
- Naive historical-byte comparison failed on process-costs.csv: four baseline CSVs already have CRLF worktree content versus LF Git blobs. No edits were made to historical files. Final comparison verified unchanged tracked content and explicitly records the four normalizations in docs-check.log; 454 baseline files preserved.
- Fresh `go test -count=1 -race ./examples/local-bm25 ./lexical ./retrieval` passed. Quickstart output is the single admitted acme document. README code equals the executable source; gofmt and git diff --check pass.
- Existing all-root broad wording-blacklist failure is assigned T21.C01; this task claims scoped checks only, not all-root PASS.

Commands use `GOCACHE=/tmp/ragy-t20-go-cache GOWORK=off GOTOOLCHAIN=local /opt/homebrew/Cellar/go/1.26.5/bin/go`; lint also uses `GOLANGCI_LINT_CACHE=/tmp/ragy-t20-lint-cache` and that Go bin first in PATH. Local runtime/backend quality boundaries are documented without new remote dispatch.
