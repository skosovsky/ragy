# T07 independent correctness acceptance

Verdict: **PASS**. No actionable correctness defects found in the reviewed T07 scope. Reviewer did not participate in implementation and did not read the completeness report.

Baseline HEAD: `2b9d3c5918764b61488f2adfc245a92db70060c4`.
Scope: T07.C01–C03, F06 and immutable ingestion review ING-F01; normative Source addressing contract in docs/contracts/remediation.md.

Reviewed SHA-256:

- `layout/layout.go`: `8322c6ca2eaabffc255d0e73311d99402dafaa62d437dd2e160a15c4377dd63a`
- `layout/project.go`: `9465b2f2d2d84c3c89b0e679e8fe7bdb24ee5bfe7c8b88d72d9924f1a9b1162d`
- `layout/reference_test.go`: `d4ea1e33a7ddb839e58152f57a17c8b8b5848f4d9ed80002d15e04c1aa48bd7c`
- `layout/README.md`: `badd21c889519ef44952c0c16cc0621723492104fd27666f1590460d280cd18f`
- `docs/contracts/remediation.md`: `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df`

Document.Validate uses the full comparable source.Reference as its uniqueness key. It rejects nonadjacent duplicates as well as adjacent duplicates, independent of text content, empty/nonempty text, length, page geometry and partial coverage. Project completes document validation before entering page projection or ImageText. Rejection retains ErrInvalidArgument and returns nil projected payload. No location identity/hash changes mask ambiguous addressing.

Reviewed sharing controls preserve distinct cell/image complete selectors, reject duplicate logical cells/full image locators, and keep normalized page references unique. Documentation explicitly limits the built-in layout.Resolver to one retained original per reference; whole-artifact selector loaders and content attestation remain host-owned. The distinct page citation regression traverses admitted Catalog/Loader resolution and checks original text and complete citation locations.

Independent checks (all executed, no skipped acceptance checks):

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./layout/... ./source/... ./chunking/... ./adapters/pdf/...`: four packages PASS, exit 0.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-cache golangci-lint run --allow-serial-runners ./layout/... ./source/... ./chunking/... ./adapters/pdf/...`: 0 issues, exit 0.
- Independent public API overlay probe `/private/tmp/ragy-t07-correctness/probe_test.go`, overlay `/private/tmp/ragy-t07-correctness/overlay.json`: `go test -race -count=1 -overlay=... ./layout -run TestT07Independent -v`, PASS, exit 0. Three ordered pages with a nonadjacent duplicate reference and an image on the first page reject before that callback. Changing only the final page's representation gives a lawful distinct-reference control: four outputs, one ImageText callback, distinct page identities. Unicode equal-length text verifies the duplicate decision does not depend on ASCII spans.
- `git diff --check`: exit 0.

An initial command used a nonexistent adapter path and a default unwritable lint cache; these command setup failures were corrected and the full intended scopes above rerun successfully. No implementation edits were made by this reviewer.
