# T07 — независимая полнота

Baseline HEAD: `2b9d3c5918764b61488f2adfc245a92db70060c4` (подтверждён `git rev-parse HEAD`). Reviewer: `/root/t07_completeness`; implementation не изменялась. Проверены T07.C01–C03, immutable F06 / ING-F01, Source addressing contract и весь layout diff. Отчёт correctness не читался.

Вердикт: **100% (3/3 completed)**. Assigned-source coverage: **100% (2/2)**. Not completed: 0; blocked: 0. SKIP не засчитан как PASS.

| Критерий | Статус | Evidence |
|---|---|---|
| T07.C01 | completed | Document.Validate использует set полного comparable source.Reference для всех страниц. Повтор отбрасывается до projectLayout и ImageText. Permanent AAA: equal/different equal-length/different-length/empty/nonempty text × complete/partial coverage, ErrInvalidArgument, nil projected output, zero ImageText. Независимый non-adjacent duplicate на третьей странице также отвергнут до callback. |
| T07.C02 | completed | Нормативный контракт и layout README определяют distinct complete cell/image selectors одного immutable whole artifact, включая cross-page sharing. Permanent fixture: 2 pages × 2 cells/images, один artifact, 10 уникальных projected IDs, точные source supports и 4 ImageText calls. Duplicate logical cell/full image locator отвергается до callback. Production identity/framing/hash не менялись; map key exact Reference, не Artifact-only. Независимый control с одним Artifact и разными Representation проходит и даёт разные IDs. Built-in Resolver limitation явно описан; arbitrary host selector loader не сертифицируется. |
| T07.C03 | completed | Distinct page references проходят scoped Catalog→admitted Loader, возвращают AAA/BBB независимо: distinct IDs, exact citation mapping/location, единственный batched load, без latest fallback. GoDoc Page/Document.Validate/Project и package README согласованы с контрактом. Свежие targeted race 4 packages и lint 0 issues. |

F06 и ING-F01 закрыты admission до projection; lengths/hash collision не служат условием отказа. Cross-kind original selectors явно отделены от unique normalized page text addresses; разрешённое sharing не отменяет host attestation. Shape validation не выдаётся за source authorization.

Проверки reviewer:

- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 ./layout/... ./source/... ./chunking/... ./adapters/pdf/...`: exit 0, все 4 packages PASS, без SKIP.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-task20-lint-t07-completeness golangci-lint run --allow-serial-runners ./layout/... ./source/... ./chunking/... ./adapters/pdf/...`: exit 0, `0 issues.` Exhaustruct deprecation warning не finding.
- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 -overlay=/private/tmp/ragy-t07-completeness/overlay.json ./layout -run TestT07Independent -v`: exit 0; `TestT07IndependentNonAdjacentDuplicateRejects`, `TestT07IndependentSameArtifactDifferentRepresentationAllowed` PASS. Overlay append использует permanent fixtures, не меняет tracked implementation/tests. Probe и overlay сохранены в указанном tmp directory.
- `git diff --check`: exit 0.

Фикс публичного layout admission не меняет PDF parser implementation; actual parser/service profile для этого scope не применим. Выполнен штатный PDF adapter regression, live parser acceptance не заявляется.

## SHA256 reviewed substantive files

| Path | SHA256 |
|---|---|
| `layout/layout.go` | `8322c6ca2eaabffc255d0e73311d99402dafaa62d437dd2e160a15c4377dd63a` |
| `layout/project.go` | `9465b2f2d2d84c3c89b0e679e8fe7bdb24ee5bfe7c8b88d72d9924f1a9b1162d` |
| `layout/README.md` | `badd21c889519ef44952c0c16cc0621723492104fd27666f1590460d280cd18f` |
| `layout/reference_test.go` | `d4ea1e33a7ddb839e58152f57a17c8b8b5848f4d9ed80002d15e04c1aa48bd7c` |
| `docs/contracts/remediation.md` | `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df` |
| `docs/task20/T07.md` | `cdc91dfc4b81941d3fe0adf57bf3df6fe3902676be2f0a433f688dc8df9a5408` |
| `source/locator.go` | `e040bbf1726130e275712b18763ea82c257f2c6e1c75dbbe1f0d5a5c3c1366eb` |
| `source/reference.go` | `e2ca9559a73359fdadec777651bfe83bc17b26a21691c87779e06db1a0e58021` |
| `layout/resolve.go` | `24e17708a7703e11ebc158f44b9b42b74915fb2d80286e199fe73f63c5b0e0c3` |
| `docs/task20/review-baseline.md` | `66dba45fc74dbdd8c0d1251db5ff4f92f8272d76b93a9a1e9d2c70baa47b3cc6` |
| `docs/task20/reviews/ingestion.md` | `904abaeb28003225342feb5e5735faf8b4294c15b0bd25e5d03b1ab8d0415985` |
