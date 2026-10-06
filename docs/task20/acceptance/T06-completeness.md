# T06 — независимая полнота

Baseline HEAD: `fd70ce6fc493620ec33448a40aacc7c19011f65e` (подтверждён `git rev-parse HEAD`). Reviewer: `/root/t06_completeness`; реализация не изменялась. Проверен весь актуальный T06 diff, включая новые identity implementation/tests, контракт Identity and Unicode и immutable F05 / G-F02. Отчёт correctness не читался.

Вердикт: **100% (3/3 критериев completed)**. Assigned-source coverage: **100% (2/2)**; F05 и G-F02 закрыты strict UTF-8 admission без изменения framing. Not completed: 0; blocked: 0. SKIP не засчитан как PASS.

| Критерий | Статус | Evidence |
|---|---|---|
| T06.C01 | completed | New проверяет ontology/policy; admit проверяет полный structural batch IDs/names/optional namespace/relation IDs/endpoints до support/clone/identity callbacks. validateDecision и relationPolicy проверяют UTF-8 до identityID/grouping. Ошибки дают полный zero Result. History validates metadata, direct input, result/group/unresolved/trace identities до authorization/serialization и повторно при decoded inventory admission. Required/optional fields и state semantics согласованы. |
| T06.C02 | completed | Permanent AAA tests: ff/fe в 12 input, 4 config, 8 host policy случаях; 32 history fields x 2 malformed inputs, empty snapshot и zero admissions. U+FFFD допустим, equal/distinct tuples, namespace/key boundaries, case, composed/decomposed Unicode и relation-key grouping проверены. Ожидаемые valid IDs вычисляются прежним JSON tuple + SHA256 алгоритмом; production identityID не изменён. Независимая overlong UTF-8 late-decision проверка также PASS. |
| T06.C03 | completed | Resolver/Config/Entity/Relation/Decision GoDoc и resolution/history README описывают error classification, domain, optional fields, no normalization, retained hash compatibility и quarantine legacy repaired records. Свежие targeted race tests пяти graphingest packages PASS; lint 0 issues. |

Сверка источников: F05 требует explicit domain, namespace/name/policy/history consistency и valid compatibility — выполнено. G-F02 требует invalid host outcomes до hashing/grouping, U+FFFD, tuple boundaries, equal/distinct keys и relation path — выполнено permanent tests и независимым probe. Дальнейшие graph README roadmap правки относятся к T14, текущий stale абзац не является пропуском T06.

Проверки этого reviewer:

- `go test -race -count=1 ./graphingest/...`: exit 0, все 5 пакетов PASS, без SKIP.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-t06-completeness-lint golangci-lint run --allow-parallel-runners ./graphingest/...`: exit 0, `0 issues.` Инструмент сообщает deprecation warning exhaustruct, не finding. Предварительные запуски без tmp cache дали sandbox/cache initialization и lock/loading failures; PASS заявлен только после успешного полного запуска с явными cache paths.
- `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 -overlay=/private/tmp/ragy-t06-completeness-probes/overlay.json -run '^TestIndependent' -v ./graphingest/resolution ./graphingest/resolution/history`: exit 0, оба tests PASS. `TestIndependentLaterMalformedDecisionStops`: второе host решение содержит overlong `f0 80 80 80`, ErrProtocol, полный zero result, ровно 2 identity callbacks. `TestIndependentRawJSONCannotRepairIdentityBytes`: valid JSON framing с raw ff отвергается до decode assignment, explicit valid U+FFFD декодируется faithfully. Overlay и probe sources сохранены в `/private/tmp/ragy-t06-completeness-probes`; production/test tree не изменялся ими.
- `git diff --check`: exit 0.
- Versions: Go `go1.27.1 darwin/arm64`; golangci-lint `2.14.0`, built with go1.27.1.

Live backend/profile не применим к этому pure resolver/history identity изменению. BYOT attributes/kinds fidelity остаётся явным host contract; review не заявляет проверку произвольных custom codecs.

## SHA256 reviewed substantive files

| Path | SHA256 |
|---|---|
| `docs/contracts/remediation.md` | `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df` |
| `docs/task20/T06.md` | `d54fa56e967b062ef847b47f4319b0e7a69cf6c7efae395e2d0b0289a5b2d170` |
| `graphingest/resolution/README.md` | `4d0618166c4b8c9bca6b82d134767ad0aa35ab22ea3f9ddb74131f27a621e0fc` |
| `graphingest/resolution/contracts.go` | `cf72e2c4b63066780ecec8352932b72e37ba203471a2619fc37652e896394fd0` |
| `graphingest/resolution/resolver.go` | `bce4c70152f32d7f991dbe31004401eae731fa24049d73e179096e3ddb50aec9` |
| `graphingest/resolution/identity_test.go` | `613d1e3c11069a26fea94b2f3fbde87d879f34fe931e62d922794910490a2e10` |
| `graphingest/resolution/history/README.md` | `48fbe57d3ca4437fa3ec06ccf47bc4ce1fb2ae821c15c8e298ccfb04808198f4` |
| `graphingest/resolution/history/history.go` | `b5bcc7e81aac9aa465ae64e90249485712fd7a66887354f292eb19a57afb7887` |
| `graphingest/resolution/history/identity.go` | `879692430920615863abcdea235eca5a1aa5d1775207933458fc627490b2227c` |
| `graphingest/resolution/history/identity_test.go` | `8c6d6a38fcb18f7c1f1d381b86e806fef090bba5371323a55822bbd6181c98f7` |
