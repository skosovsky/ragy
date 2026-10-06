# Независимый аудит корректности task12

Дата: 5 октября 2026 года. Проверена текущая worktree; это аудит до исправления нижеперечисленных находок, а не финальное одобрение реализации. ТЗ прочитано непосредственно из `.cursor/docs/task12.md`; выводы другого аудитора и проценты матрицы не использовались. Production implementation и production tests аудитор не менял.

## Подтверждённые дефекты

### AUDIT-CORRECT-01 — P2: readonly BM25Snapshot отдаёт внутренние BYOT pointers

Код: `lexical/snapshot.go:67–73`, `lexical/snapshot.go:108–117`, `lexical/bm25.go:441–449`, `retrieval/resultset.go:58–60`.

Capture корректно вызывает предоставленный host metadata cloner, однако этот cloner не сохраняется для последующей выдачи. Snapshot Retrieve делегирует в BM25Index, который отдаёт retained Meta; ResultSet клонирует только document-owned slices, а pointer/map host metadata остаётся общим. Правильный, действительно deep host cloner на входе не решает проблему.

Публичное воспроизведение: `docs/task12/audits/repro_snapshot_alias.go`. Запуск из root:

```sh
GOMODCACHE=/tmp/ragy-implementation-mod-cache GOCACHE=/tmp/ragy-implementation-go-cache go run docs/task12/audits/repro_snapshot_alias.go
```

Фактический вывод:

```text
first 1 tenant a
after caller mutation 0 error <nil> first batch changed b
```

AAA: создать настоящий scoped pinned snapshot для tenant a с `*meta` и deep cloner; Retrieve; изменить Tenant через Documents первого результата; повторить тот же query/binding. Immutable readonly corpus и уже выданный batch изменились. Это подтверждённая потеря snapshot correctness, а не доказанная межtenant утечка: fingerprint запрещает заменить binding на tenant b. Возможные data races при параллельной мутации metadata здесь отдельно не воспроизводились.

Исправление: snapshot сохраняет cloner и перед каждой выдачей клонирует host metadata с gates до/после callbacks. Любой новый ownership contract должен последовательно распространяться на snapshot index/result boundaries; нельзя объявлять immutable corpus, а оставлять shared pointers. Regression с pointer/map/nested slice BYOT metadata и последующим Retrieve.

### AUDIT-CORRECT-02 — P2: BM25 принимает non-finite параметры и успешно выдаёт NaN scores

Код: `lexical/bm25.go:70–79`, `lexical/bm25.go:370–372`, `lexical/bm25.go:442–449`.

Сравнение `K1 <= 0`/`B <= 0` не отвергает NaN; положительная бесконечность также допускается. Scoring и delivery не валидируют вычисленные score. Публичное воспроизведение: `docs/task12/audits/repro_bm25_nonfinite.go`, аналогичный go run.

Каждый из трёх случаев K1=NaN, K1=+Inf, B=NaN даёт:

```text
construct <nil>
retrieve <nil> score NaN validation invalid argument: scored document requires finite value and semantics
```

Нарушен executable finite-score contract: сам backend успешно выдаёт документ, который его публичный ValidateDocument отвергает. Это касается baseline и snapshot, а не только optional recipe.

Исправление: явная finite/domain validation конфигурации до создания индекса, документированный диапазон параметров; отдельно устойчивое вычисление или fail-closed проверка computed score, поскольку очень большие finite параметры также могут переполнить промежуточную арифметику. Не clamp NaN к нулю/normalized score. Negative regressions должны подтверждать отсутствие successful invalid documents.

### AUDIT-CORRECT-03 — P2: persistent cleanup подтверждает forged original support inventory после удаления catalog

Код: `dense/persistent/cleanup_unix.go:113–124,157–166`, `tensor/persistent/cleanup_unix.go:113–124,157–166`.

`catalogRequestInventory` сравнивает только набор Artifact.Reference; original Supports игнорируются. Пока каталог существует, retireDirectory дополнительно сверяет полный catalog inventory. После настоящего supported cleanup каталог отсутствует, а InspectCleanup опирается на ослабленную сверку durable ledger и возвращает complete для подменённого original support. Durable provenance должен оставаться проверяемым именно после физической очистки.

Воспроизведено отдельно для dense и tensor через Go overlay, без записи production test files. Исходник: `docs/task12/audits/repro_cleanup_inventory_test.go.txt`; tensor вариант заменяет quoted target `dense` на `tensor`. Сценарий: реальный persistent Stage/Publish → tombstone → durable Cleaner.Begin/Attempt → Load ledger → Clone retired manifest → изменить только Artifact.Supports[0].Artifact → InspectCleanup.

Оба теста завершаются ожидаемым FAIL:

```text
forged original supports accepted after cleanup: state=complete
```

Команды выполнены:

```sh
GOMODCACHE=/tmp/ragy-implementation-mod-cache GOCACHE=/tmp/ragy-implementation-go-cache go test -overlay /tmp/ragy-audit-overlay.json ./dense/persistent -run TestAuditInspectCleanupRejectsForgedDurableSupports -count=1
GOMODCACHE=/tmp/ragy-implementation-mod-cache GOCACHE=/tmp/ragy-implementation-go-cache go test -overlay /tmp/ragy-audit-tensor-overlay.json ./tensor/persistent -run TestAuditInspectCleanupRejectsForgedDurableSupports -count=1
```

Overlay files `/tmp/ragy-audit-overlay.json` и `/tmp/ragy-audit-tensor-overlay.json` отображают virtual `audit_overlay_test.go` в соответствующий `/tmp` reproduction source. Тест зафиксировал false confirmation, не удаление чужого существующего payload; при intact catalog дополнительная проверка препятствует этой мутации. Поэтому severity P2, не заявленная data-loss уязвимость.

Исправление: в обоих persistent targets сравнивать exact artifact/support sets через SameTargetInventory с зарегистрированным durable retired manifest, независимо от наличия directory. Валидировать supplied manifests и payload identity последовательно. Regression для absent/retired directory, forged support и корректно reordered inventory.

## Проверенные существенные подозрения

- BUG-001: schema-aware integer normalization сохраняет соседние int64 выше 2^53; целевые root/filter/retrieval/lexical и persistent integer regressions прошли с race. Отдельные supplied Qdrant и PostgreSQL adapter integer/codec/scope tests прошли. Это injected transport verification; live внешние services не запускались.
- BUG-002: Index держит writer lock и строит отдельный staged corpus, заменяя published maps только после successful rebuild. Целевые failed rebuild, codec failure, reader visibility и concurrent upsert regressions проверены с race. Новая snapshot-alias находка независима от atomic rebuild.
- Source Reader проверяет весь metadata batch до Loader.Load, проверяет exact identities/counts, затем payload validation/cloning с freshness gates. Отрицательные original/revocation/scope tests прошли; latest substitution и partial payload при observed denied descriptor не подтвердились.
- Managed graph выполняет node/edge admission до traversal, не использует denied bridge; deliver клонирует BYOT Meta. Inventory/cleanup/revocation tests прошли. Graph conflicts могут описывать admitted corpus, но утечка запрещённых nodes в просмотренном пути не подтверждена.
- Atomic budget ledger делает subtraction-before-add для bounds, сериализует reserve/settle и conservatively удерживает unknown usage. Budget/concurrency/cancellation tests прошли; превышение из-за копирования Lease не подтвердилось.
- Scope failure markers исключают rescue и partial capability skipping при authority denial; root retrieval отрицательные scope tests прошли.
- Persistent Stage/Inspect exact inventory и corrupt catalog regressions прошли. False cleanup attestation выше остаётся отдельным непокрытым случаем.
- Evidence strict-schema validator независимо прогнан: 16 positive и 32 negative fixtures passed. Evidence scope/privacy/original source tests прошли; это validation shape и механической association, а не доказательство model quality.

## Actual проверки и пределы доказательства

`docs/task12/audits/correctness-targeted.txt`: успешный race запуск 11 пакетов: lexical, retrieval, filter, source, recipe/budget, lifecycle/integration, dense/persistent, tensor/persistent, graph/managed, recipe/graphsummary, evidence. Выбор тестов: `Integer|FailedRebuild|Concurrent|Cancel|Revocation|Forg|Inventory|Unknown|Deletion|Retire|Cleanup|Race|Scope|Corrupt|Original|Privacy`, count=1. Supplemental BUG-002 run: `Failed.*Rebuild|ReadersNeverObservePartialRebuild|ConcurrentUpsertSurvivesRebuildPublication`.

Аудит не утверждает отсутствие других ошибок и не является процентом полноты. Не выполнены live model quality experiments из-за отсутствующих credentials/model/tokenizer configuration; HTTP fixtures не считаются заменой. Не проверены live remote storage services, реальные hardware power-loss, все Go consumers произвольных BYOT callbacks, поведение custom adapters вне declared conformance profile. Существующие full make lint/test logs были исходным baseline, не доказательством исправления новых находок. После исправлений необходимы повторные affected regressions и оба независимых аудита окончательной worktree.

### Ownership уточнение для исправления AUDIT-CORRECT-01

ТЗ не требует reflection-based deep copy произвольного TMeta на каждом вызове Documents(). Достаточно, чтобы snapshot использовал host cloner на каждой выдаче и передавал результату отдельную от retained corpus metadata ownership. Внутри одного выданного batch host metadata может принадлежать consumer по явно описанному контракту; общая ResultSet документация должна уточнять, что defensive/immutable относится к ragy-owned fields, а не обещать deep immutability TMeta без clone port. Формулировка `docs/task12/migration.md:70` уже не заявляет deep ownership arbitrary BYOT domain metadata. Дополнительное clone-on-Documents API не является обязательным исправлением подтверждённой cross-request ошибки.

Supplemental BUG-002 проверка завершилась успешно: `ok github.com/skosovsky/ragy/lexical 1.201s`. Adapter runs: `ok github.com/skosovsky/ragy/adapters/qdrant 1.429s`, `ok github.com/skosovsky/ragy/adapters/pgvector 1.900s`.
