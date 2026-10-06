# Task 20 — ragy: исправить retrieval contracts, admission и release после ревью

**Статус:** задача на реализацию; исправления не выполнены.  
**Дата:** 2026-10-06. **Проверенный SHA:** `b63d5e19a52c7d4e621b1a85a3ce428a92acbcb7`.  
**Совместимость:** разрешён полный clean break, удаление legacy и реорганизация. Не оставлять второй старый engine/compatibility fallbacks ради сохранения прежней формы API.

## Итог

Подтверждены **11 дефектов: 2 P1 и 9 P2**. Ниже — отдельный реестр **61 решений** по архитектуре, naming, сложности и границам. Он не увеличивает счётчик ошибок. Восемь ролевых отчётов сохраняют полный разбор частных замечаний, включая те, которые объединены в одну задачу здесь.

Ревью выполнили восемь субагентов: retrieval/ranking/recipes; lifecycle/access; graph/graphingest; tensor/lexical/persistent storage; ingestion/source/layout/evidence; external storage/filter; providers/PDF; architecture/docs/observation/release. Основной ревьюер сверил исходники и независимо повторил воспроизведения всех F-пунктов. Source implementation, dependencies, Git history и remote не менялись.

Ragy сохраняет границы самостоятельной BYOT-библиотеки. Retrieval composition, source-bound graph facts, exact citations и lifecycle publication принадлежат её области. Не требуется новая библиотека для исправления обнаруженных дефектов. Journals/unknown outcomes, pins, exact inventory, повторные read gates и managed cleanup не следует удалять только ради сокращения кода: они обеспечивают текущие гарантии.

## Проверки и ограничения

| Проверка | Результат |
|---|---|
| Штатный `make lint` | PASS, 14 modules, 0 issues; локальный golangci-lint2.14.0 выдаёт только deprecation warning для exhaustruct |
| Штатный `make test` | PASS, exit0: `go test -v -race ./...` во всех 14 modules, затем build/test planner и resilience examples; cache-hit markers в журнале отсутствуют |
| Targeted defect reproductions | Независимо повторены все F-пункты. Recipe regression assertions намеренно FAIL; остальные diagnostic PASS/программы демонстрируют текущий дефект, а не исправление |
| Source / Git | Исходный HEAD сохранён, implementation/go.mod/go.sum/go.work не изменялись; `git diff --check` PASS |

Фактическая среда: Go1.27.1 darwin/arm64; go.mod/go.work указывают1.26.1, CI lint pin2.11.3. Этот запуск не подтверждает minimum-Go/Linux/CI-toolchain parity. Go/lint cache в `/tmp`; example builds показали нефатальные sandbox warnings при записи module stat cache, сами build targets вернули0.

**Не выполнены в этом ревью:** live provider API, live remote storage enforcement/SQL execution, opt-in actual PDF parser tests, новый live quality/reference-budget benchmark, аппаратные power-loss проверки, отдельный GOWORK=off acceptance и performance/fuzz campaigns. Штатные opt-in tests дали SKIP; это не PASS real-service профиля. PostgreSQL F08 проверен как core+SQL translation и вывод из первичных SQL contracts. Файловые adapters проверялись штатными локальными/process-restart fixtures. Исторические task19 logs учитывались как датированный контекст, не как новые результаты.

## Подтверждённые дефекты

### F01 — P1: deadline превращает protection failure в успешный partial result

**Код:** [recipe/run.go:304](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/recipe/run.go:304>), [recipe/run.go:214](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/recipe/run.go:214>), [recipe/run.go:416](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/recipe/run.go:416>).

**Проблема и доказательство.** Public RunOwn: original retrieval получает d1; Planner возвращает access.NonSkippable(ErrUnavailable), одновременно injected clock пересекает local deadline. settle стирает callErr, stop объявляет bounded partial, результат содержит d1 с nil error. То же воспроизведено для ErrProtocol. Binding при этом ещё валиден. Это обход явного fail-closed контракта, а не доказанное чтение чужого tenant.

**Исправление.** Сохранить callback и settlement causes, задать приоритет protection/protocol/usage overrun над локальным deadline. Простого errors.Join недостаточно: boundedStop сейчас допускает protection, если ошибка также matches DeadlineExceeded. Учёт reservation ровно один раз, без retry.

**AAA.** Arrange: prior evidence, callback protection/protocol/ordinary error и пересечение injected deadline; Act: Run/RunOwn/RunObserved; Assert: protection подавляет весь payload/journal, cause сохранён; допустимый failed journal только по контракту RunObserved. Pure local deadline сохраняет штатный partial. Проверить backend/assessor/encoder boundaries, mixed protection+deadline и usage overrun.

### F02 — P1: release публикует посторонние файлы и теги

**Код:** [scripts/release.sh:31](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/scripts/release.sh:31>), [scripts/release.sh:93](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/scripts/release.sh:93>), [scripts/release.sh:114](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/scripts/release.sh:114>).

**Проблема и доказательство.** Tracked-only clean check допускает untracked файл; git add . включает его в release commit, push --tags отправляет все локальные tags. Исходный script в local bare fixture опубликовал private-untracked.txt и scratch-local вместе с v0.0.1. Данные synthetic, реальная история/remote ragy не менялись.

**Исправление.** Изолированный checkout либо точный allowlist файлов/refs. Stage только ожидаемые module manifests, push только перечисленные release tags. Проверить dirty/staged/untracked policy до mutation; не удалять чужие файлы.

**AAA.** Arrange: посторонний файл/tag и чистые tracked sources; Act: release fixture; Assert: опубликованы только разрешённые source/files/refs. Dirty index и существующий module tag обрабатываются явно.

### F03 — P2: ошибка release оставляет detached HEAD и занятую версию

**Код:** [scripts/release.sh:78](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/scripts/release.sh:78>), [scripts/release.sh:103](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/scripts/release.sh:103>), [scripts/release.sh:114](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/scripts/release.sh:114>), [scripts/release.sh:118](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/scripts/release.sh:118>).

**Проблема и доказательство.** Rejecting local pre-receive hook: exit1, branch пуст, локальный v0.0.1 остался, remote пуст. Повторный расчёт версии учитывает failed tag. Partial publication также требует протокола, но текущий repro доказывает полный rejection.

**Исправление.** Изолированный release worktree либо восстановление caller checkout на каждом exit; явный manifest созданных/опубликованных refs и повтор того же candidate. Atomic push где поддерживается; none/partial/unknown publication различать, не удалять tags вслепую.

**AAA.** Arrange: отказы до tag/после tag/при push; Act: release/retry; Assert: исходный checkout сохранён, состояние публикации известно или явно unknown, версия не увеличивается автоматически, unrelated refs целы.

### F04 — P2: extraction не соблюдает более ранний deadline общего ledger

**Код:** [graphingest/extraction/extraction.go:54](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/graphingest/extraction/extraction.go:54>), [graphingest/extraction/extraction.go:160](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/graphingest/extraction/extraction.go:160>), [recipe/budget/budget.go:87](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/recipe/budget/budget.go:87>).

**Проблема и доказательство.** Ledger истекает через5s, local extraction Duration60s. Model видит context deadline на55s позже attempt; сдвиг shared clock на6s внутри callback и валидный output дают entities2,nil после истечения ledger. README обещает suppression по более раннему attempt deadline. Settle разрешает поздний учёт и не служит admission gate.

**Исправление.** Использовать min(parent,ledger,local) context и проверять deadline по заявленному injected-clock контракту на границах callbacks/delivery. Один wall-clock timer не закрывает deterministic clock leap. Usage учитывать и при expiry, без повторного dispatch.

**AAA.** Arrange: разные parent/attempt/local deadlines и управляемые clock/barriers; Act: completion/clone/project после раннего deadline; Assert: zero payload + DeadlineExceeded, settled usage, один вызов. До deadline — успех.

### F05 — P2: resolver объединяет разные identity keys из-за JSON UTF-8 repair

**Код:** [graphingest/resolution/resolver.go:147](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/graphingest/resolution/resolver.go:147>), [graphingest/resolution/resolver.go:197](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/graphingest/resolution/resolver.go:197>), [graphingest/resolution/resolver.go:397](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/graphingest/resolution/resolver.go:397>).

**Проблема и доказательство.** Canonical ID хеширует JSON tuple namespace/key. Невалидные строки ff и fe нормализуются encoding/json в U+FFFD. Public Resolver возвращает одну entity,nil и одинаковый ID для двух разных host keys с одинаковым display name. Это коллизия кодирования, не SHA256. Entity-path выполнен; relation-path аналогичен по коду.

**Исправление.** Явно выбрать string identity domain: reject invalid UTF-8 до grouping/hashing либо byte-exact length-framed encoding, если байтовые keys намеренны. Согласовать namespace/name/policy/history и миграцию hash framing; не нормализовать keys молча.

**AAA.** Arrange: invalid key/namespace/name/relation key, валидный U+FFFD, различные/одинаковые valid tuples; Act: resolve; Assert: при UTF-8-only контракте invalid rejected без result; если намеренно разрешены byte keys, ff/fe/U+FFFD дают разные IDs. Distinct valid identities не сливаются, equal valid decisions сливаются по прежнему контракту.

### F06 — P2: layout допускает один exact source reference для разных текстов

**Код:** [layout/layout.go:100](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/layout/layout.go:100>), [layout/project.go:71](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/layout/project.go:71>), [source/reference.go:13](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/source/reference.go:13>).

**Проблема и доказательство.** Две упорядоченные pages0/1 с одним retained Reference и текстами AAA/BBB проходят Validate и Project; output2, identicalIDs=true. Text locator — exact reference + byte span, physical page в нём нет. Два разных оригинала нельзя разрешить одним source address; dedup/upsert теряет одну страницу. Это публичный layout input, не доказанный дефект PDF engine.

**Исправление.** Проверять уникальность exact page references до projection/callbacks. Не лечить добавлением page number только в hash при всё ещё неоднозначном Loader reference. Для cell/image sharing сначала определить допустимые случаи.

**AAA.** Arrange: две pages с equal reference и разным текстом одинаковой/разной длины; Act: Validate/Project; Assert: отказ, ноль projected payload/ImageText calls. Distinct references дают независимо разрешимые цитаты.

### F07 — P2: BM25Snapshot вызывает следующий codec после отмены чтения

**Код:** [lexical/bm25.go:398](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/lexical/bm25.go:398>), [lexical/snapshot.go:20](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/lexical/snapshot.go:20>).

**Проблема и доказательство.** filterScoredDocs не получает context/Binding и проходит все MatchDocument callbacks. Codec первого candidate отменяет context; callback второго всё равно вызывается. Probe: calls2, protected cancellation, results0. Финальный result скрыт корректно, но обещанный freshness gate каждого callback нарушен.

**Исправление.** Проверять read/context до и после каждого metadata callback. Protection error подавляет partial candidates и не теряется за ordinary codec error. Не добавлять authority cache/retry и не менять borrowed ownership raw BM25.

**AAA.** Arrange: два candidates, cancel/revoke внутри первого Encode; Act: snapshot/managed retrieve, cache hit/miss; Assert: только один callback, no later clone/delivery, empty protected error. Контроль без cancellation сохраняет ranking.

### F08 — P2: pgvector отрицание расходится с portable filter на absent fields

**Код:** [adapters/pgvector/store.go:411](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/adapters/pgvector/store.go:411>), [adapters/pgvector/store.go:451](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/adapters/pgvector/store.go:451>), [filter/match.go:52](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/filter/match.go:52>), [filter/match.go:186](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/filter/match.go:186>).

**Проблема и доказательство.** Отсутствующие schema fields допустимы. Core matcher для category!=x, NOT(category=x), NOT(category IN[x,y]) возвращает true; SQL использует <>/NOT над attributes->>category и даёт NULL/unknown, поэтому WHERE исключает строку. Результаты retrieval/DeleteByFilter расходятся. Probe проверил core и фактический SQL; live PostgreSQL не запускался, SQL вывод основан на official NULL/JSON semantics. Mandatory scope Eq/In/And — другая область; scope bypass не заявляется.

**Исправление.** Специфицировать two-valued missing-field truth table и нормализовать leaf predicates до NOT/AND/OR. Один COALESCE внешнего WHERE недостаточен. Для != возможны negated normalized equality/IS DISTINCT FROM с корректным scalar domain.

**AAA.** Arrange: missing/equal/unequal values всех supported scalar kinds; Act: core matcher и query/delete PG; Assert: одинаковые IDs для вложенных NOT/AND/OR и сравнений. Сохранить exact int64 и injection tests; real DB parity — отдельный acceptance profile.

### F09 — P2: Neo4j Retrieve выдаёт payload после cancellation

**Код:** [adapters/neo4j/neo4j.go:64](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/adapters/neo4j/neo4j.go:64>), [adapters/neo4j/neo4j.go:93](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/adapters/neo4j/neo4j.go:93>), [adapters/neo4j/neo4j.go:133](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/adapters/neo4j/neo4j.go:133>).

**Проблема и доказательство.** Runner отменяет supplied context перед возвратом валидного snapshot. Direct Retrieve выдаёт nonempty,nil, хотя Read.Check уже отвергает context. Нет final DeliverRead. Scoped/pinned profiles здесь отклоняются заранее: это delivery cancellation inconsistency, не утечка по tenant revocation.

**Исправление.** Применить общий public Retrieve/private retrieve/DeliverRead pattern ко всем success/empty/partial paths. Сохранить ordinary partial errors, не повторять traversal.

**AAA.** Arrange: cancellation на выходе Runner и projection error после cancellation; Act: direct Retrieve; Assert: empty protected result, errors.Is(context.Canceled), один traversal. Обычные success/partial scenarios неизменны.

### F10 — P2: structured adapter теряет deadline при чтении response body

**Код:** [adapters/openai/structured/client.go:227](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/adapters/openai/structured/client.go:227>).

**Проблема и доказательство.** Context-aware response body после headers возвращает requestCtx.Err(). Клиент выходит с ErrProtocol до context gate. Public deterministic probe: IsDeadlineExceeded=false, IsProtocol=true. Таймаут классифицируется по стадии HTTP exchange, что ломает accounting/retry policy потребителя.

**Исправление.** Проверять cancellation/deadline перед sanitized protocol classification body/transport ошибок; output пустой, usage неизвестен, body закрыт. Не читать truncated JSON ради выдуманного usage и не добавлять retry.

**AAA.** Arrange: valid config, body ждёт per-attempt timeout/parent cancel; Act: Call; Assert: правильный errors.Is, один dispatch, zero output/unknown usage, body closed. Ordinary I/O error остаётся sanitized protocol/согласованным transport class.

### F11 — P2: structured BaseURL с bare query меняет pathname запроса

**Код:** [adapters/openai/structured/client.go:74](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ragy/adapters/openai/structured/client.go:74>).

**Проблема и доказательство.** https://provider.example/v1? принимается: RawQuery пуст, ForceQuery не проверен. Конкатенация endpoint даёт path=/v1, query=/chat/completions. Shared providerhttp уже отвергает ForceQuery. Это config admission bug, не cross-host credential leak.

**Исправление.** Единая политика parsed URL/ForceQuery и построения endpoint. Reject query/fragment/credentials/opaque URLs до dispatch, сохранять разрешённый base path. Не нужен новый публичный transport framework.

**AAA.** Arrange: bare ?, query, fragment, credentials и валидные bases с/без slash; Act: New/Call; Assert: invalid typed rejection до network, valid pathname ровно /v1/chat/completions.

## Отдельные решения по архитектуре, naming и странностям

Это задачи на осознанный disposition: изменить, оставить с обоснованием либо уточнить контракт. Не все требуют нового кода. Подробные code references/отвергнутые подозрения — в восьми ролевых отчётах из пакета evidence; неподтверждённые native security/долговечность claims не переносить в F-список.

| ID | Область | Решение / критерий |
|---|---|---|
| D01 | Границы retrieval composition | Bounded recipes/fallback/rescue допустимы в ragy. General workflow — flowy, agent loop/UI/Ask–Plan–Action — harness; не создавать новую библиотеку на каждую retrieval стратегию. |
| D02 | Два composition engines и ranking façade | Свести result-only и execution-aware внутренние engines, если это уменьшит drift; inventory consumers перед удалением ranking compatibility aliases. Без второго legacy engine ради совместимости. |
| D03 | Merger fallback | Ошибка custom fusion сейчас ведёт к score merge с error. Это documented degradation, но лучше explicit policy/original observed sets; не сравнивать разные score scales автоматически. |
| D04 | Zero ExecMeta | reflect.DeepEqual(zero) означает omitted и мешает reset в корректный zero. При clean break ввести presence/update contract или всегда уважать returned value, без reflective угадывания. |
| D05 | Partial result/error authority | Согласовать separate ResultSet и nested PartialFailureError.Result, сохранить outer causes при errors.Join/As. Не объявлять source-level подозрение отдельным доказанным багом. |
| D06 | Nil callbacks и resolver defaults | Унифицировать constructor validation; custom ResultSet не обязан быть built-in для сохранения resolver. Explicit resolver capability/argument лучше молчаливого default. |
| D07 | Threshold stage | Различить pre-postprocessor и terminal threshold, reranked/native score. Raw BM25/персистентные backends должны явно сообщать поддержку options; не выдавать topK за work cap. |
| D08 | MergeKey и evidence equality | Winner merge может объединять supports разных texts, RRF строже. Определить semantic grouping vs identical evidence; GroupBy остаётся явной сборкой content. |
| D09 | Config/metadata ownership | Документировать freeze/clone для Artifact pointer, SynonymMap, Run config и BYOT metadata. Host callbacks остаются cooperative/concurrency-safe; не blanket JSON deep-copy всего. |
| D10 | Budget semantics | Calls не refunded, unknown usage reserved, ledger scope может быть shared. Развести known tokens и known cost при необходимости; это не организация billing/quota и не metry ledger. |
| D11 | Model-free attestation | Это declaration host, не sandbox. UnsupportedQueryEncoderBridge не должен хранить неиспользуемый Embedder; explicit unsupported, без скрытого model call. |
| D12 | Counts против bytes/work | MaxDocuments, candidate count, cache entries и retained snapshots не гарантируют bounded RSS/CPU. Описать конкретные units/limits и trusted bounded callbacks, не хардкодить универсальный cap. |
| D13 | Cache/copy cost | MemoryCache O(capacity) eviction и repeated result slice copies измерять; byte budget/singleflight только по потребности. Никакого refresh daemon/distributed cache по умолчанию. |
| D14 | Fallback vs Rescue | Сохранить отличие empty-success от empty-error. Nil conditional predicate и rank degradation должны быть явной configuration policy, не неочевидным разрешением dispatch. |
| D15 | Publication terminology | Развести source→manifest pointer, immutable read capture и durable metadata pin в глоссарии/названиях. Capture не lease, Acquire pin защищает metadata, не source/target payload availability. |
| D16 | Capture namespace check | Удалить tautology snapshot.Namespace!=namespace, сравнить requested namespace после Store.Load. Custom misrouted Store repro подтверждает hardening gap; native filestore cross-tenant defect не найден, F-счётчик не увеличивать. |
| D17 | Lifecycle CAS/reuse | Namespace-wide CAS консервативен; positive CheckReuse — observation point, не permission/lease. Уточнить comment: только positive confirmation повторно проверяет generation. Retry/reconcile выбирает host. |
| D18 | Lifecycle unknown/cleanup | Read-only inspect может вернуть OutcomeUnknown без разрушительного вызова. Cleaner делает один шаг и выдаёт schedule/backoff; Complete не означает forensic/source erasure. Дать operation-specific recovery table. |
| D19 | Постоянные reservations | Retired skeletons, released IDs, digests и receipts нужны против ABA. Не TTL-delete и не auto-evict при capacity; миграция/retention policy принадлежат host. |
| D20 | Lifecycle boundedness/clone | Finite port calls не равны constant CPU/bytes: full snapshot и scans растут с history. Оптимизация через temporary indexes/explicit deep clone — после замеров и differential validation. |
| D21 | Plan equality/CAS authority | Порядок artifacts/targets может быть частью idempotency. Не менять canonicalization существующих reservations молча. Raw CompareSwap — trusted low-level interface, не authenticated command API. |
| D22 | Inventory fences | Сохранить sorted locks, exact-once synchronous observer callback и exact inventory. Нет принудительного прерывания arbitrary callback через unbounded goroutines. |
| D23 | Graph ingestion scope | Extraction/resolution/materialization/history — retrieval-domain facts/provenance. Ontology, aliases, semantic truth, model transport и lifecycle driving — host. |
| D24 | Relation closure/naming | Source-local endpoint support обязателен по контракту; пример cross-document relation. Name — canonical, не mention label; рассмотреть CanonicalName/MentionKind вместо перегруженного Kind. |
| D25 | Graph generic kinds/history | Stable comparable enum/value и faithful bounded JSON callbacks — host contract. History capture не пересчитывает ontology и не доказывает policy truth; maxSupports также считает другие populations. |
| D26 | Graph history bytes/platform | Marshal cap — accepted wire, не peak allocation. FileStore darwin/linux и host-prepared durable root описать явно; no automatic retention/retry после unknown append. |
| D27 | Graph scaling | Support unions/variant grouping и summary admission могут быть quadratic. Сначала benchmark upper bounds; map[Locator] со stable order допустим, новая indexing framework не нужна. |
| D28 | Extraction full-input budget | MaxInputBytes сейчас source-text only, не весь model envelope. Назвать точно или отдельный envelope cap; CountInputTokens должен учитывать реальный provider input. |
| D29 | Summary dependencies | Selected citations не равны всем model derivation inputs. Уточнить revocation policy для unselected input; при необходимости separate dependency inventory. Structural association не semantic truth. |
| D30 | Summary delivery/coverage | Resolve проверяет текущий доступ; возвращённый MappedText уже owned и не отзывается задним числом. Coverage — declared set coverage, не качество prose, MissingCoverage не запускает hidden repair. |
| D31 | Graph recipe heuristics | Seed-only graph как Insufficient и fail-zero при node/edge limit — explicit policies. Volatile managed graph не восстанавливается из одного manifest; не fallback на raw/current graph. |
| D32 | Error precedence в graph recipes | Graphsummary/graphexpand/extraction также могут скрыть callback/settle causes за gate. Расширить F01 acceptance на эти пути; отдельный payload-release defect там не доказан. |
| D33 | Materialization/ergonomics | Build validation не обязательно возвращает normalized metadata. Определить normalization stage, дать минимальную композицию generics/callbacks; не добавлять DI builder framework. |
| D34 | Chunk overlap/separators | Overlap применяется лишь fixed-splitting fallback; описать best-effort либо реализовать boundary overlap. Custom non-whitespace delimiters могут исчезать — explicit keep/drop policy и coverage properties. |
| D35 | Markdown/sentence heuristics | Hash-line splitter не полноценный Markdown, punctuation sentence splitter не NLP. Назвать поддерживаемую грамматику, учесть code fences/Setext либо дать BYO ports, без скрытых моделей. |
| D36 | Segment/chunk admission | Определить allowed gaps, Total0, unmapped UTF-8 и Index ordering. Проверять trusted callback contract пропорционально; не выдумывать source coordinates без оригинала. |
| D37 | Projection partial/identity | Унифицировать all-or-nothing vs partial projection. Descriptor/chunk identity precedence, StorageID для всех chunks и URI MergeKey требуют ясной authority; не global ID service. |
| D38 | Source ownership/RawStore | Reader Catalog→admitted Loader и no-latest гарантии сохранить. RawStore — explicit unscoped administration. Project получает уже authorized document, Binding сам не доказывает его принадлежность scope. |
| D39 | Mapping strictness/attestation | Persisted mapping JSON менее строг locator/evidence wire; решить duplicate/case/Unicode policy. Structural mapping не доказательство exact source text — нужен authorized resolve, не «защита» hash-ом. |
| D40 | Source/layout cost/supports | Повторное UTF-8 scanning каждого span может быть O(words×text). Validate once на trusted snapshot boundary. Slice сохраняет broad support ancestry; whole media Resolve не crop API. |
| D41 | OCR/image projection | Partial coverage и original-vs-derived distinctions сохранить. ImageText callback явно заменяет OCR даже при empty output; описать или explicit host policy, не hidden fallback. |
| D42 | Multimodal URL | Transport-neutral URL не downloader/SSRF policy service. Providers валидируют supported schemes/MIME, host — network allowlist; trim/normalization и inactive fields согласовать. |
| D43 | Evidence scope/bounds | Membership tuple без Transformation может означать original source, не index artifact. Зафиксировать семантику; rank int→float64 ограничить meaningful ordinal domain; output wire cap не входной CPU bound. |
| D44 | Tensor score/work | MaxSim может быть negative/>1, Dot не требует normalized input. Candidate-bound != token×dimension bound; ctx checks между rows/candidates и limits нужны по профилю, не implicit clamping. |
| D45 | Candidate catalog semantics | Exact-within-candidates не ANN/exhaustive. Budget считает docs, projection dedup считает refs; catalog scan шире payload budget. MaxRecords не должен скрывать несколько независимых units. |
| D46 | Lexical cache/config | Snapshot count не byte/transient-builder cap; generation-wide invalidation безопасна. B=0/K1=0 как default мешает математическим zero значениям — optional config при clean break. Borrowed raw metadata явно ограничивает concurrency. |
| D47 | Повторные admission passes | Некоторые clone/build/check нужны для ownership/freshness; объединять только с fence tests. CPU scoring/sorting BM25 требует cancellation checkpoints при больших corpora. |
| D48 | Local file profile | fsync files/root не доказывает power-loss mkdir ancestor durability. Явный pre-provisioned root либо synced init при нужном контракте; не заявлять hardware crash verification по process-crash tests. |
| D49 | Filesystem boundaries | Nonblocking flock конфликт возвращает host; describe handle/OS semantics, no networkFS fallback. Trusted root не hostile filesystem sandbox. Explicit Retired check полезен, native retired-payload bypass не подтверждён. |
| D50 | Dense/tensor duplication | Узкий internal helper для inventory/cleanup может сократить drift; не публичный универсальный storage engine, скрывающий gates и type-specific validation. |
| D51 | Storage codec/filter contract | PG custom codec получает json.Number, ES/Qdrant normalized scalar: согласовать input types/empty attrs. Portable missing/null/wrong-kind truth table и constructor sentinel errors нужны независимо от SQL. |
| D52 | ES tokenizer/config | SynonymMap следует clone/freeze; tokenize один раз и передавать результат. Stateful tokenizer не должен менять preflight и dispatched query; не новый tokenizer framework. |
| D53 | PG identifiers/batch/order | Unquoted safe regex защищает injection, но uppercase/keywords требуют policy/quoting. Upsert batch bound host-owned без hidden chunked partial commits. Equal-distance tie order определить с учётом ANN planning. |
| D54 | Bridge capability honesty | Neo4j — Runner bridge, не native driver; generic Filters mapping/rejection и projection validation явны. Unknown affected counts не заменять длиной input. Local wire tests не certification live scope enforcement. |
| D55 | Provider transports | Shared URL/error contract tests предотвращают F10/F11 drift; structured usage/schema semantics не обязательно объединять с embedding client. Consistent cancellation/I/O/protocol taxonomy и credential preflight. |
| D56 | Provider JSON/schema | Unknown fields evolution отдельно от duplicate/case/surrogate policy. Structured choice Index missing/null требует explicit admission; executable domain schema host-owned, не новый universal validator. |
| D57 | Space/config naming | Model/Space.Model дублируются; ModelRevision/Configuration — host attestation, не remote fact. MaxOutputTokens для tensor rows лучше MaxVectorRows. Defaults/explicit limits и Cohere query+docs accounting описать таблицей. |
| D58 | Usage/error materialization | Известный usage не должен теряться только из-за rejected vectors; где неизвестен, не оценивать как known. Cohere retained input/prefix с error не считать reranked success; billing/prices остаются host. |
| D59 | PDF limits/errors | Catch-all invalid_pdf скрывает engine bugs; sanitized internal class рассмотреть отдельно. Output limits не гарантируют RSS/CPU. Документировать tested Python deps/fingerprint/real parser opt-in, host process isolation вне core. |
| D60 | Observation boundaries | Сохранить bounded enums, known/unknown, payload-free events и synchronous serialized callbacks. Pair/drop/event units описать; custom error.Is/Unwrap не sandbox. Adapter to metry optional, без exporter daemon. |
| D61 | Docs/CI/release surface | Current guides вынести из task/.cursor chronology; убрать blanket word blacklist в пользу symbol/snippet tests. Standalone GOWORK=off CI, fresh acceptance и publishable-module manifest отдельно от example modules. |

## Документация, которую нужно привести к текущему API

1. Дать короткий runnable local BYOT пример: установка, Go requirement, schema/meta, BM25, explicit Read, обработка partial/protection/errors; текущий Quick start — параметризованный helper, а не полная сборка. Отформатировать snippets.
2. Стабильные integration, lifecycle, source/layout и ownership contracts разместить в публичных topic paths. Исторические docs/task13–19 и .cursor acceptance оставить датированным evidence, не основным руководством API.
3. Исправить устаревший последний абзац graphingest/resolution/README: extractor, summary recipes и materialization/history integration уже существуют; описывать текущие границы package.
4. Таблицы bounds/units/defaults: calls vs tokens vs rows vs bytes vs candidate counts, wire cap vs peak allocation, exact-within-candidates, cache snapshots vs memory, source-text vs full provider envelope.
5. Error/recovery matrix: protection всегда suppressive; bounded local deadline отдельно от callback failure; unknown dispatch/cleanup отдельно от read-only unknown; raw administration отдельно от scoped read.
6. Общий callback/ownership contract: pure/stable/concurrent-safe ports, borrowed vs owned metadata, cancellation gates, отсутствие preemption arbitrary callbacks. Выделить retained source authority и distinction citations/derivation dependencies.
7. Capability matrix сохранить честной: wire/local/process-crash/real-service/quality — разные профили. Historical checklist100% и scripted benchmark не доказывают production readiness. Codex CLI external advisory runner остаётся допустимым; новый live reference-budget прогон не является условием закрытия этого ревью.
8. Release runbook: reviewed source commit, publishable modules vs examples, exact files/tags, platform, no/partial/unknown publication, recovery и clean consumer install. До v2 определить semantic import version; текущую v0 реализацию не объявлять сломанным v2.
9. Владелец выбирает LICENSE/versioning/contribution/security-reporting policy для публичного сопровождения. Не добавлять выдуманную лицензию от имени автора; отсутствие шаблонов не runtime blocker.

## Границы библиотеки

| В ragy | У host / соседних библиотек |
|---|---|
| Typed retrieval, filter IR, scores/fusion, bounded recipes | Agent goal/loop, tool execution, workflow scheduler, Ask/Plan/Action/UI |
| Source locators, exact/derived mapping, chunking/layout normalization | Blob storage/retention service, authorization policy, crop/render UI |
| Graph extraction protocol, resolution/materialization/supports | Ontology/alias truth, model selection/prompts, semantic grading |
| Generation/publication/pins, one-step cleanup/reconcile | Poll/backoff scheduling, business retention, distributed IAM authority |
| Attempt reservations, actual observed usage, deadlines | Global quota/prices/billing, metry collection, evaly experiments |
| HTTP/wire/provider/storage capability adapters | Native driver deployment, remote schema migration, retries, credentials store |

## Порядок реализации и Definition of Done

1. **Spec-First / Contract-First:** записать error precedence и clock composition; filter missing-field truth table; exact identity/Unicode domain; callback/delivery gates; release scope/recovery. API rename/reorganization принимать по уменьшению ответственности и числа независимых implementation paths.
2. Закрыть P1 F01/F02. Применить error-precedence acceptance к graph recipe boundaries, не считать `errors.Join` достаточным исправлением без теста joined protection+deadline.
3. Исправить F03–F11 с AAA regression tests: probes из evidence — исходные доказательства, diagnostic PASS нельзя сохранить как тест правильного поведения. Проверять zero payload/zero callbacks/no mutation на rejected inputs и exact call/settlement counts.
4. Для фильтров добавить portable conformance и real PG parity на isolated test corpus. Для provider ошибки body/URL проверять fake Doer без платных API; для layout/resolution — source identity properties. Не «лечить» failed boundary fallback-ом.
5. Разобрать каждый D-пункт и частные замечания ролевых отчётов: retain/change/contract decision с кратким обоснованием. Удалять legacy, stale facades и дублирующие engines после usage inventory, сохраняя pinned-publication/protection/unknown-outcome guarantees.
6. Синхронно обновить current GoDoc/README/examples и contracts. Сохранить исходные исторические результаты, добавив remediation reference вместо переписывания прошлой приёмки.
7. Повторить all-module lint/race/examples и meaningful changed-scope tests. Проверять release candidate GOWORK=off/clean consumer; live backend/parser acceptance запускать для изменённых соответствующих adapters и явно фиксировать skips. Не требовать произвольный live LLM benchmark для code-review fixes.
8. Для оптимизаций bounds/clone/union/cache — измерения до/после на relevant profiles, без подгонки порогов под текущий output. Concurrency/protection properties должны сохраняться.

## Доказательства

[Материалы, команды и полный реестр ролевых замечаний](</Users/skosovsky/Library/Mobile Documents/com~apple~CloudDocs/Загрузки/Projects/ai/ai-libs/reviews/ragy-2026-10-06/README.md>) содержат восемь отчётов, постоянные copies probes/overlays, baseline logs и независимые reproductions. Мастер-задача определяет окончательную классификацию; raw reports сохраняют исходные формулировки/временные paths. Все code line numbers относятся к review SHA.

SQL-вывод F08 опирается на первичные документы PostgreSQL: [сравнения и NULL](https://www.postgresql.org/docs/current/functions-comparison.html), [JSON missing-field operators](https://www.postgresql.org/docs/current/functions-json.html). Это проверка семантики SQL, не заявление о выполненном live PostgreSQL тесте.
