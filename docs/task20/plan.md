# Task20 — последовательная remediation

Старт: 2026-10-06, baseline `b63d5e19a52c7d4e621b1a85a3ce428a92acbcb7`.

Источник: [master review](review-baseline.md); восемь неизменённых [ролевых отчётов](reviews/). Оригинал master находится в ignored `.cursor/tasks/task20-ragy-review-remediation.md`. Snapshot сохраняет классификацию и исходные доказательства, а не объявляет исправления выполненными. SHA256 источников — [sources.json](sources.json).

## Исполняемый порядок

[backlog.json](backlog.json) содержит все критерии и состояния. [traceability.json](traceability.json) содержит отдельные строки для 11 F, 61 D, 161 частного design-замечания, 12 ролевых defect/hardening entries, DOC1–9 и DOD1–8. Каждый F/D/raw decision имеет primary task; shared contract фиксируется T01, final audit T22 проверяет весь реестр. Работа идёт строго T00 → T22, одновременно реализуется только одна задача. Связанные решения из другой области принимаются в её задаче, не теряются за общими формулировками.

## Общая приёмка каждой задачи

Перед реализацией уточнить контракт, acceptance cases и scope в task record; решения внутри разрешённого clean break доступны исполнителю, изменение согласованной спецификации требует вопроса владельцу. Все assigned rows получают `fix`/`change`/`retain`/`contract` с rationale и evidence. `retain` не закрывает F-дефект; для D/raw findings требуется обоснованное решение, не обязательно новый код. Неопровергнутые positive guarantees и rejected suspicions сохраняют исходную классификацию.

Для каждого task criterion и каждого assigned master/raw finding полнота считается отдельно: выполненные / все применимые критерии × 100; gate требует 100% в обоих реестрах. Нельзя менять знаменатель, удалять неудобные требования или отмечать заблокированное как выполненное. SKIP/N/A не заменяют обязательный PASS. Условный профиль (например, parser при изменении PDF) исключается только по документированному scope decision и проверенному diff, с независимой приёмкой.

После реализации создать двух отдельных read-only субагентов, не участвовавших в реализации и не получающих отчёт другого до собственного вердикта:

- Completeness: сверяет criteria, primary F/D/raw/DOC/DOD, исходные отчёты и evidence; выдаёт таблицу выполнено/не выполнено/заблокировано и оба процента.
- Correctness: независимо проверяет diff/контракты/error precedence/protection/callbacks/ownership/unknown outcomes, запускает необходимые проверки; выдаёт severity/воспроизведение или PASS без обнаруженных ошибок.

Оба отчёта фиксируют baseline HEAD, exact content digest проверяемых implementation/spec files, команды и результаты в `acceptance/Txx-*.md`. При замечаниях исправить текущую задачу и повторить обе приёмки актуального diff; пока не принята, следующую не начинать. Допускается повторный запуск тех же независимых приёмщиков, не допускается их участие в реализации. Gate: полнота 100%/100%, correctness PASS, нет открытых замечаний или обязательных skipped проверок.

После gate обновить журнал/статус, stage только файлы задачи и commit с коротким английским сообщением в стиле истории. `!` использовать только при фактическом breaking change; предложенные сообщения уточняются по final diff. SHA фиксируется в следующем journal update (коммит не может включать свой собственный hash); для последнего коммита SHA находится через `git log` по task record и сообщается в финале. Acceptance digest не включает только bookkeeping fields текущей приёмки, чтобы избежать self-reference; новые substantive edits требуют перепроверки.

## Сохраняемые гарантии и границы

BYOT; zero payload при protection; exact citations/source authority/no latest fallback; publication pins/inventory fences; journals/unknown outcomes; permanent ABA reservations; cooperative pure/stable/concurrent callbacks; no hidden retry/cleanup scheduling, модели или arbitrary-callback preemption. Рефакторинг удаляет заменённый legacy после consumer inventory. Ragy остаётся retrieval-библиотекой: policy/ontology/blob retention/pricing/UI/general workflow принадлежат host.

Regression tests используют AAA и утверждают исправленное поведение; diagnostic PASS исходных probes не считается regression PASS. Измерения до/после обязательны для фактически выполненных optimizations bounds/clone/union/cache/metric; retained implementation требует обоснования и не объявляется ускоренной. Исторические evidence/acceptance не переписываются.

Обязательная real PostgreSQL parity может потребовать isolated DB runtime; до её получения T09 остаётся открытой. Applicable changed-adapter backend/parser profiles фиксируются заранее; отсутствующие службы/зависимости — blocker соответствующей задачи, не успешная приёмка. Платный live LLM/новый quality benchmark и hardware power-loss campaign не требуются. Owner выбирает license/contribution/security policy; инвентаризация разрешена, выдумывать лицензию запрещено. Required unresolved owner decision фиксируется blocker, а не сужением goal. Реальный release/push отсутствует; release fixtures используют disposable local bare remotes.

## Последовательный backlog

### T00 — backlog and traceability

Предполагаемый коммит: `docs: remediation backlog`.

- **T00.C01** Зафиксированы неизменённые master review и восемь ролевых отчётов с SHA256 и review SHA.
- **T00.C02** Каждый F01–F11, D01–D61, 161 частное design-замечание и все ролевые дефекты сопоставлены задаче без пропусков.
- **T00.C03** Документационные требования 1–9 и DoD 1–8 сопоставлены задачам; ограничения и сохраняемые гарантии включены в общий gate.
- **T00.C04** Backlog имеет строгий последовательный порядок, проверяемые критерии для каждой задачи и отдельный предполагаемый коммит.
- **T00.C05** Определены независимые роли приёмки, расчёт полноты, повторная приёмка изменённого diff, запрет SKIP=PASS и журнал SHA.
- **T00.C06** Проверка целостности источников и полноты реестра выполнена; implementation и исторические результаты не изменены.

### T01 — remediation contracts

Предполагаемый коммит: `docs: remediation contracts`.

- **T01.C01** Опубликован error/recovery contract: protection/parent cancellation suppressive; callback protocol/ordinary errors и usage overrun не становятся bounded success при deadline; сохраняются причины и разрешённый failed journal.
- **T01.C02** Определены clock composition min(parent, ledger, local), injected-clock gates на callback/clone/project/delivery и exact-once settlement независимо от истечения.
- **T01.C03** Опубликована two-valued truth table missing/equal/unequal для каждого supported scalar/operator, NOT/AND/OR; null/wrong-kind и validated lookup precondition заданы явно.
- **T01.C04** Identity domain зафиксирован как valid UTF-8 без молчаливой нормализации; namespace/key/name/policy/history согласованы, valid ID hashing сохраняется; invalid response vs invalid config errors разделены.
- **T01.C05** Определены unique exact page reference и допустимые media/cell sharing, callback/read delivery gates и authorized-input precondition.
- **T01.C06** Определены release reviewed SHA, publishable-module/file/ref allowlist, dirty/staged/untracked policy, caller preservation, candidate retry и none/partial/unknown publication; v2 требует отдельного import-version плана.

### T02 — deadline precedence

Предполагаемый коммит: `fix: deadline precedence`.

- **T02.C01** F01 устранён в Run/RunOwn/RunObserved: callback, settlement и gate causes сохранены; mixed protection+deadline suppresses все payload/side output; failed journal разрешён только своим контрактом.
- **T02.C02** AAA regression matrix охватывает planner/backend/assessor/encoder, protection/protocol/ordinary error, joined deadline, known usage overrun, unknown usage, parent cancel и pure local bounded partial; один dispatch и settlement.
- **T02.C03** D32 и graph:20 проверены на extraction/graphexpand/graphsummary: simultaneous causes не теряются и protection classification сохраняется.
- **T02.C04** Current error GoDoc/README синхронизированы; targeted race tests соответствующих пакетов PASS.

### T03 — release isolation

Предполагаемый коммит: `fix: release isolation`.

- **T03.C01** F02: release source строится от reviewed SHA; stage ограничен разрешёнными manifests, push ограничен перечисленными release refs.
- **T03.C02** AAA local bare fixtures доказывают отсутствие unrelated untracked files/tags remotely; caller files остаются целыми.
- **T03.C03** Dirty tracked/index policy и existing module-tag collision проверяются до mutation; publishable modules отделены от examples.
- **T03.C04** Scope runbook и script contract синхронны; fixtures и shell syntax check PASS.

### T04 — release recovery

Предполагаемый коммит: `fix: release recovery`.

- **T04.C01** F03: ошибки до tag/после tag/push не меняют caller branch/index/worktree и чужие refs.
- **T04.C02** Candidate manifest содержит source/candidate SHA, версию, intended refs и publication status; повтор использует тот же candidate, failed local tag не повышает версию.
- **T04.C03** Atomic push применяется при поддержке; none/partial/unknown состояния имеют явный inspect/recovery протокол без blind ref deletion.
- **T04.C04** AAA rejecting-hook, existing-tag и partial/unknown fixtures PASS; runbook соответствует поведению.

### T05 — extraction deadlines

Предполагаемый коммит: `fix: extraction deadlines`.

- **T05.C01** F04: model получает min(parent, shared-ledger, local) deadline; injected shared clock проверяется на completion/clone/project/delivery.
- **T05.C02** AAA controlled clocks/barriers: ранние parent/ledger/local deadlines, leap during model/project/clone, just-before success; expired result пустой, errors.Is DeadlineExceeded и usage settled один раз.
- **T05.C03** D28/graph:12: MaxInputBytes явно source-text bound либо отдельный envelope cap; CountInputTokens покрывает actual request; limits docs синхронны.
- **T05.C04** Extraction/budget targeted race tests PASS без hidden retry.

### T06 — resolution identity

Предполагаемый коммит: `fix: resolution identity`.

- **T06.C01** F05: malformed namespace/key/name/relation key и config/policy identities отвергаются до grouping/hashing с zero result; история следует выбранному UTF-8 domain.
- **T06.C02** AAA U+FFFD, ff/fe, equal/distinct valid tuples, tuple boundaries и relation path PASS; valid IDs остаются совместимыми или предоставлен явный migration contract.
- **T06.C03** Resolver/identity/history GoDoc и README синхронны; targeted race tests PASS.

### T07 — layout reference admission

Предполагаемый коммит: `fix: layout references`.

- **T07.C01** F06: duplicate exact page references отвергаются Validate/Project до ImageText и projection, для equal/different-length texts.
- **T07.C02** Разрешённые cell/image sharing определены и проверены; добавление page только в hash не используется для маскировки неоднозначного Loader.
- **T07.C03** Distinct refs имеют независимо разрешимые цитаты; AAA zero payload/callback и control tests PASS; contract docs обновлены.

### T08 — lexical callback gates

Предполагаемый коммит: `fix: lexical callback gates`.

- **T08.C01** F07: context/read проверяются до и после каждого metadata callback; первый cancel/revoke запрещает следующий codec и дальнейший clone/delivery.
- **T08.C02** AAA snapshot и managed cache hit/miss, cancellation+ordinary codec error дают empty protected result, one callback; normal ranking сохраняется.
- **T08.C03** Raw BM25 borrowed metadata contract сохранён; targeted race tests и current docs PASS.

### T09 — portable filter parity

Предполагаемый коммит: `fix: filter parity`.

- **T09.C01** F08: atomic PG predicates two-valued до composition; Eq/Neq/In/order и nested NOT/AND/OR согласованы с core для всех supported scalar kinds.
- **T09.C02** Portable conformance покрывает missing/equal/unequal, null/wrong-kind policy; exact int64 >2^53 и injection tests сохранены.
- **T09.C03** На isolated real PostgreSQL corpus query и DeleteByFilter возвращают те же IDs, что core matcher; SQL fake/translation не заменяет real parity.
- **T09.C04** Targeted tests и live PG profile PASS с versions/corpus/command evidence; недоступность DB — pending blocker, не PASS.

### T10 — neo4j delivery

Предполагаемый коммит: `fix: neo4j delivery`.

- **T10.C01** F09: public Retrieve/private retrieve/DeliverRead покрывают success/empty/ordinary partial paths.
- **T10.C02** AAA cancel-on-Runner-return и projection error after cancel дают empty protected result, errors.Is Canceled, один traversal; ordinary partial и success сохраняются.
- **T10.C03** Runner bridge scoped/pinned unsupported profile сохранён; current docs и targeted race tests PASS.

### T11 — structured transport

Предполагаемый коммит: `fix: structured transport`.

- **T11.C01** F10: headers/body/transport context cancel/deadline сохраняются до sanitized classification, zero output/unknown usage, body закрыт, один dispatch.
- **T11.C02** F11: ForceQuery/query/fragment/credentials/opaque config отклоняется до network; valid custom base path/slash даёт exact endpoint.
- **T11.C03** D55/providers:1: общая internal URL/error policy либо common contract suite предотвращает drift без public transport framework.
- **T11.C04** Fake Doer AAA body timeout/parent cancel/ordinary I/O/valid baseline и URL matrix PASS; docs синхронны.

### T12 — retrieval composition

Предполагаемый коммит: `refactor!: retrieval composition`.

- **T12.C01** D01–D14 и все retrieval:01–18 имеют retain/change/contract решение с конкретным rationale и evidence, включая measurement decision для cache/copy.
- **T12.C02** Consumer inventory выполнен перед ranking façade removal; один internal composition engine выбран по снижению drift; любой retained independent path обоснован ответственностью.
- **T12.C03** Explicit fusion degradation/conditional predicate, zero ExecMeta reset, single partial-result authority и outer joined causes закреплены контрактом и meaningful regressions.
- **T12.C04** Nil/typed-nil validation, custom resolver capability, threshold stage и MergeKey evidence policy согласованы; Artifact/config ownership и unsupported unused Embedder решены.
- **T12.C05** Budget scopes/calls/known-unknown и bounded ports описаны; retained guarantees, targeted race/conformance и examples PASS; оптимизации имеют before/after measurements.

### T13 — lifecycle contracts

Предполагаемый коммит: `fix: lifecycle contracts`.

- **T13.C01** D15–D22 и lifecycle:01–18 имеют явные dispositions; terminology/capture-vs-pin, namespace CAS, positive reuse observation, unknown recovery, one-step cleanup и Complete semantics опубликованы.
- **T13.C02** D16 AAA strict/partial capture wrong/matching namespace, empty/nonempty/malformed snapshot: zero publication/no writes/ErrProtocol и unchanged success.
- **T13.C03** Permanent reservations/ABA, order-sensitive replay/CAS trusted boundary и exact inventory fences сохранены; nil-context convention определён.
- **T13.C04** History-scaled CPU/bytes и filesystem profile описаны в current lifecycle guide; clone/index оптимизации только с замерами и differential ownership/order validation.
- **T13.C05** Targeted lifecycle race/replay/fence tests PASS.

### T14 — graph ingestion contracts

Предполагаемый коммит: `docs: graph contracts`.

- **T14.C01** D23–D27,D29–D31,D33 и соответствующие graph:01–22 имеют dispositions; stale resolution roadmap заменён current package boundaries.
- **T14.C02** Source-local endpoint closure и cross-document example, canonical name/mention kind, stable generic kinds, history attestation/cardinality/platform/bytes описаны.
- **T14.C03** Graphsummary repeated admission cost, selected citations vs complete derivation inputs/revocation, owned delivered text и set coverage policy явно зафиксированы.
- **T14.C04** Graph fail-zero limits, seed-only heuristic и volatile managed unavailable/no raw fallback сохранены; normalization stage Build/Stage определён.
- **T14.C05** Минимальный complete generic composition/lifecycle example runnable; targeted race tests PASS; support/group/summary оптимизации имеют upper-bound before/after benchmarks.

### T15 — source and chunk contracts

Предполагаемый коммит: `fix!: source contracts`.

- **T15.C01** D34–D43 и ingestion:01–28 имеют dispositions; overlap/separator keep/drop, Markdown/sentence grammar и BYO segmenter coverage явно определены.
- **T15.C02** Chunk Total0/index/UTF-8/admission/gaps и contextual callback/order policy согласованы; meaningful property/regression tests проверяют выбранное поведение.
- **T15.C03** Projection all-or-nothing/partial и descriptor/chunk identity/StorageID/URI authority едины в docs/code; borrowed metadata и authorized layout input явно описаны.
- **T15.C04** Mapping duplicate/case/Unicode policy закреплена; exact source authority/no-latest, raw administration, support ancestry/whole media сохранены.
- **T15.C05** OCR partial/ImageText empty override, multimodal transport URL/provider/host boundaries и evidence membership/rank/input-work semantics решены и проверены.
- **T15.C06** Current source→layout→chunk→index→authorized resolve guide/examples runnable; targeted race tests PASS; range-validation оптимизации имеют realistic before/after measurements.

### T16 — local retrieval contracts

Предполагаемый коммит: `fix!: local retrieval contracts`.

- **T16.C01** D44–D50 и tensor:01–20 имеют dispositions; metric-dependent normalization, negative/>1 scores, candidate/token/dimension work и cancellation checkpoints согласованы.
- **T16.C02** Candidate budget/dedup/first order/catalog scan/MaxRecords units и lexical snapshot/transient cache bounds описаны; explicit B=0/K1=0 поддержаны либо решение обосновано контрактом.
- **T16.C03** Повторные admission/clone gates сохранены либо объединены с fence tests; generation invalidation и raw borrowed ownership явно описаны.
- **T16.C04** Durable root/power-loss vs process-crash, flock handle/local profile, redacted errors/trusted root, explicit retired rejection и dense/tensor duplication решены.
- **T16.C05** Targeted race/process-restart/retired/fence tests PASS; CPU/cache/clone optimizations имеют before/after measurements без threshold fitting.

### T17 — storage bridge contracts

Предполагаемый коммит: `fix!: storage bridge contracts`.

- **T17.C01** D51–D54 и storage:D1–D17 имеют dispositions; codec canonical scalar/empty attributes и filter constructor sentinels согласованы.
- **T17.C02** ES SynonymMap ownership и single tokenizer dispatch проверены mutation/stateful fixtures; no double tokenization.
- **T17.C03** PG identifier quoting/policy, host batch bound, equal-distance tie policy и Rows.Close outcome policy явно определены.
- **T17.C04** Neo4j generic filters и projection vs full snapshot validation закреплены; affected counts exact/unknown без fabrication; raw operations остаются explicit administration.
- **T17.C05** Bridge-only/live certification границы сохранены; targeted wire/conformance и applicable changed-adapter real-service profiles PASS, skips pending.

### T18 — provider and PDF contracts

Предполагаемый коммит: `fix!: provider contracts`.

- **T18.C01** D55–D59 и providers:01–18 имеют dispositions; shared URL/error/credential policy согласована, JSON unknown-fields vs duplicate/case/surrogate отдельно, choice Index missing/null явно отвергаются либо контракт обоснован.
- **T18.C02** Space vs Model identity/attestation, purpose/metric, defaults/explicit caps и vector-row/Cohere query+docs units опубликованы; current API naming согласовано.
- **T18.C03** Known usage при rejected payload сохраняется; unknown не оценивается как known; Cohere prefix/input с error не считается reranked success.
- **T18.C04** PDF sanitized invalid-input/internal-error policy, supported platform/dependency versions/fingerprint и output-vs-RSS bounds опубликованы.
- **T18.C05** Compilable provider examples/GoDoc, fake transport tests и actual PDF parser profile при изменении PDF PASS; live платный LLM benchmark не требуется; metric оптимизации только с замерами.

### T19 — observation contracts

Предполагаемый коммит: `docs: observation contracts`.

- **T19.C01** D60 и arch-docs:A-D03–06 имеют dispositions; finite payload-free events/enums, known/unknown, pair/drop/callback units и cooperative custom error boundary опубликованы.
- **T19.C02** Synchronous serialized callbacks/no workers/retry и optional host bridge/OTel сохранены; targeted privacy/order/reentrancy/race tests PASS.

### T20 — public documentation

Предполагаемый коммит: `docs: public guides`.

- **T20.C01** D61 и assigned arch-docs findings имеют dispositions; root onboarding содержит install/Go/schema/meta/local BM25/explicit Read/partial-protection-error handling, runnable и gofmt.
- **T20.C02** Stable integration/lifecycle/source/layout/ownership/score/limits/recovery guides находятся в public topic paths с working links; README разгружен, языковая структура согласована.
- **T20.C03** DOC1–DOC8 синхронны final API: units/defaults/bounds, callback ownership, source authority/derivation deps, profile capability matrix и exact release runbook.
- **T20.C04** DOC9: license/versioning/contribution/security policy inventory сохранён; существующие owner decisions используются, отсутствующие явно отмечены без выдуманной лицензии; required owner decisions — blocker.
- **T20.C05** Исторические task13–19 acceptance/evidence сохранены с отдельным remediation reference; checklist percentages не выданы за production readiness; conformance suite limits и external advisory runner описаны.

### T21 — fresh acceptance and release tooling

Предполагаемый коммит: `test: fresh acceptance`.

- **T21.C01** D61 и arch-docs:A-D12,A-D14–18 выполнены: broad wording blacklist заменён targeted symbols/contracts/links/snippets; fresh acceptance отделена от cached developer runs.
- **T21.C02** All modules перечислены без двойного тестирования examples; explicit -count=1/race и examples build; normal CI имеет GOWORK=off matrix и validated Go/lint versions.
- **T21.C03** Fuzz functions перечисляются индивидуально с bounded budgets; пустые PHONY targets исправлены/удалены.
- **T21.C04** Publishable-module manifest отделяет examples, script platform/portable editor и semantic import-version guard documented; local candidate clean consumer smoke PASS.
- **T21.C05** Targeted CI/tooling fixtures PASS; real release/push не выполнялся.

### T22 — final acceptance

Предполагаемый коммит: `test: remediation acceptance`.

- **T22.C01** Все T00–T21 приняты двумя независимыми агентами и имеют отдельные commits; каждый F/D/raw requirement закрыт evidence/disposition, все blockers разрешены.
- **T22.C02** Fresh all-module lint и race tests, planner/resilience/conformance examples PASS; versions/commands/exit codes и skips зафиксированы.
- **T22.C03** GOWORK=off matrix и isolated release-candidate clean consumer install/build/tests PASS; fake/local remote остаётся единственным release publication fixture.
- **T22.C04** F08 real PG query/delete parity и остальные applicable changed-backend/parser profiles PASS; непроверенные live enforcement/quality/power-loss claims не заявлены.
- **T22.C05** Все фактические bounds/clone/union/cache/metric оптимизации имеют before/after relevant measurements и сохранённые concurrency/protection properties.
- **T22.C06** Final API docs/examples/trace registry и git diff --check сверены с current state; финальные независимые completeness 100% и correctness PASS, затем отдельный commit.

## Журнал

- T00 accepted: completeness 6/6 = 100%, coverage 262/262 = 100%; correctness PASS. Отчёты: [completeness](acceptance/T00-completeness.md), [correctness](acceptance/T00-correctness.md). Implementation не менялся; commit SHA фиксируется следующим journal update.

- T00 commit: `517179b87209a1fb2a41c47057f766dfdce3f959` — `docs: remediation backlog`.
- T01: [target contracts](../contracts/remediation.md) подготовлены; implementation не менялся, приёмка pending.

- T01 accepted: completeness 6/6 = 100%, coverage 16/16 = 100%; correctness PASS. Initial F11/empty-map/optional-namespace findings исправлены, обе приёмки повторены. Contract SHA256 `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df`; commit SHA фиксируется следующим update.

- T01 commit: `4776e89` — `docs: remediation contracts`.
- T02 in_progress: F01/D32 по принятому контракту. Acceptance matrix: planner/backend/assessor/encoder × ordinary/protocol/protection/joined deadline/usage overrun; Run/RunOwn/RunObserved, pure local/parent cancellation, exact dispatch/settlement; graph callbacks preserve simultaneous causes.

- T02 accepted: completeness 4/4 = 100%, correctness PASS after standalone renderer P2 fix. Independent race eight packages PASS, lint 0 issues, adversarial privacy/cause repros PASS; [completeness](acceptance/T02-completeness.md), [correctness](acceptance/T02-correctness.md). Commit SHA записывается следующим journal update.

- T02 commit: `6d18bca7403386643bc4b7a0e6be3039785a86e5` — `fix: deadline precedence`.
- T03 in_progress: reviewed SHA / isolated checkout / publishable-module and file/ref allowlists. Persistent candidate recovery остаётся T04.

- T03 accepted: completeness 4/4 = 100%, coverage F02/A-F01 2/2 = 100%, correctness PASS. Relative-origin/colon P2 resolved and both independent gates repeated; nine fixture methods and independent adversarial cases PASS. Reports: [completeness](acceptance/T03-completeness.md), [correctness](acceptance/T03-correctness.md). Commit SHA записывается следующим journal update.

- T03 commit: `b32a7e35106b8442f5a306559bf5589ff8e391ad` — `fix: release isolation`.
- T04 in_progress: persistent exact candidate, inspection states, retry and atomic partial completion; scope per accepted target contract.

- T04 accepted: completeness 4/4 = 100%, F03/A-F02 coverage 2/2 = 100%, correctness PASS. Canonical manifest P2 fixed and both gates repeated on final tree; 12 recovery + 9 isolation methods, independent interruption/identity/canonical-tamper fixtures PASS. Reports: [completeness](acceptance/T04-completeness.md), [correctness](acceptance/T04-correctness.md). Commit SHA записывается следующим journal update.

- T04 commit: `f282696c8fdf15e13bcbba70fca8e6bb146e6b14` — `fix: release recovery`.
- T05 in_progress: local/ledger clock composition, boundary gates, source-text byte and actual-envelope token contracts.

- T05 accepted: completeness 4/4 = 100%, F04/G-F01/D28/graph:12 coverage 4/4 = 100%, correctness PASS. Final authority clock leap and Prepare error+expiry findings resolved; both independent gates repeated, fresh six-package race/lint and adversarial timer/accounting checks PASS. Reports: [completeness](acceptance/T05-completeness.md), [correctness](acceptance/T05-correctness.md). Commit SHA записывается следующим journal update.

- T05 commit: `fd70ce6fc493620ec33448a40aacc7c19011f65e` — `fix: extraction deadlines`.
- T06 in_progress: reject malformed UTF-8 identity before grouping/hashing and history serialization; preserve valid JSON tuple hashes and host normalization ownership.

- T06 accepted: completeness 3/3 = 100%, F05/G-F02 coverage 2/2 = 100%, correctness PASS. Fresh five-package race/lint and independent late-host/raw-persisted/legacy-digest probes PASS. Reports: [completeness](acceptance/T06-completeness.md), [correctness](acceptance/T06-correctness.md). Commit SHA записывается следующим journal update.

- T06 commit: `2b9d3c5918764b61488f2adfc245a92db70060c4` — `fix: resolution identity`.
- T07 in_progress: unique normalized page text references at document admission; selector-addressed cell/image sharing remains explicit host retention contract.

- T07 accepted: completeness 3/3 = 100%, F06/ING-F01 coverage 2/2 = 100%, correctness PASS. Fresh four-package race/lint and independent nonadjacent duplicate/representation controls PASS. Reports: [completeness](acceptance/T07-completeness.md), [correctness](acceptance/T07-correctness.md). Commit SHA записывается следующим journal update.

- T07 commit: `dd81b21abe6e49f1f52b4c03c09c5496a6f25c83` — `fix: layout references`.
- T08 in_progress: per-codec/clone context/read gates across snapshot capture, scoring, managed cache miss/hit and delivery; raw metadata remains borrowed.

- T08 accepted: completeness 3/3 = 100%, F07/T-F01 coverage 2/2 = 100%, correctness PASS after cause-retention P2 fix. Scoped readfailure preserves errors.Is without payload/text unwrap leakage; both independent gates repeated. Fresh four-package race/lint and callback/cause/privacy probes PASS. Reports: [completeness](acceptance/T08-completeness.md), [correctness](acceptance/T08-correctness.md). Commit SHA записывается следующим journal update.

- T08 commit: `528164b711ef219b510a6056ae0335ba1ff0df7f` — `fix: lexical callback gates`.
- T09 in_progress: leaf-level PG boolean normalization, portable all-scalar missing/equal/unequal corpus and actual isolated PostgreSQL query/delete parity.

- T09 accepted: completeness 4/4 = 100%, F08/S1 coverage 2/2 = 100%, correctness PASS. Both independent real PostgreSQL 17.11 / pgvector 0.8.7 profiles PASS (81 records, 56 predicates, 55 actual deletes, 12 malformed admission samples), targeted race and tagged lint PASS, separate boundary/injection probes PASS. Reports: [completeness](acceptance/T09-completeness.md), [correctness](acceptance/T09-correctness.md). Reviewed SHA256 checked against current diff before bookkeeping. Additional whole-root race completed exit 1 solely existing broad wording blacklist/history README issue, explicitly assigned T21.C01; remaining root packages PASS, full log [here](acceptance/T09-root-race.log). No global suite PASS claimed. Commit SHA записывается следующим journal update.
- T09 runtime cleanup: stopped only owned `ragy-task20-t09-528164b` container after both reviewers completed; --rm removed disposable instance. Other containers untouched.

- T09 commit: `a2fc0a48ed24112cf3a1579e153ce2936454effb` — `fix: filter parity`.
- T10 in_progress: final delivery for Neo4j Runner bridge, suppress cancellation on success/empty/partial and preserve observed error causes.

- T10 accepted: completeness 3/3 = 100%, F09/S2 coverage 2/2 = 100%, correctness PASS. Final delivery covers success/empty/partial and cancellation, observed ordinary causes retained safely; final documentation scope clarification independently accepted. Fresh three-package race/lint and independent actual deadline/cause/privacy/scoped/pinned probes PASS. Reports: [completeness](acceptance/T10-completeness.md), [correctness](acceptance/T10-correctness.md). Reviewed SHA256 verified before bookkeeping. Commit SHA записывается следующим journal update.

- T10 commit: `e960f9e645433e1b8b93d95b18797a8e5156fa47` — `fix: neo4j delivery`.
- T11 in_progress: shared parsed URL endpoint admission and sanitized context-first transport/body errors.

- T11 accepted: completeness 4/4 = 100%, F10/F11/providers:01/P-01/P-02 coverage 5/5 = 100%, correctness PASS. Shared parsed URL/endpoint and sanitized context-first error helpers prevent structured/provider drift; ordinary adapter-specific classes retained. Fresh eight-package race/lint and independent canceled non-2xx headers, partial/complete/oversized body cancellation, wrapped sentinel/privacy, escaped actual Post and invalid endpoint probes PASS. Paid live smoke excluded, no SKIP counted as PASS. Reports: [completeness](acceptance/T11-completeness.md), [correctness](acceptance/T11-correctness.md). Reviewed SHA256 verified before bookkeeping; commit SHA записывается следующим journal update.
