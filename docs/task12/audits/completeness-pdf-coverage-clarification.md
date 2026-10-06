# Уточнение E-17: PDF coverage и immutable evidence

Denominator независимого аудита сохраняется: **195**. Это clarification существующего атома E-17, не добавление API или нового mandatory scope. Verdict E-17 остаётся U до проверки actual combined recorder fixture; этот документ не засчитывает планируемую реализацию.

## Вывод

**Достаточно host-derived `Outcome=Partial`, `Reason=MissingEvidence` из фактически сохранённой OCR/layout partial coverage, вместе с exact original source/locator export. Новый generic coverage projection contract не обязателен.** Автоматический экспорт BYOT domain metadata не нужен и противоречил бы privacy/ownership границам задачи.

ТЗ §4.5 требует «Parser adapter сохраняет ... coverage и diagnostics. Partial parse не становится complete после chunking». §5.7 уточняет: «Page 2 partial OCR означает partial document coverage; coverage не повышается после chunking». Это обязательная семантика parser/projector/index/artifact пути, но ТЗ не задаёт единый wire format для per-page parser coverage внутри retrieval recorder.

ТЗ §5.5 отдельно требует outcome/partial reason, revision association, stages/scores и explicit unavailable. §2.1/§4.3 оставляет domain metadata у host и предусматривает explicit projector/clone/serialized snapshot. Следовательно consumer вправе проецировать фактическую retained parser coverage в общие outcome/reason; exporter не должен сериализовать произвольный TMeta, authorisation metadata или diagnostic strings автоматически.

`evidence.Input.Coverage` имеет тип `retrieval.ReadCoverage`: immutable admission/branch/inventory coverage (`retrieval/coverage.go:25–34`, schema `ragy.read-coverage/admission`). Это другой предмет гарантии, чем `layout.Document.Coverage`/`layout.Projected.Coverage` (`layout/layout.go:18–24`, `layout/project.go:12–19`). `CompleteReadCoverage()` при actual complete admission и `Outcome=Partial` из parser evidence не противоречат друг другу, если documented declared profile явно различает эти scopes. Не маркировать admission branch skipped из-за unreadable OCR и не приписывать ветке failure, которого не было.

Existing evidence validator допускает `Partial/MissingEvidence` при корректном ReadCoverage (`evidence/record.go:264–294`). Это working целевой контракт, не compatibility layer.

## Что должно быть подтверждено combined fixture

1. Реальный parser output partial/OCR observations → typed host metadata и original mapping → durable Stage/Publish → новый adapter/ledger → actual retrieved r1 artifacts. Partial marker, representation/revision и payload fingerprint должны происходить из actual retained source; не вписывать константу Partial исключительно в recorder assertion.
2. Host projection читает owned фактическую parser coverage из retrieved/persisted metadata. Хотя records ранжированы по успешно доступным частям, recorder для полного declared PDF profile сохраняет `Partial/MissingEvidence`, не повышая partial document до complete. Если selected fixture включает complete и partial документы, projection policy и scope outcome должны быть явно определены.
3. Recorder сохраняет exact original locator identity/supports (page/span/cell/image где они реально observed), original/derived distinction и actual stage ranks/scores. Недоступные поля отмечаются unavailable, не придумываются. `ReadCoverage` сохраняет реальное admission observation, независимое от parse coverage.
4. Strict Go wire + executable JSON Schema round-trip; mutation caller hits/meta/mapping после capture не меняет immutable record. Default query/text/auth privacy и allowlisted location policy соблюдены.
5. Denied/deleted r1 до export блокируется current source admission и не выдаёт record/content с latest r2 substitution. Required sink failure сохраняет retrieval fact без нового retrieval, если этот path заявляется в fixture.
6. Decoder/export association сверяется с actual original references и geometry из retained parser data. Consumer-required source/location capability без actual observation — explicit error.

Per-page OCR diagnostics можно хранить в host retained payload и разрешать consumer по exact source refs. Если host хочет отдельный exported parser-coverage document, это optional host schema/projection с explicit privacy policy; не делать такую схему обязательной ragy runtime dependency ради E-17.

## Граница E-17

E-17 требует подтвердить actual cross-capability recorder path, а не новый universal data model. PDF часть может быть закрыта описанной host projection и integration tests. Tensor часть остаётся самостоятельной проверкой actual candidate/rerank IDs/budget/native scores/source revisions в immutable record; generic PDF coverage API её не заменяет.

Этот clarification не меняет implementation, матрицу или исходный числитель. После остальных fixtures и final checkpoint независимый re-audit повторно оценивает исходные195 атомов.
