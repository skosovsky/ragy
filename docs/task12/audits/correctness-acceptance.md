# Независимый аудит mechanical acceptance checkpoint

Дата: 5 октября 2026 года. Аудитор проверил актуальные actual external joint_read composition fixtures, PDF durable lifecycle recording и candidate/MaxSim recording. Production implementation и production tests не изменялись. Независимость от completeness audit сохранена. Этот отчёт не считает live quality acceptance выполненной.

## AUDIT-ACCEPT-01 — P2: tensor recorder fixture теряет dense publication association

Код: `examples/conformance/tensor_comparison/recording_unix_test.go:27–54,134–189`, `examples/conformance/tensor_comparison/profile.go:183–190,245–253` (проверять актуальные строки после дальнейших edits).

Actual dense candidates исполняются под p.denseRead, а actual MaxSim под p.tensorRead. Build создаёт два разных persistent stores и independently captured publications. Evidence.Capture записывает обе observed stages с Complete outcome под p.tensorRead; последний содержит только tensor target. Wire stages не содержат собственную publication identity. Следовательно, record связывает dense stage с tensor-only logical publication, хотя фактически dense snapshot другой.

Независимый overlay использует действительный recordedTensorStages, actual persistent target calls и тот же Capture. Assert проверяет, что единственный записанный publication binding включает все executed targets. Actual expected FAIL:

```text
complete record publication 38c9b2aad1fa01b134f10034df861e93c0557651aaa12d4c00ca6e38f286f2ef omits executed dense snapshot 2e1ee0940518664053562fec71ec43e36c3f70fa456a9175283ae1a972e549a3; stages=2
```

Evidence: `docs/task12/audits/tensor-publication-association.txt`; reproduction: `docs/task12/audits/repro_tensor_publication_association_test.go.txt`; overlay `/tmp/ragy-audit-tensor-publication-overlay.json`. Actual command from examples/conformance: `GOWORK=off GOMODCACHE=/tmp/ragy-implementation-mod-cache GOCACHE=/tmp/ragy-implementation-go-cache go test -race -overlay /tmp/ragy-audit-tensor-publication-overlay.json ./tensor_comparison -run TestAuditTensorRecorderUsesAllExecutedPublicationTargets -count=1`.

Это consumer acceptance/provenance gap, не подтверждённая authorization leak core. Generic Capture доверяет host SourceAdmission; core не обязан выводить ledger identity из stage name. Admission map в fixture сформирован из самих observed hits: он позволяет продемонстрировать deleted admission entry, но не подтверждает, что каждую stage извлекли из recorded publication.

Исправление: capture один host logical binding с обоими actual target inventories до обеих стадий и исполнять dense/tensor с ним; записывать тот же binding. Или предусмотреть честную association с per-stage publication identity. Одного переименования record/publication или заявления Complete недостаточно. В negative test проверить отсутствие нужного target до I/O/export, а не только изменение frozen score.

## Проверенные новые пути

### Joint composition

Независимый actual race запуск passed: joint_read 8.642s. Fixtures действительно создают persistent dense/tensor и volatile managed lexical/graph, durable common publication; forbidden private/foreign records физически присутствуют. Nested aggregate/fallback/rescue/route и planner/capability negotiation исполняют реальные target paths. Strict unsupported branch preflights до planner/target I/O, explicit partial остаётся Partial, foreign planner даёт пустое пересечение. Во время настоящего dense ReadPayload revocation подавляет выдачу и дальнейший secondary dispatch. Supplemental fallback/rescue tests действительно достигают secondary, а не засчитывают только successful primary.

Ограничение: эти fixtures используют Concurrency=1 для детерминированного post-I/O revocation assertion; они не заменяют параллельные fan-out tests, проверенные предыдущим аудитом.

### PDF recording

Независимый actual parser/durable race запуск passed: adapters/pdf 2.446s. Test действительно выполняет parser, Stage/Publish/reopen pinned r1, retained r2 и original layout resolver. Evidence source projection использует original SourceMapping.Supports вместо dense-vector index identities. SourceAdmission вызывает scoped Reader.Lookup с полным exact original batch; deleted/denied r1 отклоняется до любого payload load, не подставляет retained r2. Wire remains immutable после producer mutation. Partial parser/OCR coverage помечает Outcome Partial/MissingEvidence; ReadCoverage Complete означает выполненные retrieval branches, а не обещание полного OCR. При отключённых snippets record не утверждает, что derived image description является original image bytes/text.

OCR текст в fixture моделируется отдельно; это не проверка OCR accuracy. Перечисление parser/source metadata и admission port принадлежит host, не core IAM.

### Tensor recording

Existing actual candidate/MaxSim recorder test passed independently (1.420s): persistent candidates/native 2/1/-1, frozen values, required missing capability и re-admission отказ. Этот PASS не закрывает AUDIT-ACCEPT-01: existing assert проверяет только candidate count/native scores и допускает потерянный dense publication association. Tiny corpus не подтверждает meaningful latency percentiles или production quality gain.

## Проверки и пределы

Actual logs: `docs/task12/audits/acceptance-joint-tensor.txt`, `docs/task12/audits/acceptance-pdf.txt`. Все использовали count=1, -race, external module GOWORK=off; PDF использовал RAGY_PDF_PYTHON из configured runtime. Root BM25/dense/tensor prior fixes не изменились; previous correctness-recheck evidence остаётся применимым в своей affected scope.

Full make lint/test logs начали формироваться основной исполнительницей. Наличие trailing successful output не заменяет подтверждённый process exit; аудитор не объявляет full gate PASS до terminal сообщения. Не выполнялись paid calls, live model experiments или remote production storage. Credential/model/tokenizer absence остаётся непроверенной приёмкой. Отсутствие других новых подтверждённых defects не доказывает отсутствие ошибок; после исправления AUDIT-ACCEPT-01 нужна перепроверка окончательного consumer path и оба итоговых аудита final worktree.
