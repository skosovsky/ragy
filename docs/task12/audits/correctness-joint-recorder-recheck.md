# Независимая перепроверка joint candidate recorder

Дата: 5 октября 2026 года. Проверена окончательная mechanical worktree после предыдущего consumer publication association defect. Аудитор не изменял production implementation/tests и не опирался на completeness percentage. Live quality acceptance не заявляется.

## Закрытые подтверждённые находки

AUDIT-ACCEPT-01 (предыдущий tensor-only publication record) исправлен переносом actual recorder fixture в external joint_read module. Действительно используется один durable joint dense+tensor ledger и одна Ready publication; обе стадии исполняются с тем же immutable read binding. Independent positive reproduction проверяет presence обоих targets, equality dense/tensor publication identity и original UTF8 source representation каждого exported hit. Source allowlist теперь берётся из отдельно предоставленного host input.Lexical inventory, а не выводится из returned hits. Original SourceMapping создаётся до ingest и сохраняется в настоящих persistent payloads.

При первой проверке нового consumer envelope дополнительно воспроизведено отсутствие final MaxSim→CandidateIDs проверки: подмена только final hit ID вне фактического candidate universe проходила Go decoder. Затем отдельно воспроизведены два decoder/schema mismatch: synchronized non-SHA configuration+recipe и duplicate dense candidate controls/stage IDs. Все три consumer validator defects сообщены основной исполнительнице и устранены. Это нарушения consumer evidence association/validation, не доказанная core authorization leak.

## Текущая независимая adversarial проверка

Actual overlay command, executed after current formatter/target checks:

```sh
GOWORK=off GOMODCACHE=/tmp/ragy-implementation-mod-cache GOCACHE=/tmp/ragy-implementation-go-cache go test -race -overlay /tmp/ragy-audit-joint-recorder-overlay.json ./joint_read -run TestAuditJointRecorder -count=1
```

CWD: examples/conformance. Reproduction source: `docs/task12/audits/repro_joint_recorder_recheck_test.go.txt`. Overlay отображает virtual joint_read/audit_recorder_overlay_test.go в `/tmp/ragy-audit-joint-recorder-test.go`; production tests не записывались. Actual log: `docs/task12/audits/joint-recorder-recheck.txt` — PASS, 2.930s.

Подтверждены:

- actual positive shared publication и independently approved original UTF8 references;
- final hit ID вне actual recorded candidate universe отклоняется;
- изменённый final native score отклоняется;
- изменённый final rank отклоняется;
- duplicate final hit отклоняется;
- неверная final stage identity отклоняется;
- synchronized configuration+record.recipe вне canonical SHA256 отклоняется;
- duplicate dense candidate document ID/control отклоняется.

Decoder связывает stage names/status, unique actual candidate IDs, unique dense IDs, one-to-one original source support sets между dense/candidate observations и exact delivered WireHit subset. CandidateIDs enumeration хранится отдельно от dense ranking и MaxSim ranking. Candidate-observations действительно собираются из actual query result: score/rank не присваиваются по позиции в CandidateIDs. Fixture intentionally меняет dense vs MaxSim order; native 2/1/-1 сохраняются. CandidateBudget берётся из actual RerankResult, не реконструируется по длине returned list.

Independent schema script прогнан отдельно после последнего изменения Python validator: `PYTHONPATH=/tmp/ragy-schema-validator python3 docs/task12/verify_tensor_run_schema.py` — PASS, 1 actual positive / 17 negatives / cross-field association. Schema проверка дополняет Go decoder, не заменяет actual scoped source permission policy.

## Проверенное состояние и full gates

SHA256 examined test sources:

- recording_unix_test.go: `82c0ab3e6f1df932925ef4966011c587948e04cb74dbd503bf426df166f760a8`;
- recording_envelope_unix_test.go: `24be46397edac5e5a11c01074a1ff991a3a1f5bead5f9ed479a8fe9be7f3a6a3`.

Основная исполнительница сообщила terminal EXIT0 обоих текущих процессов make lint/make test (71927/72548); аудитор прочитал соответствующие логи `docs/task12/results/joint-recorder-full-lint.txt` и `joint-recorder-full-test.txt`. После этого production/test sources не менялись, что подтверждено повторными hashes выше. После full gates менялся только standalone schema validator, который независимо повторён успешно. Предыдущие root BUG-001/BUG-002, snapshot ownership/freshness, BM25 finite scores и exact durable cleanup fixes остаются в состоянии ранее независимой recheck; приёмка этих исправлений не заменяется новой consumer fixture.

## Вердикт и ограничения

В проверенном mechanical checkpoint все подтверждённые findings этого аудитора устранены; существенные новые подозрения в изменённых recorder paths проверены. Новых подтверждённых неисправленных correctness defects не осталось в проверенной области. Это не доказательство отсутствия всех возможных ошибок.

Actual joint/PDF additional integration race results из предыдущего acceptance report остаются применимыми к неизменённым paths. Не выполнены live model quality experiments, нет paid/model calls, не проверены production remote storage services, hardware power-loss и произвольные custom adapters/host callbacks. Original source authority/retention — explicit host contract; fixture approved inventory не является IAM или криптографическим proof of content. JSON schema/decoder checks отвергают механические несогласованности, но не удостоверяют доверенность producer, который согласованно подделывает все поля.

Полнота 100% и завершение общего goal этим отчётом не утверждаются. До live acceptance task12 в полном объёме не принята, release/issue closure не выполнялись.
