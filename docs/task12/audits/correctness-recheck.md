# Независимая affected-state перепроверка

Дата: 5 октября 2026 года. Область: три подтверждённые находки первого correctness audit. Это не финальная сертификация task12: conformance/recorder/cache acceptance ещё изменяются, live experiments не выполнены. Production implementation и production tests аудитор не менял.

## AUDIT-CORRECT-01: исправлено с дополнительной поправкой freshness

Snapshot теперь хранит host CloneMeta и клонирует metadata каждой выдачи. Независимый overlay проверил pointer → map → nested slice: изменение исходного input после capture и изменение первого результата не меняют pinned corpus; второй запрос с тем же binding сохраняет tenant и nested value. Clone-on-every-Documents не требуется: передаваемая каждому batch host metadata ownership отделена от retained corpus, общий ResultSet комментарий больше не обещает arbitrary BYOT deep immutability.

Первая affected-state проверка выявила оставшийся дефект: при clone callback failure после ctx cancellation возвращался raw private-clone-error вместо protected context.Canceled. Проблема была сообщена основной исполнительнице и исправлена в обоих местах: Retrieve и initial capture. Текущий код проверяет freshness immediately после clone callback до обработки cloneErr (`lexical/snapshot.go:82–87`, `127–132`).

Дополнительные независимые отрицательные сценарии:

- delivery clone callback одновременно cancel context и возвращает обычную ошибку: zero docs и protected context.Canceled;
- initial capture clone callback делает то же: nil snapshot и protected context.Canceled;
- capture/delivery callbacks отзывают trusted Authority и возвращают sensitive error: protected ErrUnavailable без raw callback failure.

## AUDIT-CORRECT-02: исправлено

Исходное публичное воспроизведение с K1=NaN/+Inf и B=NaN теперь получает ErrInvalidArgument от конструктора; successful invalid result отсутствует.

Независимый overlay отдельно допускает гигантский finite K1=MaxFloat64 и B=1, индексирует документ из трёх одинаковых terms и вызывает настоящий Retrieve. Переполнение native арифметики возвращает ErrProtocol и zero docs. Native score не clamped и не заменён вымышленным normalized значением. Обычная finite конфигурация остаётся supported.

## AUDIT-CORRECT-03: исправлено

Dense и tensor cleanup сверяют SameTargetInventory с durable retired manifest. Независимые overlays создают реальные persistent файлы и ledger, регистрируют artifact с двумя distinct supports, выполняют Stage/Publish/tombstone/Cleaner cleanup, затем проверяют:

1. Перестановка обоих supports и artifact inventory сохраняет корректный set: InspectCleanup возвращает complete.
2. Изменение одного original support при тех же artifact refs: InspectCleanup отклоняет запрос.

Таким образом, проверка подтверждает и fail-closed mutation rejection, и отсутствие ложного отказа на reordered valid inventory после исчезновения physical catalog.

## Evidence

Reproduction sources сохранены для воспроизводимости:

- `docs/task12/audits/repro_snapshot_recheck_test.go.txt`;
- `docs/task12/audits/repro_dense_cleanup_recheck_test.go.txt`;
- `docs/task12/audits/repro_tensor_cleanup_recheck_test.go.txt`.

Go overlays добавляют virtual `audit_overlay_test.go` к существующему пакету и используют fixture helpers. Отображение выполнено через `/tmp/ragy-audit-snapshot-overlay.json`, `/tmp/ragy-audit-dense-cleanup-recheck-overlay.json`, `/tmp/ragy-audit-tensor-cleanup-recheck-overlay.json`; production test files не записывались.

Команды: `go test -race -overlay <overlay> ./lexical|./dense/persistent|./tensor/persistent -run TestAudit -count=1`, с GOMODCACHE=/tmp/ragy-implementation-mod-cache и GOCACHE=/tmp/ragy-implementation-go-cache. Actual outputs:

- `docs/task12/audits/snapshot-recheck.txt`;
- `docs/task12/audits/dense-cleanup-recheck.txt`: PASS, 1.632s;
- `docs/task12/audits/tensor-cleanup-recheck.txt`: PASS, 1.592s.

Три прежних находки больше не воспроизводятся в проверенном состоянии; промежуточная freshness ошибка устранена и перепроверена. Это evidence только затронутых контрактов, а не отсутствие других ошибок или полнота 100%. Финальные оба аудита должны выполняться на окончательной worktree после оставшихся изменений и фактических live experiments.
