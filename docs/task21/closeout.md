@skosovsky, блокер снят, реализация поставлена в **ragy v0.8.1**. Core API и обязательные зависимости ragy не изменены. Интеграция находится в отдельном optional consumer module, с BYOT и явными host policies.

Что изменить в вашем коде:

1. Заменить field-copy glue на контракт `Bridge[P,R,A,M,U]`: `Map` возвращает `Reference{Scope, RecordID, Revision}`, а `Project` строит документ из разрешённого canonical payload. Index ID и index text сами по себе не доказывают identity, revision или право публикации.
2. Вызывать `Bridge.Run` с синхронным `Publish` зарегистрированного managed sink. Admission выполняется через canonical `WithDerivedWrite`; context/index sinks должны участвовать в Forget. `SnapshotSink` — пример in-process sink; durable storage, retries/reconciliation и production purge остаются у host. `CleanupSink.Apply` подтверждает только завершённую tombstone/cleanup операцию. Не входить рекурсивно в тот же canonical store из publication callback.
3. Сохранять native absence/scale/history/rank отдельно от производного search ranking. Для presence-aware API: отсутствие — `memy.Score{}`, настоящий ноль — `memy.ScoreOf(0)`. Не заменять отсутствие нулём и не сравнивать несовместимые native scales.
4. Сохранять `Published.Durable` через зарегистрированный `Registry[U]`, восстанавливать через `Decode[U]`; обычный `json.Marshal(message)` исключает extensions. Публиковать только `Published.Public`, без private metadata/sidecar. Canonical extractor identity, losses и uncertainty observations сохраняются независимо от optional typed host uncertainty.
5. Задать host data-role policy и отдельные final UTF-8 byte/rune/durable JSON/model-token limits с вашим tokenizer. Prefix/suffix переводит byte spans в координаты final decoded text; `Rewrite`/`TruncateRunes` удаляет final exact spans и помечает delivery uncertain. Source-only supports после изменения текста не объявлять exact citations к новому тексту.
6. Старые context без обязательного evidence безопасно пересобрать из доверенных sources либо отклонить. Не угадывать source revisions/exact spans. Перед serving сохранённого snapshot заново проверить canonical eligibility: структурный decode не даёт нового разрешения.

```go
// Было: index text выдаётся напрямую, extension теряется при plain JSON.
message.Parts = []contexty.ContentPart{contexty.TextPart{Text: indexDocument.Content}}
raw, err := json.Marshal(message)

// Теперь: mapper содержит explicit scope/revision, canonical projection и managed Publish.
output, err := mapper.Run(ctx)
// В tool response идёт output.Public; в private storage — output.Durable.
restored, err := bridge.Decode[Uncertainty](ctx, output.Durable,
    bridge.Registry[Uncertainty](mapper.Options.UncertaintyType))
```

Optional module является reference consumer: запускайте его из Git checkout тега либо перенесите glue в свой host module. Это не новый импорт в core; nested example module не включён в root Go module zip. В host module используйте опубликованные зависимости и уберите development self-replace.

Поставка и приёмка:

- `make release-patch RELEASE_SOURCE=730e94137f5aee0e0213be1a1e90e0c405e54f97` завершён успешно; release commit — `b12fedb529edfa7304fa713e823d114e496254ef`, все 11 refs подтверждены. Patch выбран для optional example/docs/CI без изменения core API.
- Независимая полнота **100%: AC1–AC10 PASS**; adversarial review не оставил подтверждённых дефектов в проверенном scope. Полный `make acceptance` и повторная release acceptance прошли.
- [Полный CI](https://github.com/skosovsky/ragy/actions/runs/37605796008): **34/34 jobs PASS**, включая checkout и published semantic/race/demo. Checkout использует явные проверенные current source refs, а не устаревшие default branches.
- После публикации новый Git checkout `v0.8.1` содержит все 13 файлов optional consumer. Его собственный runner повторно прошёл `published --ragy-ref v0.8.1` с `GOWORK=off`, без replace: свежие race tests и demo PASS. Опубликованная композиция: ragy `v0.8.1`, memy `v0.3.1`, contexty `v0.13.1`; origin commits/checksums сохранены.

Ссылки: [исходный релиз и consumer](https://github.com/skosovsky/ragy/tree/v0.8.1/examples/context-bridge), [mapping contract](https://github.com/skosovsky/ragy/blob/v0.8.1/docs/context-bridge.md), [migration guide](https://github.com/skosovsky/ragy/blob/v0.8.1/docs/context-bridge-migration.md), [финальные reviews](https://github.com/skosovsky/ragy/tree/codex/context-bridge/docs/task21/reviews), [release/published verification records](https://github.com/skosovsky/ragy/tree/codex/context-bridge/docs/task21/results).

Проверки ограничены указанными offline canonical/renderer/codec и managed lifecycle fixtures; live providers и distributed purge ими не сертифицируются. Требования issue выполнены, закрываю issue.
