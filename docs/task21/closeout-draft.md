# Issue closeout draft — not posted

Post only after both dependency modes pass, independent acceptance reaches 100%, and the source release is verified. Add actual commit, CI and release links before publication. Until then issue #4 remains open.

## Message to the issue author

The integration is implemented in the optional `examples/context-bridge` consumer module. Core retrieval API and dependencies are unchanged. Replace field-copy host glue with explicit canonical mapping and a managed publication callback:

```go
// Before: index payload and a plain message marshal lose canonical gates/evidence.
message.Parts = []contexty.ContentPart{contexty.TextPart{Text: indexDocument.Content}}
raw, err := json.Marshal(message)

// After: configure Bridge[P,R,A,M,U] with host policies and real canonical state.
mapper.Map = func(ctx context.Context, d retrieval.Document[Meta]) (bridge.Reference, error) {
    return bridge.Reference{Scope: scope, RecordID: d.Meta.CanonicalID,
        Revision: d.Meta.CanonicalRevision}, ctx.Err()
}
mapper.Project = projectCanonicalRecord
mapper.Publish = managedSink.Publish // Register the sink in canonical Config.Sinks.
output, err := mapper.Run(ctx)
// Expose output.Public; persist output.Durable in host-private storage.
restored, err := bridge.Decode[Uncertainty](ctx, output.Durable,
    bridge.Registry[Uncertainty](mapper.Options.UncertaintyType))
```

Required changes in your host code:

- Provide exact scope, canonical ID/revision and retained source ID/revision mappings. Index text does not become accepted canonical knowledge. `Bridge.Run` performs canonical recall and `WithDerivedWrite` admission before the synchronous `Publish` callback; do not recursively call the same store inside that callback.
- Register context/index sinks with canonical Forget. `SnapshotSink` is an in-process reference sink; production durability and reconciliation remain host responsibilities. Use `CleanupSink.Apply` for explicit lifecycle tombstone/cleanup, and acknowledge only confirmed completion. Fence late index events; retry cleanup idempotently.
- Retain `ScoreAbsent`, native scale/history and native rank, including zero. The ordinal search ranking policy is separate. Do not compare incompatible native score scales.
- Preserve the registered sidecar via `output.Durable` and `Registry[U]`. Plain message JSON does not persist extensions. Canonical extractor identity, losses and uncertainty observations survive even without an optional typed host extension.
- Prefix/suffix wrapping translates final decoded-text spans. `Rewrite`/`TruncateRunes` removes final exact spans and marks delivery uncertain; preserved source-only supports are not citations to the transformed final text.
- Configure data-role policy, independent final UTF-8 byte/rune/durable JSON/model-token limits and a host tokenizer. Return only `output.Public`; metadata and sidecar stay private.
- Rebuild saved contexts without evidence from trusted sources or reject them explicitly. Never guess source revisions/exact spans. Decoding a snapshot is structural validation; freshly check canonical eligibility before serving it.

Reference material for the published source: mapping contract `docs/context-bridge.md`, migration `docs/context-bridge-migration.md`, executable demo, semantic fixtures, `docs/task21/reviews/`, and recorded checks under `docs/task21/results/`. Offline fixtures do not certify live providers or distributed purge.

Close issue https://github.com/skosovsky/ragy/issues/4 only after this message includes verified implementation, review, CI and release links.
