# Host migration to the canonical context bridge

The root retrieval API is unchanged. Replace field-copy integration glue with the optional consumer's explicit mapping and publication contracts. No compatibility wrappers are supplied.

Before, unsafe glue often copied index text, treated zero as a score, carried old citation spans through formatting and used ordinary JSON encoding of a context message:

```go
// Index data is not a canonical publication decision.
message.Parts = []contexty.ContentPart{contexty.TextPart{Text: indexDocument.Content}}
raw, _ := json.Marshal(message) // Extensions are excluded.
```

After, configure host types and policies through `Bridge[P,R,A,M,U]`:

```go
mapper.Map = func(ctx context.Context, d retrieval.Document[Meta]) (bridge.Reference, error) {
    return bridge.Reference{Scope: hostScope, RecordID: d.Meta.CanonicalID,
        Revision: d.Meta.CanonicalRevision}, ctx.Err()
}
mapper.Project = projectCanonicalRecord // Uses only authorized canonical payload.
mapper.Options.UncertaintyType = "host.extractor-uncertainty/1"
mapper.Publish = managedSink.Publish // Register this sink in canonical Config.Sinks.
output, err := mapper.Run(ctx)
if err != nil { return err }
// Public tool output is output.Public. Persist output.Durable in host-private storage.
restored, err := bridge.Decode[Uncertainty](ctx, output.Durable,
    bridge.Registry[Uncertainty](mapper.Options.UncertaintyType))
```

Supply exact scope and canonical namespace/ID/revision rather than inferring them from an index ID. The source projection must prove retained source authenticity and attach revision-bound mappings. The bridge checks locator source/revision against canonical provenance; it cannot authenticate arbitrary source storage for the host.

Keep `ScoreAbsent`, native zero, negative scores, scale identity and score history intact. Search uses an explicit ordinal ranking policy; its computed candidate score is separate from native SearchSignal evidence. Do not use native scores from different models/configurations as one comparable scale.

Formatting a final prefix/suffix translates decoded-text byte spans. Arbitrary `Rewrite` or `TruncateRunes` clears final-text spans and preserves source-only supports with unavailable precision. JSON escaping has its own byte coordinates. Recompute mappings in host code if you need exact quotations after an arbitrary transformation; labels and ID matches are insufficient.

Always use `Registry[U](identity)` and the message extension codec for persistence. Rebuild old saved context lacking evidence from trusted sources, or explicitly reject it/mark precision unavailable. Do not guess source revisions or exact spans. Changing the uncertainty schema requires a new host identity; unknown identities fail until the correct codec is registered.

Register all managed context/index sinks with canonical Forget. `Run` captures a fence before search; its synchronous callback executes under `WithDerivedWrite` with all exact lineage. Never recursively call the same canonical store from this callback. An external effect may have committed even if a callback returns an error: reconcile the operation rather than assuming rollback. Physical sink failure leaves Forget pending while canonical reads remain blocked. Bind queued upserts to the same epoch and make lifecycle cleanup idempotent. `CleanupSink.Apply` must tombstone and clean all artifacts in the supplied batch, acknowledging only confirmed completion.

Budget final UTF-8 bytes, runes, durable JSON bytes and final public role/text model tokens separately. Supply a tokenizer identity and callback. Configure host data-role policy explicitly. Keep metadata/evidence in private durable storage and expose only the public projection. Restore bytes only as data: perform fresh canonical eligibility checks before serving saved context; decoding is not a renewed authorization grant.
