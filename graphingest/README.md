# Source-bound graph ingestion

`graphingest.Pipeline` composes the existing extraction, resolution and materialization contracts through explicit typed ports. The host supplies ontology, identity rules, support admission, attribute cloning and its attempt budget ledger. The core has no default ontology or model selection.

Build mapped snippets from chunks or layout evidence, then call:

```go
pipeline, err := graphingest.New(graphingest.Config[ACL, Kind, Relation, Attr, Meta]{
    Extraction: extractor,
    Resolution: resolver,
    Materialization: materializer,
})
result, err := pipeline.Build(ctx, binding, ledger, snippets, materializationRequest)
```

`result.Plan` contains a planned manifest and managed payload. `result.Decisions` preserves the resolver's decisions, and `result.Usage` reports extraction usage with explicit known/unknown state. Build performs no index writes, retries or publication. A failed stage yields no publishable plan.

The host explicitly calls lifecycle `Prepare`, `Stage` and `Publish` with that plan and payload. Publication remains subject to the executor's inventory and readiness checks; a partially completed or uncertain stage cannot claim publication.

[The executable integration example](pipeline_integration_test.go) runs mapped splitting, bounded extraction, resolution, materialization and actual managed graph/lifecycle handoff, including an interrupted staging scenario. [Stage failures](graphingest_test.go) also verify cancellation and suppression of partial plans.

The former Stage/Provider/raw Upsert facade is removed. Independent raw graph storage ports remain available under `graph`; they do not claim managed lifecycle guarantees.

[Composition guide](composition.md) identifies every mandatory callback and the
complete deterministic runnable profile. Ontology/aliases and semantic truth are
host policy; extraction, resolution, history and materialization remain independent
retrieval-domain packages. General workflow/agent loops remain outside this pipeline.
