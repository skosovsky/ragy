# Complete typed graph composition

The smallest existing end-to-end profile is
[pipeline_integration_test.go](pipeline_integration_test.go), with five explicit
types: struct{} access, string entity kind, string relation kind, graphAttrs and
graphMeta. It uses a deterministic extraction model fixture and real local
managed graph/lifecycle ports. It performs no external model calls. Run on the
supported darwin/linux filesystem profile:

```sh
go test -race -count=1 -run '^TestGraphPipelineExplicitManagedLifecycleHandoff$' ./graphingest
```

The file is complete; the top-level README snippet only shows wiring.
`graphPipeline` constructs all three stages and supplies every mandatory callback:

| Stage | Host responsibilities supplied in the runnable file |
|---|---|
| Extraction | Finalized access schema, ontology/config identity, finite snippet/text/entity/relation/support caps, clock/duration, access/attribute cloners, access attributes, original snippet admission, entity/relation validators, quote, actual request token counter and one-dispatch model |
| Resolution | Ontology/policy IDs, population bounds, entity/relation validators, explicit namespace/key/canonical-name identity policy, relation key, attribute clone/equality and original support admission |
| Materialization | Matching ontology/policy and graph schema, fact/support caps, attribute/metadata clones, node/edge projection, source support admission |
| Lifecycle | Durable filestore, registered target/managed adapter, clone/validate payload ports and explicit host Prepare/Stage/Publish calls |

`graphPlan` makes an original byte locator, mapped source text and a chunk, creates
a finite shared ledger, then passes mapped snippets and explicit source identity,
manifest/idempotency key and content/payload fingerprints to Pipeline.Build. The
example's constant token count, no-op admission, validators and literal fingerprints
are deterministic fixture ports. A real host must count its actual request, check
retained source access/domain kinds and derive fingerprints from actual content
and config. They are not production defaults. Ports are bounded, cooperative and
concurrency-safe; clone ports must deeply own their BYOT values.

`graphLifecycle` constructs the managed target with separate record/admission caps
and the Executor. `checkGraphHandoff` explicitly prepares, stages, publishes,
captures a publication, binds a read and verifies two nodes/one edge and one model
call. Its interrupted-stage branch dispatches a real stage then reports deadline;
Publish is rejected until the host performs explicit reconciliation. No background
retry or claim of certainty is made.

Build returns the actual Decisions. When durable decision history is required,
retain the extractor's actual typed input together with these decisions in
[history.Record](resolution/history/README.md); do not construct a synthetic
Result or use Capture as proof of ontology correctness. Pipeline does not append
history or automate a cross-store transaction. Host coordinates that optional
record and its lifecycle plan; a history append is not a publication receipt.
