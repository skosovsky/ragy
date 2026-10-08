# Optional canonical context bridge

This is a host composition example in a separate Go module. It uses the actual retrieval renderer, canonical lifecycle and context message codec; the retrieval transport is an offline fixture. The core library has no imports from this example.

Run from this directory:

```sh
GOWORK=off go run ./cmd/demo
GOWORK=off go test -race -count=1 ./...
```

The demo commits approved canonical knowledge, supplies deliberately hostile index text, renders only canonical payload, roundtrips a registered evidence extension, publishes through `WithDerivedWrite`, and invokes Forget on the managed context sink. Its code-point counter is a deterministic test measurer; production hosts supply their actual tokenizer on the final role/text JSON.

`Bridge[P,R,A,M,U]` takes typed retrieval, identity mapper, canonical projection, metadata clone, optional uncertainty, read binding, resource limits and a synchronous publish callback. `Run` captures the epoch before retrieval and publishes under exact-scope lineage exclusion against Forget. Unknown/stale canonical references fail explicitly. The API assumes non-nil contexts and cooperative host callbacks with stable inputs during a run. Renderer dedup callbacks are unsupported by this one-input-per-snippet profile and rejected; canonical inputs remain distinct.

`Options.UncertaintyType` selects the host JSON contract; use the same identity with `Registry[U](identity)`. `Decode[U]` verifies sidecar presence, schema, exact-text association, UTF-8 span validity and score evidence. Durable JSON is host-private. `PublicJSON` emits only role/text; never publish `Published.Durable` as tool output. The example admits only user/tool data roles.

`SnapshotSink[U]` is a managed in-process host sink demonstrating owned serialized snapshots and canonical cleanup. It is not durable database storage. `CleanupSink` adapts a host-owned finite tombstone/cleanup procedure to canonical Forget; return an error until the complete exact inventory is deleted. Tests execute actual managed lexical Stage/Publish/Cleaner operations. Register every managed sink when constructing the canonical engine; external delivery, backups and distributed purge remain host responsibilities.

The durable codec stores source evidence, native scores and uncertainty without private domain metadata. Native score absence remains absent. Search ranking uses reciprocal ordinal; native signals and scales are separate evidence. A final rewrite/truncation leaves source-only support evidence but clears final-text spans and marks delivery uncertain. Prefix/suffix wrapping translates exact final-text spans.

Portable verification from the repository root:

```sh
make test-integration
```

Go integration tests copy the consumer into a temporary module. Checkout mode uses
current ragy source and pinned memy/contexty revisions fetched into disposable
repositories; no sibling checkout is required. Published mode drops all replacements
and resolves a pinned baseline with GOWORK=off. Both run semantic race tests and the
demo. Release supplies its exact new version to the published consumer test.
