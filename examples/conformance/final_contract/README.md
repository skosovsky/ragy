# Final public composition contract

This is an external consumer package in `example.com/ragyconsumer`. Run with `GOWORK=off`: its `go.mod` resolves the public core and OTel adapter through explicit local replaces. Consumer metadata, cloning, authorities, file catalog/loader, formatting/tokenizer policies and model scripts are defined here; none enters core.

From `examples/conformance`:

```sh
GOWORK=off GOCACHE=/tmp/ragy-task19-conformance-cache go test -race -count=1 ./final_contract ./joint_read ./observation_contract
GOWORK=off GOCACHE=/tmp/ragy-task19-conformance-cache GOLANGCI_LINT_CACHE=/tmp/ragy-task19-conformance-lint /opt/homebrew/bin/golangci-lint run --allow-parallel-runners ./final_contract/... ./joint_read/... ./observation_contract/...
```

The final package composes actual BM25, dense persistent storage, typed JSON codecs, source.Reader over actual files, RRF, packing, evidence, bounded recipes and lifecycle filestore. The cache test uses a barrier around actual MemoryCache.Load to revoke authority before the next delivery clone. The hydration test revokes during actual file loading. Scripted planner/assessor ports are host test doubles; they establish dispatch/accounting contracts, not provider effectiveness. Retrieved hostile instructions are ordinary evidence data; no agent or prompt-injection immunity is tested.

Existing `joint_read` proves real dense/lexical/tensor/graph scope and publication admission through all cache/projected/OTel decorator permutations, nested aggregate/fallback/rescue/route paths, and payload-reader revocation. Existing `observation_contract` proves actual BM25 pipeline/cache/branch diagnostic privacy, bounded sessions and cancellation. The final package reuses these suites.

A PublicationPin preserves lifecycle metadata only. Cleaner payload retention remains an explicit host policy. A registered old pin blocks metadata retirement while cleanup follows its declared retained references; releasing it permits explicit retirement. Tests must not be read as automatic payload availability promises.

See [conformance evidence and limitations](../../../docs/task19/conformance.md).
