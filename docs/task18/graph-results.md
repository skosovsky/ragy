# Graph implementation and validation

Managed graph now requires separate positive `MaxRecords` (one write/basis) and `MaxAdmissionRecords` (total selected facts before admission). A selected fact cardinality preflight rejects oversized reads before materializing the scoped view. Original source supports are indexed once per selected target inventory. Outbound/inbound adjacency is built only after filters and conflict removal, and only when both endpoints remain admitted; each expansion examines incident edges. Retired lifecycle manifests, including empty inventories and captured retired candidates, cannot be confirmed.

The low-level `graph.Port` remains unchanged. Admission still scans selected facts; lifecycle Load/Validate and version selection still scan retained metadata. Cardinality bounds are not byte or complete CPU bounds. [Contract](graph-contract.md) and [public adapter guide](../../graph/managed/README.md) state these limits.

Validation uses Go 1.26.1 and a task-local Go build cache. The following commands passed (logs under `results/`):

| Command | Log | Result |
|---|---|---|
| `golangci-lint run --allow-parallel-runners ./graph/... ./graphingest/... ./recipe/graphexpand ./recipe/graphsummary` | `graph-lint.txt` | PASS, 0 issues |
| `go test -race -count=1 ./graph/...` | `graph-race.txt` | PASS |
| `go test -race -count=10 ./graph/managed -run 'TestAdmission\|TestManagedAdjacency\|TestManagedAdmissionRevocation\|TestAdjacency\|TestConfirmed'` | `graph-adversarial-race.txt` | PASS |
| `go test -race ./graphingest/... ./recipe/graphexpand ./recipe/graphsummary` | `graph-consumer-race.txt` | PASS |
| `go test -race ./lifecycle/integration` | `graph-lifecycle-race.txt` | PASS |
| OpenAI module: `GOWORK=off go test -race ./structured -run '^Test.*Published\|^Test.*Publication'` | `graph-openai-race.txt` | PASS, real in-process HTTP/lifecycle integration; no live service |
| Conformance module: `GOWORK=off go test -race ./graph_comparison -run '^TestActualResolverMaterializerPublishedGraphMatchesGold$'` | `graph-external-gold-race.txt` | PASS, actual scoped published graph matches gold |
| Conformance module: `GOWORK=off go test ./... -run '^$'` | `graph-conformance-compile.txt` | PASS, compile only |

The graph suite retains existing lifecycle cleanup/shared-source, conflict, private-bridge and freshness checks. Added tests cover selected shared facts counting before dedup, excluded host basis counting before scope, exact admission boundary, constructor rejection of zero/negative admission capacity, inbound/outbound/undirected cycles and self-loops, private edges, dangling/conflicting endpoints, retired empty inventories, and an atomic/channel barrier that revokes an actual suspended reader before delivery. Benchmark setup/workload and before/after results are owned separately in TASK18 results; no standalone SLO is claimed here.
