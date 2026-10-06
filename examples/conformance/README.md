# External adapter conformance

This module uses the import path `example.com/ragyconsumer`, outside ragy's module
namespace. It imports the public `contracttest` package and supplies its own intent,
request metadata, source metadata and schema fields. No internal helper imports or
workspace resolution are needed:

```sh
cd examples/conformance
GOWORK=off go test -race -v ./...
```

The reference adapter uses the shipped BM25 engine for scoped candidate selection
and an instrumented host payload port for materialization. Nine direct-backend
scenarios cover allowed/contradictory/unsupported queries, missing binding, policy
revocation/expiry, cancellation, mid-I/O revocation and deadline propagation.

Negative fixtures reject an undeclared adapter before its Retrieve method runs,
detect forbidden payload materialization hidden by final post-filtering, and detect
payload I/O before a leaf gate even when final denied output is empty. A separate
composition fixture retains explicit partial coverage without weakening scope.
Persistent dense/tensor fixtures stage and publish actual payload files through the
durable lifecycle store, reopen adapters and run the same nine direct-backend scenarios.
Their optional PayloadReader wraps bounded physical-file reads and observes admitted
references before delivery. No additional scope gate in the host request projection
can hide a leaf failure. Separate planner tests cover empty, contradictory and unsupported
conditions; unsupported plans must fail before materialization. These fixtures certify
the declared local-filesystem scope profile; custom external service semantics remain
host responsibilities.

Integer metadata fixtures persist adjacent IDs above the exact float64 integer range
and int64 bounds, reopen dense/tensor adapters and check Eq/In/NotEq plus mandatory
Eq/conflict against observed physical payload reads. Invalid host codec outputs must
fail before index files are written. Mandatory NotEq is explicitly rejected by the
reference Eq/In/And authorization profile. These are local-filesystem and transport
boundary tests; they do not claim live validation of an external database service.

Managed graph fixtures use BYOT node/edge metadata and actual durable
prepare/stage/publish/capture before running the public nine-case scoped read suite.
A public node reaches a private bridge leading to another public node; a foreign
seed is also requested. Payload cloning observations must include neither the
private/foreign facts nor the public node behind that bridge. A forwarding wrapper
records context deadlines without adding admission/delivery gates or filtering.
Three direct planner cases additionally cover empty, contradictory and unsupported
filters. Scope is enforced before clone/projection, and revocation during cloning
suppresses output. Graph records remain in process memory; this does not certify
external engine persistence or every custom graph adapter.

Eight direct FindByIDs cases exercise allowed/conflicting/unsupported lookup,
missing binding, revoked/expired/canceled reads and revocation during metadata
cloning. Private/foreign and nonexistent IDs return no payload; duplicate IDs
remain deduplicated. A public node behind the private bridge is accessible by
explicit ID, while traversal still cannot reach it through that bridge. Returned
facts preserve exact original support references. This is scope-aware lookup,
without retrieval Backend projection wrappers.

`recipe_comparison` supplies the fixed text corpus/qrels and explicit actual BM25/model capture and offline scoring commands
for baseline/three-strategy executions. Complete orchestration passes protocol
fixtures; live model capture/tokenizer qualification remain outstanding. Its tests verify
metrics, rejection of incomplete captures and conservative recommendation gates.
