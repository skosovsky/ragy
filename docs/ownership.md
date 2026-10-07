# Ownership and cooperative ports

Ragy carries host types through `Document[TMeta]`, request intent, request metadata and execution metadata. A value copy of a struct containing a map, slice or pointer does not own the referenced data. Keep host inputs immutable throughout execution and concurrent reads. Filter wire attributes and private adapter JSON maps are boundary representations, not replacements for typed business metadata.

| Boundary | Ownership |
|---|---|
| Request value and `WithPlan` | Shallow copies; intent, metadata and their references remain host-owned. |
| `CopyRequestOptions` | Owns core option/planning collections and range endpoints; does not deep-copy arbitrary host types. |
| ResultSet construction and `Documents` | Copies ragy-defined document slices, source mappings/supports and score history; BYOT metadata references retain host ownership. |
| Mutable BM25 input metadata | Borrowed immutable metadata; caller must not mutate references while indexed or read. Constructor owns search fields, parameters and synonym collections. |
| Managed lexical/graph and persistent capture | Explicit host codecs/cloners and immutable captures provide the package's stated ownership. A metadata pin alone does not retain payload or source blobs. |
| Callback/configuration values | Host owns their captured state and synchronization; copying a function does not isolate its state. |

Identity, grouping, codec, cloning, projection, token counting, clock and comparison policies must satisfy the purity, stability and concurrency rules of their ports. Parallel branches and contextual chunking may invoke host code concurrently. Use deterministic identity policies; do not derive IDs from mutable counters, current time or map iteration order. Do not mutate request intent or metadata from a branch.

Cancellation is cooperative. Read/context gates run before and after relevant payload callbacks, including cache-hit clone/projection paths. A callback already running must return before the library can reject its output and stop the next callback. Context is not a timeout sandbox for arbitrary Go code; there is no detached worker or forced preemption. Error classification also executes host `Is`, `As` or `Unwrap` methods cooperatively. Sanitized error text does not make arbitrary error objects trusted or bounded.

The [observer](../observation/README.md) is synchronous and serialized. Hosts own any queue, worker, export retries and shutdown protocol; callback failure is distinct from downstream export failure. Accepted observation operations require an explicit End. Calling Stats inside a callback is supported; explicit same-session reentry violates the observer contract.

Retained original source remains the authority for exact text. Declared mappings, hashes, index payload and metadata pins cannot replace authorized revision-specific source resolution. Citations identify admitted source supporting the selected payload; derivation dependencies record the inputs that produced an artifact and may be broader. Do not promote unused dependencies into citations or treat provenance as factual correctness.

See [source](../source/README.md), [layout](../layout/README.md), [chunking](../chunking/README.md), [lifecycle](../lifecycle/README.md), [managed lexical](../lexical/managed/README.md) and [graph ingestion](../graphingest/README.md) for package-specific capture and authority rules.
