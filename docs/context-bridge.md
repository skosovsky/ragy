# Context bridge contract

The optional `examples/context-bridge` module demonstrates host-owned retrieval composition. The root module has no integration dependency. Business payload, metadata, source reference, principal and extractor uncertainty remain generic host types.

| Boundary | Contract |
|---|---|
| Retrieval ID → canonical reference | Explicit host mapper returns exact scope, canonical ID and revision. No equality-based identity inference. Cross-scope or ambiguous mapping fails. |
| Search score | Deterministic rank order with canonical ID tie-break; reciprocal ordinal is a declared search policy, never the native retrieval score. Native state, scale, rank and history remain in evidence. |
| Canonical materialization | Real canonical Recall and host projection provide payload. Index content never supplies accepted knowledge. Projection must preserve ID and source revision associations. |
| Publication | Capture epoch before search; `WithDerivedWrite` validates all selected canonical revisions and authority under exact-scope exclusion with Forget. The synchronous callback must not call that store recursively. |
| Source citation | Exact mappings refer to retained source representation and the decoded final text. Host projection owns source authenticity; bridge verifies locator source/revision against canonical provenance. |
| Formatting | Prefix/suffix wrapping preserves offsets by translation. Arbitrary rewriting keeps supports with unavailable precision. Whole-candidate renderer omission remains distinct from host final-text truncation. |
| Durability | Sidecar is a typed context extension registered by schema identity. Message codec encode/decode, with a fresh registry, retains evidence. Strict decoding rejects duplicate/unknown fields, lost inventory, invalid relational spans, contributor-to-canonical mismatches and source/revision mismatches. A checksum detects accidental wire corruption; it does not authenticate a sender. |
| Privacy | Public tool JSON is a fixed projection of message role and text. No metadata or sidecar is exposed. Durable sidecar is host-private and contains source evidence. Errors retain safe cause classes only. |
| Resource | Bytes/runes apply to final decoded message; JSON budget applies to the complete durable envelope; tokenizer measures the public role/text envelope. These units are independent. |

The supported publication profile is synchronous managed writes with distinct canonical inputs and no renderer dedup callback. Merged contributor policies are unsupported and rejected explicitly. A durable host sink must be registered with canonical Forget; completed publication does not certify distributed purge, remote IAM revocation atomicity, disk erasure or live providers. Empty retrieval uses a fresh epoch check and no publication callback. Missing source, stale canonical references or an unsupported codec never become empty success. Public role is explicit host policy, restricted to user/tool data for this example.

Typed uncertainty uses a host-selected JSON-capable type and codec identity. JSON ownership is deliberate at this optional durable boundary, not a reflection-based clone policy in core. Canonical extractor identity, losses and uncertainty observations are always preserved independently of the optional typed host uncertainty. A missing typed host extension is represented by a nil pointer. Native rank zero stays zero in inputs; snippet rank records the renderer's positional fallback. Duplicate contributors and contradictory final delivery flags are rejected. Projection metadata uses the renderer's explicit CloneMeta callback.

Wire schema: [context bridge sidecar](../examples/context-bridge/sidecar.schema.json). Structural JSON Schema and executable validation jointly define the contract: UTF-8 span boundaries, exact text identity, source mapping validation, score semantics and ordinal associations are relational checks.

See the module README and [migration](context-bridge-migration.md) for setup and host responsibilities.

Host callback inputs must remain stable for the duration of Run. Typed uncertainty requires deterministic JSON roundtrips under its declared identity. Snapshot decoding is structural validation, not renewed canonical authorization or sender authentication.
