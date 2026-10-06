# Review: external storage bridges + filter

Baseline b63d5e19; inspected pgvector, qdrant, elasticsearch, neo4j, typed filters and access glue. No production source edits. Targeted probes use Go overlay only; no live DB/service ran. Root is clean after probes.

## Confirmed defects

### S1 — P2: pgvector negation has different absent-field semantics from the core filter

References: `adapters/pgvector/store.go:411-417`, `:438`, `:451-464`, `:496-497`, `:548-560`; reference `filter/match.go:52-58`, `:186-190`, `:145-157`.

Optional fields are accepted by Schema.NormalizeAttributes (missing fields are not required). For stored `attributes={}`, core MatchCondition returns true for `category != "x"`, `NOT(category == "x")`, and `NOT(category IN ["x","y"])`. Actual adapter render emits respectively `attributes->>'category' <> $1`, `NOT (attributes->>'category' = $1)`, and `NOT (attributes->>'category' IN ($1,$2))`. PostgreSQL missing-key extraction yields NULL, comparisons and NOT retain unknown; WHERE drops those rows. Hence the same valid Condition has different retrieval and DeleteByFilter effects across built-in backends. Mandatory Eq/In/And scope profile is not a negation profile: do not describe this as a scope bypass.

Proof: `/tmp/ragy-storage-probe_test.go` TestReviewMissingPredicates, overlay `/tmp/ragy-storage-overlay.json`. `GOCACHE=/tmp/ragy-review-go-cache go test -overlay=/tmp/ragy-storage-overlay.json -run '^TestReview' -v ./adapters/pgvector/...` PASS diagnostic; exact output absent reference=true for all three, native SQL above. No live PostgreSQL was executed: database behavior follows official SQL contract, not a fake's invented evaluator.

Sources verified 2026-10-06: [PostgreSQL comparisons](https://www.postgresql.org/docs/current/functions-comparison.html) and [JSON operators](https://www.postgresql.org/docs/current/functions-json.html). These establish comparison unknown for NULL and missing-field NULL respectively.

Fix: define missing-field truth table as part of portable filter contract, render every atomic positive predicate as a two-valued boolean (e.g. comparison IS TRUE), implement != consistently as negation of equality (or IS DISTINCT FROM for non-null expected scalar). Do not only coalesce the outermost WHERE; NOT of a compound still needs leaf normalization.

AAA: Arrange schema fields each scalar kind, rows missing/present equal/present unequal and compound conditions; Act MatchCondition plus adapter query/delete on the same corpus; Assert identity sets equal for Eq/Neq/In/order and nested NOT/AND/OR. Preserve exact int64 and injection tests. Live PG parity profile should be explicit integration evidence; wire test alone checks translation.

### S2 — P2: Neo4j Retrieve omits the final delivery cancellation gate

References `adapters/neo4j/neo4j.go:64-69` only entry gate, runner at `:93-104`, final success `:133`; compare `retrieval/access.go:120-144`, `access/access.go:262-285`, other storage adapters' Retrieve wrappers.

Direct Neo4j Store.Retrieve can expose payload after the context has been canceled during Runner.Traverse. Runner returning a successful result when cancellation races its completion is legitimate; wrapper is still required to check delivery before exposing it. Current method yields nonempty result + nil, although req.Read.Check(ctx) now rejects with protected context.Canceled. Scope/pinned reads are rejected already, so this is cancellation/delivery contract inconsistency, NOT tenant revocation leak.

Proof `/tmp/ragy-neo4j-probe_test.go`, `/tmp/ragy-neo4j-overlay.json`: valid Runner cancels supplied context immediately before returning valid node; diagnostic TestReviewCancellationAtDelivery PASS prints `canceled=context canceled returnedEmpty=false err=<nil>`. Command `GOCACHE=/tmp/ragy-review-go-cache go test -overlay=/tmp/ragy-neo4j-overlay.json -run '^TestReview' -v ./adapters/neo4j/...`. Local Runner only.

Fix: same public Retrieve + private retrieve + DeliverRead pattern as pgvector/qdrant/ES, encompassing successful, empty, and partial projection returns. Do not introduce retries or suppress ordinary partial errors independently of protection.

AAA: Arrange cancel-on-return runner, also projection failure after cancellation and empty snapshot; Act direct Retrieve; Assert errors.Is(context.Canceled), empty result, no rerun. Include unaffected normal/ordinary partial result case.

## Separate architectural / oddity decisions (not confirmed defects)

D1. `pgvector/store.go:374-383` hands raw json.Number to custom codec directly and skips codec on empty wire; Qdrant `:326-334` and ES `:273-299` normalize first. Probe TestReviewPGCodecCanonical using codec expecting schema-canonical int64 prints `canonical tenant: got json.Number` for 9007199254740993; built-in JSONCodec remains correct because it normalizes itself. MetadataCodec's current interface does NOT explicitly promise canonical Decode input, so classify as contract inconsistency to settle, not precision regression. Prefer adapter-owned normalization then custom codec receives stable scalar kinds on every backend; document empty attributes behavior.

D2. `elasticsearch.go:95` retains caller SynonymMap and nested slices, while BM25 constructor deep clones (`lexical/bm25.go:86-89`). Decide ownership once: clone immutable config maps/slices at construction, document callback concurrency separately. Otherwise post-construction config edits change behavior and concurrent edits can race. No concurrent mutation probe run; not a proven caller-contract violation.

D3. ES calls Tokenizer twice (`elasticsearch.go:129`, `:168`, `:212`) per successful retrieval. Stateful or expensive tokenizers see duplicate work and potentially inconsistent empty admission vs dispatched query. Compute once and pass tokens/expanded string; require stable tokenizer behavior explicitly. Avoid creating a tokenization framework.

D4. pgvector adapter names accepted by `ValidateSQLIdentifier` are interpolated unquoted (`:180`, `:271`), so uppercase names fold and SQL keywords fail. Safe regex prevents injection; this is not an injection defect. Specify lowercase unquoted identifiers vs correctly quote identifiers, including schema-qualified names if intentionally supported. Do not expand identifier policy speculatively.

D5. Filter missing/NULL/wrong-kind contract should be documented independently of backend implementation, with conformance cases. RawAttributes nil rejected, absent accepted; portable missing-field behavior still matters without explicit null. `MatchIR` takes a lookup that can supply NaN/wrong numeric types; decide validated-boundary precondition vs graceful rejection. Don't let arbitrary user maps become domain metadata API.

D6. Neo4j currently exposes a typed Runner abstraction, not Cypher/driver implementation (`neo4j.go:26-29`). Keep that honest in naming and docs; a built-in driver is an optional separate adapter, not a prerequisite for correctness. Existing capability matrix explicitly says no native driver and scoped/pinned unsupported — retain this honesty.

D7. Neo4j Retrieve accepts generic Filters/Plan filters but only passes Graph.NodeFilter/EdgeFilter. Contract should explicitly reject unsupported generic filter fields or define mapping; don't silently reinterpret node-vs-edge domain semantics. Graph module owner should assess together with managed graph query behavior. No scope bypass claim because this backend denies scoped reads.

D8. Neo4j Traverse normalizes full returned snapshot, while Retrieve projects only document fields (`neo4j.go:106-125`, `:153`). Decide whether invalid labels/edge endpoints/metadata from Runner must be rejected during retrieval or intentionally excluded from the projection contract; documenting only schema-validated traversal is insufficient if consumers assume full returned graph validation.

D9. pgvector Upsert builds one SQL statement with four parameters per record. Host must bound batch size (Postgres parameter/statement capacities); no implicit chunking that turns atomic single-statement semantics into partial commits. Expose/document explicit capacity or host batching responsibility and errors.

D10. pgvector search orders only distance (`store.go:182`) and lacks tie-break ID. Define whether deterministic equal-score ordering is required, especially evaluation reproducibility/caches. Add stable tie-break if supported without undermining ANN index planning; no ANN guarantees currently claimed.

D11. RawStore FindByIDs/Delete methods deliberately bypass read Binding and use explicit raw administration contract. Keep them clearly named/documented. Do not retrofit service IAM into every raw storage operation or misreport this as boundary violation.

D12. Returned affected counts from Qdrant bridge trusted; transport driver must implement exact vs unknown counts honestly (especially async deletes). Current int API leaves unknown unrepresentable. Decide known/unknown DeleteResult count contract if real drivers cannot provide exact count; don't synthesize count from request length.

D13. pgvector Rows.Close errors ignored; query/decode partial handling otherwise explicit. Decide if close error can contain a meaningful host-driver query outcome and whether to expose it; no blanket defer-error machinery without real driver requirement.

D14. All storage bridges lack native live transport, retries, migrations and remote profile inspection. These are documented host responsibilities and appropriate boundaries. Keep no hidden fallback/backoff; do not add automatic remote DDL/schema mutation, shadow collections or re-embedding to adapter constructors.

D15. Capability claims for pgvector/Qdrant/ES rely on host bridge actually enforcing supplied predicate before loading payload. Local SQL/DSL/condition tests cannot prove bridge/server enforcement. Keep integration certification separate, including tenant pairs, >2^53 int64 neighbors, omission metadata cases and unsupported pinned requests. Existing docs/task19/capabilities.md distinguishes wire/local/live correctly.

D16. Public filter aliases expose constraints with underlying scalar (~) types but constructors produce only built-in typed fields, while internal scalarKind/toValue handle exact types. Avoid speculative generic alias support until public constructor permits it; simplify constraints or ensure alias-normalization deliberately if that API expands.

D17. Public builders return bare errors for nil/unfinalized states (`filter/builder.go:15`, `:38`) while schema/adapters use ErrInvalidArgument. Standardize sentinel classification where callers need it; naming/building no schema provenance token is fine because final schema validation checks names+kinds.

## Positive findings / keep

- Filter IR sealed, schema immutable registry, typed builder validates kinds; SQL values are parameters, identifiers constrained.
- Scope preparation intersects mandatory filters; core delivery gate protects pgvector/Qdrant/ES; unsupported pinned profiles explicitly rejected.
- pgvector exact integer wire uses RawAttributes.UseNumber; JSONCodec handles canonicalization. Qdrant/ES normalize schema attributes before typed decode.
- Vector Space identity and metric checks occur before I/O; pgvector only cosine, Qdrant supported profile explicit. Don't claim declaration certifies remote collection/table data.
- Structured backend/projection errors, no built-in retry or policy engine. Separate modules keep driver-like responsibilities out of core.
- Sparse documented scope profile Eq/In/And is sensible; don't promise full backend-query equivalence without missing-field conformance.

## Validation

Two targeted diagnostic overlay commands exited 0, no root/full suite or live services run by this agent. Probe tests log the current incorrect behavior; their PASS is not remediation verification. Production git status clean at report time. Capture logs can be regenerated by parent using stated commands; both /tmp tests and overlay files retained.
