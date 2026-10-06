# Retrieval remediation contracts

Contract revision: 2026-10-06. This document specifies the target behavior for task20. The baseline implementation at `b63d5e19` does not yet satisfy it; task acceptance is tracked in [the sequential plan](../task20/plan.md). Each implementation task must synchronize its package GoDoc, tests and examples with these rules. This contract is authoritative for the changes below; it does not certify live services or arbitrary host callbacks.

## Error precedence and delivery (F01, F07, F09, F10, D32)

An operation retains independently observed callback, settlement and freshness/deadline causes. Joining errors preserves `errors.Is`/`errors.As`; membership in `context.DeadlineExceeded` alone never authorizes partial success. Outcome classification uses the origin of a failure as well as its class.

| Observed condition | Delivery | Error / recovery |
|---|---|---|
| Parent context canceled/expired, authority revoked/expired, publication unavailable, or callback returns a protection failure | Zero payload, zero payload-bearing journal; no subsequent payload callbacks | Suppressive protected failure preserving every already observed relevant cause; no rescue/retry |
| Callback returns protocol failure, ordinary non-deadline failure, or independent protected failure while local time expires | Failed result; ordinary Run/RunOwn return no payload; RunObserved can retain only its documented failed journal | Callback cause and timing/settlement causes remain discoverable; not bounded success |
| Actual usage exceeds reservation, including simultaneous deadline | Failed result under the same journal rule | `budget.ErrUsageExceeded` retained; no retry/refund |
| Pure attempt-local deadline, parent and read authority still valid, no independent callback/settlement fault | Core text recipe may deliver previously captured evidence as its documented bounded partial outcome | `DeadlineReached`; extraction and graph recipes keep their documented zero-output deadline policy |
| Reservation exhausted or required price unknown before dispatch, no higher-priority cause | Core text recipe's documented bounded partial outcome | `BudgetExhausted` / `PriceUnavailable`; denied dispatch never occurs |
| Ordinary adapter projection/transport failure, valid read binding | Only the adapter's explicit partial contract may return retained input/prefix with error | Caller must inspect error; retained input/native score is not a successful rerank |

A deadline from a callback counts as pure local expiry only when it is solely the supplied operation context's deadline, the local/ledger limit actually expired, and parent/authority checks still pass. A callback `ProtectionError` is always independent and suppressive, including `errors.Join(protection, context.DeadlineExceeded)`. A joined ordinary callback error remains failure. An internal read gate seeing the child timer expire must distinguish that timer from a parent or authority failure; it must not manufacture a rescuable classification for a host protection error. If provenance cannot be established, fail closed.

Read/parent authority is checked independently before considering bounded partial delivery. Settlement records usage once even after a failed callback or expired deadline. A successfully reserved call is never refunded. Unknown usage retains its reservation; known usage overrun does not become success because its callback timed out. No hidden dispatch retry or late payload delivery is permitted.

`RunObserved`'s failed journal is an owned record of admitted observations, not selected successful evidence: outcome Failure, stop StageFailure, selected payload empty. It may exist only for non-protection failures while the parent read remains valid. Protection suppresses the entire returned result/journal. `Run` and `RunOwn` expose no failed result. Payload-free observation events remain subject to their separate bounded privacy contract; no content is copied into error text or diagnostics.

Final public delivery gates apply to every success, empty and partial branch. If a freshness gate and an ordinary callback fail together, freshness suppresses results while both causes stay inspectable. Sanitization removes raw host/provider/filesystem text; it does not erase cancellation/protocol/usage classification. Custom error Is/Unwrap and callbacks are cooperative host code, not sandboxed work.

Structured HTTP cancellation/deadline is checked at transport, headers and body boundaries before ordinary sanitized I/O/protocol classification. Incomplete response bytes imply zero output and unknown usage, one dispatch, closed body. Truncated JSON is never parsed to invent usage.

## Provider endpoint admission (F11, D55)

Structured and shared provider transports validate a parsed absolute HTTP(S) BaseURL before dispatch. Reject nonempty RawQuery, ForceQuery (including a trailing bare `?`), fragment, user credentials, opaque URL, missing host and unsupported scheme as `ragy.ErrInvalidArgument`. An invalid constructor/configuration never reaches the network. Host network policy remains separate from structural URL admission.

Construct the endpoint from the validated URL path, preserving an allowed custom base path and a single separator before the endpoint. Do not concatenate endpoint text to the raw URL string. For BaseURL `https://provider.example/v1` or the same URL with one trailing slash, the structured request pathname is exactly `/v1/chat/completions`, with no query or fragment. A root base addresses `/chat/completions`. No URL normalization changes host, credentials or supported scheme, and no hidden redirect/retry is introduced. Shared internal helpers or a common contract suite must cover the same invalid and valid cases.

## Clocks and reservations (F04, D10, D28, D32)

Local duration starts at operation entry using the configured clock. Its absolute deadline and the ledger deadline are immutable for that attempt. Parent context supplies its own real deadline. Cooperative context time is bounded by the earliest remaining parent, ledger and local duration; clocks are not assumed to share wall-clock epoch.

For a clock-driven deadline `D` and that scope's injected `Now`, remaining time at context creation is `max(0, D - Now())`. The resulting real timer deadline is `time.Now() + remaining`; compose this with the parent's real deadline. With wall-clock-aligned clocks this is the literal minimum absolute deadline. With deterministic clocks it is the minimum remaining interval, not an arbitrary fake timestamp used with `context.WithDeadline`. Each clock-driven scope also checks `Now() < D` at every admission/callback/project/clone/delivery boundary. Equality is expiry. A wall-clock timer alone cannot observe an injected clock leap.

If a ledger is shared, its own clock checks its deadline; the adapter clock checks the local deadline. Using one clock to evaluate the other scope's timestamp is invalid unless their shared time domain is part of configuration. Parent cancellation/deadline always suppresses delivery, irrespective of injected time. Backward-moving clocks cannot extend the already installed real context timer; the host supplies stable concurrency-safe clocks.

Every successfully acquired lease is settled exactly once, including cancellation between reservation and dispatch. The settlement operation is accounting, never renewed admission. Known actual usage is recorded even after expiry; unknown usage remains conservative. Failed reservation creates no lease/call. No dispatch may begin after a post-reservation gate failure. Cancellation during a synchronous callback is cooperative: wait for return, settle, reject its output, and never start the next callback.

Extraction context composes both the supplied ledger and its local Duration. The model, output validation, cloning and projection all observe the same bounded child context and clock gates. MaxInputBytes measures source text only; ontology/configuration/JSON framing are not included in that count. CountInputTokens must account for the actual complete provider envelope. Any separate envelope byte cap must have an explicit field/unit, not silently change MaxInputBytes semantics.

## Portable filter truth table (F08, D51)

A validated schema admits optional absent fields. Canonical stored scalar domains are UTF-8 string, bool, exact int64 and finite float64. Conditions and attribute values are schema-validated before backend execution. Numeric normalization is explicit at the schema/codec boundary; int64 comparisons never round through float64. Null is not a supported stored scalar. A nil or empty whole RawAttributes map is admitted as all fields absent and normalizes to an empty attribute set. A present field with a nil/null value, nonfinite value or unnormalizable/wrong-kind value is rejected by schema admission; field absence is represented by omitting that key. This preserves the optional-field profile and makes empty codec input policy explicit.

`MatchCondition`/`MatchIR` low-level lookup is a trusted validated boundary: callers supply normalized present values or `(nil, false)` for absence. Behavior of arbitrary wrong-kind/NaN lookup values is not a portable backend guarantee; adapter/store admission must reject them before query/payload work. Conformance distinguishes absent from malformed metadata instead of treating malformed as absent.

| Leaf predicate | Field absent | Present equal x | Present unequal x |
|---|---:|---:|---:|
| Eq(x) | false | true | false |
| Neq(x) | true | false | true |
| In(values) | false | true if member | true if member, otherwise false |
| Gt/Gte/Lt/Lte(x), numeric fields | false | Gt/Lt false; Gte/Lte true | ordinary exact scalar comparison |

Eq/Neq/In support all four scalar kinds; ordered predicates support int/float only. Ordering of bool/string is rejected by condition/schema validation, not silently approximated. Empty condition is true. AND and OR combine two-valued children; NOT negates a two-valued child. Neq is the negation of normalized Eq, not SQL's nullable `<>`. Example: absent f gives `NOT(Eq(f,x)) = true`, `NOT(In(f,[x,y])) = true`, `NOT(AND(Eq(f,x), Eq(g,y))) = true` if f is absent regardless of g. No SQL NULL/unknown propagates through these predicates.

PG normalizes every positive atomic predicate to a non-null boolean before NOT/AND/OR. An outer WHERE coalesce alone is insufficient. Values stay bound parameters and identifiers follow the adapter's explicit validated policy. Query and DeleteByFilter share exactly the same rendering semantics. Malformed remote attributes are outside the admitted stored corpus; a bridge must preserve schema validity, not infer that wrong-kind JSON is authorized omission.

Acceptance compares core matcher IDs with query and actual deletion IDs on an isolated PostgreSQL corpus across all supported scalar kinds, nested groups, omissions and exact int64 neighbors above 2^53. SQL rendering tests are additional evidence, not real-service parity. Portable fixtures include separate rejection cases for malformed/null values; they do not run malformed records as valid backend rows.

## Identity and Unicode (F05)

Required resolution identities are nonempty valid UTF-8 strings. This applies to extraction entity IDs/names, relation IDs/endpoints, ontology/policy/configuration identifiers, resolved decision namespace/key/canonical name, and relation keys. Extraction entity namespace may be absent: the host identity policy can return Ambiguous and core must not guess a namespace. Optional identity fields are allowed empty only where the contract explicitly permits absence (e.g. extraction namespace, ambiguous decision, history Parent). Every nonempty optional field must still be valid UTF-8. Whitespace/case/Unicode normalization, aliases and semantic identity are host policy; ragy never silently repairs invalid bytes or canonicalizes keys.

Constructor/configuration and direct malformed input failures are `ragy.ErrInvalidArgument`; malformed identity decisions/keys returned by host policy are `ragy.ErrProtocol`, with zero result. Structural input batch admission happens before identity/grouping callbacks. A returned decision is checked before its ID is hashed or grouped; already admitted earlier callbacks may have run, but no partial result is delivered and no later grouping/projection callback runs after rejection.

Canonical entity/relation IDs retain the existing JSON tuple + SHA256 framing for valid UTF-8 inputs. JSON tuple framing separates tuple components and valid identities retain their previous hashes. U+FFFD is itself valid and is never confused with rejected ff/fe byte strings. This change rejects invalid identities, so there is no silent hash migration for valid records. Persisted history may contain only domain-valid identity strings, checked before serialization; schema-faithful bounded BYOT marshal/unmarshal remains a host contract. Old malformed records cannot be trusted as restored original bytes after JSON repair: host migration/quarantine is required, not automatic reconstruction.

Same canonical namespace/key with different canonical names is a protocol conflict; per-mention labels remain in extraction. Equal valid decisions merge supports under the existing variant contract; distinct valid tuples remain distinct. Ontology, comparable kind stability, alias truth and policy recomputation remain host responsibilities.

## Source addressing and callback gates (F06, F07, F09)

A source.Reference names one immutable retained artifact/revision/representation. A page's normalized text has a unique exact Reference within a layout Document, independent of text length, content equality, physical order or coverage. Reusing a page text Reference is rejected during Document.Validate and before Project starts any metadata/ImageText callback. Adding physical page to a hash does not repair an ambiguous retained loader address.

Cells and image regions use complete Locators, which include reference, physical geometry and cell/region selector. Multiple distinct selectors may share a retained original representation when they address the same immutable artifact: e.g. two cells of one table or two image regions of one retained page. Duplicate logical cells/full image locators remain rejected by the existing page contract. Sharing never means one exact original locator resolves different source bytes/text; OCR/derived text stays explicitly derived from the original locator. Cross-page sharing is valid only for a host-retained whole artifact whose complete selectors independently resolve each location; it does not relax unique normalized page text References. Loader attestation is host-owned; geometry equality does not prove content authenticity.

Project receives an already authorized document. Binding freshness alone does not prove that arbitrary supplied bytes belong to its scope. Source Catalog→admitted Loader is the authoritative read path, with no latest-version fallback. RawStore stays explicit unscoped administration. Document-shape rejection returns zero projection and zero ImageText calls; existing ordinary partial policies are changed only in their assigned contract task.

Lexical callback protection retains callback identity/classification through `errors.Is` alongside gate causes, while `errors.As` exposes protection/gate errors rather than a payload-bearing callback sibling. Protected error text is fixed and excludes callback content. This scoped boundary preserves callback classification without changing the global `access.Protect` sanitization policy.

Before each payload-bearing codec/clone/project/model/loader callback, check context and read freshness; check again immediately after its return. On post-callback failure, suppress output and stop before the next callback. Preserve its error alongside gate causes. This applies to snapshot/managed cache hit and miss alike; cache identity cannot cache authorization. Raw mutable metadata remains borrowed under its existing host stability contract; owning snapshots use explicit cloners. Arbitrary Go callback interruption or speculative background workers are outside this contract.

Neo4j's public Retrieve applies the same final DeliverRead behavior as other storage bridges to success/empty/partial results, without repeating Traverse. Generic filter/domain capability decisions remain a separate T17 contract; unsupported scoped/pinned profiles stay explicit.

## Release scope and recovery (F02, F03, D61)

Release is prepared from an explicitly reviewed source commit, in an isolated checkout. The caller's HEAD, branch, index, tracked files and untracked files remain unchanged on every exit. Before mutation, reject dirty tracked files and staged changes. Untracked files are ignored and never copied/staged/deleted. Repository-local ignored state for release recovery is separate from caller payload files.

A version-controlled publishable-module manifest enumerates root and adapter modules, excluding example modules. Validate every module path and tracked go.mod before candidate creation. Caller-supplied module lists cannot expand this allowlist. Versioning preserves the current v0 patch/minor policy; a request requiring a >=v2 semantic import path change is rejected until a reviewed import-version migration exists. No automatic import rewrite is inferred from a version increment.

The exact source commit, candidate version, modules, expected tag names and permitted changed files are reviewable before publication. Only allowlisted module go.mod/go.sum files whose changes are required for coherent release dependency manifests can enter a candidate commit. No broad `git add .`, no publishing arbitrary local tags/branches. Tags point to the exact candidate commit. Root and submodule tags are pushed via exact refs; examples receive no tags. Existing refs pointing to a different candidate are collisions, never overwritten. Existing matching refs are idempotent evidence, not a reason to advance the version.

A persistent candidate record is written before publication and binds:

| Field | Required meaning |
|---|---|
| Version / release kind | Chosen candidate; retry uses this exact version |
| Reviewed source SHA / candidate SHA | Exact source and derived manifest commit |
| Publishable modules / changed-file allowlist | Fixed reviewed release scope |
| Intended refs and expected object IDs | Exact root/module tags, including tag-object IDs if annotated |
| Local created refs | Only refs owned by this candidate, distinct from preexisting user refs |
| Remote identity / observed refs | Publication destination and reconciliation evidence |
| Publication status / observation time | none, complete, partial or unknown; no inference from command exit alone |

Use atomic push for the complete intended ref transaction. Unsupported atomic capability is an explicit failure; no automatic non-atomic fallback. Inspection still handles partial preexisting/external publication and unknown transport outcomes. Remote observation compares exact objects (including peeled candidate commit where needed), not only tag names. Failed/unobservable inspection means unknown. Preexisting unrelated refs remain intact. Do not delete tags blindly to recover a version.

| State | Meaning | Recovery |
|---|---|---|
| none | Successful remote inspection proves none of the intended refs published | Retry the same persisted candidate after fixing the cause |
| complete | All intended refs match the candidate | Idempotent completion; do not republish/increment implicitly |
| partial | Successful inspection proves a proper subset matches; remaining expected refs absent | Reconcile the same candidate; push only the missing exact refs atomically after collision checks |
| unknown | Destination inspection unavailable or outcome not established | Explicit inspect before any further publication; preserve candidate |
| collision | Any intended ref exists with a different object | Stop; require explicit host resolution; never force/delete unrelated refs |

Candidate calculation cannot rely on an unpublished failed local tag as the latest published release. Any outstanding record must be inspected/resumed or explicitly resolved before a new candidate is created. A failure before creating tags still leaves sufficient record/state to preserve source/version or prove no candidate existed. Clean consumer GOWORK=off install/build is performed against the isolated release candidate/module graph, with exact manifests and submodule versions; it does not require real remote publication.

Task20 exercises publication only in disposable repositories with local bare remotes. The production repository receives implementation commits, not a release or push. Platform support, manifest editing and real release approval remain explicit in the release runbook. License selection belongs to the owner.

## Acceptance and compatibility

The contract-first commit does not close defects. T02–T11 implement and independently verify the relevant sections, including the full original AAA matrices. T12–T21 decide remaining design items and synchronize current public guides; T22 verifies the complete final state. Regression tests assert target behavior and exact call/settlement counts, not the historical diagnostic output. Clean breaks may remove APIs/legacy implementations after consumer inventory while preserving the guarantees above. Substantive contract revisions require explicit rationale, traceability and two independent acceptances of the implementing task.
