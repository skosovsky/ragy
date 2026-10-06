# T01 — independent completeness acceptance

Verdict after independent recheck: **ACCEPTED**: explicit criteria 6/6 (100%), requested source coverage 16/16 (100%). No substantive files changed by this reviewer; independent correctness report was not read.

Baseline HEAD: `517179b87209a1fb2a41c47057f766dfdce3f959`. Historical review SHA: `b63d5e19a52c7d4e621b1a85a3ce428a92acbcb7`.

| Criterion | Status | Evidence in docs/contracts/remediation.md |
|---|---|---|
| T01.C01 | выполнено | Lines 7–26: independent callback/settlement/gate causes, origin-aware deadline, joined ProtectionError always suppressive, ordinary/protocol and usage overrun fail, parent/authority checks independent; RunObserved failed journal owned and non-protection only. |
| T01.C02 | выполнено | Lines 36–44: min remaining intervals with real parent deadline and independent ledger/local clocks, equality expiry, pre/post callback and clone/project/delivery gates; exact-once settlement including reservation-to-dispatch cancellation and late accounting. |
| T01.C03 | выполнено | Lines 48–63: schema-normalized UTF-8/bool/exact int64/finite float domains; nil/empty whole attributes all absent vs present null/wrong kind; Eq/Neq/In and numeric order table, AND/OR/NOT two-valued; trusted normalized lookup precondition, atom-level PG normalization, real query/deletion parity required. |
| T01.C04 | выполнено | Lines 67–73: valid UTF-8 namespace/key/name/config/policy/relation/history, optional extraction namespace/Ambiguous and optional-state distinction, unchanged valid tuple JSON+SHA256 framing, U+FFFD distinct from rejected invalid bytes; input/config InvalidArgument vs host response Protocol. |
| T01.C05 | выполнено | Lines 77–85: unique exact normalized page reference before callbacks, allowed cell/image selectors and cross-page retained artifact sharing, authorized-input prerequisite, borrowed/owned distinction, per-callback and final public success/empty/partial read gates. |
| T01.C06 | выполнено | Lines 89–119: isolated reviewed SHA source, fixed module/manifest/ref scope, dirty/index rejection and preservation of caller files, durable same-candidate retry, exact remote none/complete/partial/unknown/collision states, atomic publication and v2 migration prerequisite. |

## Source coverage

| Source item | Coverage | Contract evidence |
|---|---|---|
| F01 | complete | Error precedence, failed journal and joined protection/deadline provenance; original planner/backend/assessor/encoder matrix remains implementing-task responsibility. |
| F02 | complete | Release module/file/ref allowlist and excluded unrelated files/tags. |
| F03 | complete | Caller preservation, same-candidate retry and none/partial/unknown recovery. |
| F04 | complete | Own-clock ledger/local composition, post-model/project/clone/delivery expiry and accounting. |
| F05 | complete | UTF-8 identity domain and stable valid IDs/history. |
| F06 | complete | Exact normalized page uniqueness and explicit allowed media/cell sharing. |
| F07 | complete | Pre/post callback gates, cache hit/miss, no next callback after cancel/revoke and borrowed raw metadata. |
| F08 | complete | Scalar truth table, NOT/AND/OR normalization and real PG query/delete parity. |
| F09 | complete | Final Neo4j success/empty/partial delivery gates and single traversal. |
| F10 | complete | Transport/header/body cancellation before sanitized classification, zero output/unknown usage, body close and one dispatch. |
| F11 | complete after recheck | Provider endpoint admission explicitly rejects RawQuery/ForceQuery/fragment/credentials/opaque/non-HTTP(S)/missing host before dispatch with ErrInvalidArgument; parsed path construction preserves base path and specifies exact root/v1 endpoint with zero/one trailing slash. |
| D10 | complete for T01 scope | Shared ledger, calls not refunded, known accounting and unknown reservation. Further cost/token separation dispositions remain later task responsibility. |
| D28 | complete | MaxInputBytes source text only; CountInputTokens complete actual envelope. |
| D32 | complete | Preserved simultaneous causes and extraction/graph fail-zero policy. |
| D51 | complete for T01 scope | Missing/null/wrong-kind and trusted normalized lookup domain; codec input constructor details remain later task responsibility. |
| D61 | complete for T01 scope | Durable public contract and publishable modules/GOWORK=off candidate/consumer gates; broader docs/CI surface remains later task responsibility. |

## First finding and independent recheck

Initial review of contract SHA256 `71c2d4f7f01454e78766c548f956823ef45d7d6a32f46e1af3847dfcf9427f2d` found C01–C06 complete but source coverage 15/16 (93.75%): baseline F11/providers P-02 parsed-URL admission and path policy were absent. This finding was sent to the implementer, without changing the contract.

Recheck reviewed the new Provider endpoint admission section at lines 28–32, against baseline F11 and provider P-02 AAA. The section now covers every required invalid case, ErrInvalidArgument and pre-network rejection; root/custom base path and trailing separator behavior are explicit. Intermediate URL-only recheck contract SHA256 was `3bc7bdd4f8280f4b15d8d31be0c41932c813b412f2de787455c0bc25e53f66cf`. Earlier six criteria remain covered in the current document. Final independent full-document recheck reviewed nil/empty whole RawAttributes as all absent vs present null/nil rejected, and optional extraction namespace/Ambiguous with no guessing. These preserve explicit domain/admission boundaries; all six criteria and sixteen source items remain complete. Final checked contract SHA256: `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df`. No outstanding completeness findings. This is contract-first only: no implementation defect is claimed fixed. No implementation tests run: documentation review compares target rules against preserved original F matrices and criteria; running historical code would not prove contract completeness.

## SHA256 of checked substantive files

| File | SHA256 |
|---|---|
| docs/contracts/remediation.md | `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df` |
| docs/task20/backlog.json | `ec2644a2c980862b642af3de45a98d887bb858c65063c49160b99d4a40e4293d` |
| docs/task20/review-baseline.md | `66dba45fc74dbdd8c0d1251db5ff4f92f8272d76b93a9a1e9d2c70baa47b3cc6` |
| docs/task20/reviews/retrieval.md | `ae336aef859bc9b1e23ef1f10fd96bbde14a2693c73029e409be4547a6695c59` |
| docs/task20/reviews/graph.md | `e15e682cf22954635fb4ea883a552ff7d8d2663530d354e1ca16bbfc85e2d780` |
| docs/task20/reviews/ingestion.md | `904abaeb28003225342feb5e5735faf8b4294c15b0bd25e5d03b1ab8d0415985` |
| docs/task20/reviews/storage.md | `8ca61bd22fe05816310d3baeffcbb6a34cf57ce2e32cff625343a08fe84ac0e7` |
| docs/task20/reviews/providers.md | `e71c77948b649d9d63ed6abfbc3f30bedf7c5833d650a9c8f209cbf773a1553e` |
| docs/task20/reviews/arch-docs.md | `20ea51ebacdd2d16bf93aec1b92f5170690210e727f22c73a6b6e88b33bac6ba` |
