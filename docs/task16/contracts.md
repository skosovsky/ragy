# TASK-16 bounded delivery contract

Baseline: `becf3cd`. Status: implementation in progress. Original task scope is mandatory.

## Decisions before implementation

- Artifact rendering replaces rune-only snippet Budget with an explicit resource policy: limit, unit/profile identity, context-aware measurement of the complete final serialized text, maximum measurement count and final output bytes. A named rune policy is a stdlib convenience; tokenization remains host-owned. Envelope is measured first. Measurement/error/oversized/mandatory-envelope failures expose explicit typed outcomes without snippets.
- Packing conservatively keeps whole projected snippets, rather than assuming tokenizer additivity or formatter monotonicity. Every trial measures complete formatted output; the returned string is the exact accepted measured string, never reformatted or sliced afterward. Custom formatting must declare where the unchanged snippet content appears, and the library verifies that span before claiming delivered provenance. Labels/boundary/separators count. Projected partial/derived text cannot assert that all source-document facts were delivered.
- Text recipes explicitly accept a caller ledger, with a separately named own-ledger convenience. Text/graph callers can share one ledger. All dispatches reserve before invocation; calls are never refunded; unknown accounting retains reservation. Context deadline is minimum of parent, recipe and shared ledger deadline.
- Text recipe config declares whether backend is model-free; generic Backend is not proof. Optional query encoding is an explicit budget-aware capability, independently quoted/reserved as a model operation; each text variant receives its own embedding and profile. Supplied TASK-15 adapters cannot enforce strict remote token caps, so strict integration must reject unsupported before dispatch. Host capable encoding ports may enforce the reserved limits. Original precomputed vector remains unsupported for rewrites.
- Admission, retrieved, selected and delivered evidence remain distinguishable. Mechanical per-query coverage is recomputed from retained contributors after TopK and optional artifact packing. Missing/uncertain delivery cannot produce Complete solely because the assessor selected original queries. Host semantic sufficiency remains a signal, never a truth guarantee. Model-free runs need no tokenizer when artifact rendering is omitted.
- Graph summaries accept explicit host limits for communities, snippets, bytes and calls. Global accepts 1..N communities and performs one map per community plus one reduce; no discovery/recursion. Original supports remain freshly admitted; reducer bounds apply before its dispatch.

## Mandatory acceptance matrix

| ID | Requirement |
|---|---|
| B01 | Whole final artifact resource/usage/profile/packing status explicit; former ambiguous Budget removed |
| B02 | Boundary/labels/separators/custom formatting counted; Unicode and non-additive/non-monotonic measurement verified |
| B03 | Finite iterations/measurements/final bytes; envelope, formatter, measurement, cancellation/revocation failures suppress snippets |
| B04 | Custom rendered positions and source mappings/provenance verified; no silent serialized slicing or invented precision |
| B05 | Admission/retrieved/selected/delivered evidence distinguished; contributors survive fusion and packing |
| B06 | Two independent subquestions with TopK=1 cannot claim full delivery; partial/unattributable text remains uncertain |
| B07 | Shared explicit text/graph ledger, separately explicit own-ledger run; minimum effective deadline |
| B08 | Reserve each actual retrieval/model/encoding dispatch once; failed/cancelled/unknown usage cannot refund calls or fabricate usage |
| B09 | Concurrent text/graph callers competing for last slot cause exactly one dispatch |
| B10 | Each rewrite encoded independently, identity/accounting retained; strict unsupported encoding rejects before dispatch |
| B11 | Backend model-free contract explicit; generic backend does not attest hidden model behavior; no model required for plain pipelines |
| B12 | Summary supports 1/2/N within explicit limits; N+1/reducer overflow reject before prohibited dispatch |
| B13 | Consumers/records/schemas/examples/docs updated, clear break, all affected core/nested checks pass |

No conversation context quota, organization billing, recursive research, retry engine, agent loop or final answer generation is introduced. Untrusted boundary is a consumer signal, not prompt-injection immunity.
