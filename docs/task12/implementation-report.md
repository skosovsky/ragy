# Task12 implementation and acceptance report

All seven issue #3 cards and BUG-001/BUG-002 are implemented in the local working tree. Final independent completeness is **195/195 =100%**, with unchanged scope and no mandatory gaps. The correctness audit confirms no unresolved confirmed defect in the reviewed area. Final independent verdicts are in [completeness audit](audits/completeness-cli-profile.md) and [correctness audit](audits/correctness-cli-profile.md). The matrix maps every requirement to implementation and verification in [requirements.md](requirements.md). No release, publication or issue closure has been performed.

## Delivery and boundaries

- RAG-001: typed lifecycle inventory, publication, pinned reads, tombstone barriers, bounded cleanup and durable reconciliation; actual dense/lexical/tensor/graph and joint profiles.
- RAG-002: opt-in single rewrite, multi-query and nonrecursive decomposition; budget/deadline/call controls, actual source-linked evidence and explicit partial/insufficient outcomes.
- RAG-003: typed tensor space/capabilities, native MaxSim, bounded persistent dense candidates→tensor rerank, deterministic oracle and actual saved comparison.
- RAG-004: BYOT extraction/resolution/identity history, source-supported graph deletion, local expansion and bounded community/global summaries with original support provenance.
- RAG-005: strict immutable evidence export, stage/rank/score/revision associations, redaction and disabled/best-effort/required recording; actual tensor/PDF/graph records.
- RAG-006: mandatory scope/publication and freshness gates throughout leaf, composition, planner, fallback/rescue, hydration and owned caches; public passing/rejecting custom-adapter conformance.
- RAG-007: original/derived text/page/table/cell/image locators, actual PDF parsing/projection/resolution, OCR/layout partial coverage and source retention boundaries.

Core remains universal and independent. Host metadata/types, authority, original blobs, model transport, tokenizer/pricing policies and consumer summary assessment are injected. Optional engines and Codex CLI exist in adapters/external consumer examples. No CLI/auth/model dependency, common experiment framework or mandatory tokenizer executable was added to core. Persistent dense/tensor reference backends have actual filesystem integration; raw external database/service adapters do not claim live service certification. [Capabilities](capabilities.md) records these limits.

Replaced contracts use a clear break. All12 author/host migration steps are in [migration.md](migration.md): explicit read/publication bindings, BYOT codecs/schemas, processor signatures, score semantics, lifecycle inventories/retention, original mappings, bounded model/budget ports and evidence policies. Remove replaced legacy call paths; no compatibility shim is promised.

## Verification and fixes

Final all-module `make lint` and `make test` after all source corrections both terminated exit0 (owner sessions63973/25997 closed); outcomes are recorded in [lint](results/cli-final-lint.txt) and [test](results/cli-final-test.txt). Tests include race detection, examples, actual persistent producers and the configured PDF runtime. Five independent schema suites and adversarial associations are part of acceptance, not interface-only claims. [Verification record](verification.md) preserves current and explicitly marked historical checkpoints.

BUG-001 retains exact integers through JSON normalization; BUG-002 stages and atomically publishes BM25 updates. Subsequent independently reproduced findings were corrected: metadata ownership/freshness, finite BM25 configuration/results, exact cleanup support inventory, tensor recorder associations, failed CLI trace retention and ambiguous duplicate JSON fields. CLI corrections preserve complete stdout/stderr/events/all returned settlements, make unavailable aggregate usage sticky after malformed/overflowing settlement, and reject nested/escaped/Unicode-fold-equivalent duplicates. Independent rechecks reproduce the former failures and verify their corrections. No unresolved confirmed defect may be counted as accepted. The correctness audit states its review limits; absence of findings is not proof of absence of errors.

## Actual comparative experiments

Issue [clarification](https://github.com/skosovsky/ragy/issues/3#issuecomment-6011559475) and task12§7.1 separate unchanged reference budget conformance from calibrated CLI live quality. Calibration precedes each frozen host profile. All model inputs exclude qrels/expected answers. Actual library recipes/retrieval/resolution/validation/evidence execute; no scripted response is credited as live.

The corrected-retention text run completed20rows/30actual model invocations: [capture](results/text-cli-v2-capture.json), [report](results/text-cli-v2-report.json), [profile](results/text-cli-v2-capture.json.profile.json). All profiles have0 no-answer errors and0 failed executions. Full reported usage is161085input/1216output tokens, including executor context; raw timings are preserved.

| Text profile | Recall@3 | MRR@3 |
|---|---:|---:|
| Baseline | 0.5 | 0.5 |
| Single rewrite | 0.75 | 0.625 |
| Multi-query | 0.75 | 0.625 |
| Decomposition | 1.0 | 0.875 |

The corrected-retention graph run completed4validated/materialized extractions plus6baseline/recipe query rows: [capture](results/graph-cli-v2-capture.json), [report](results/graph-cli-v2-report.json), [profile](results/graph-cli-v2-capture.json.profile.json). Preparation22320input/753output tokens and48.682585834seconds summed attempts are separate from query21598input/221output tokens. All8receipts retain full stdout/stderr and returned settlements.

| Graph case | Baseline support Recall@3 | Recipe support Recall@3 |
|---|---:|---:|
| Local | 1 | 1 |
| Community | 1 | 0 |
| Global | 1 | 2/3 |

Community model abstention with empty support selection is rejected by validation and honestly recorded as failed. Global covers both configured prod-community dependencies with original s1/s4 supports, omitting the redundant s2 qrel. [External consumer review](results/graph-cli-summary-review.md) compares every actual checkable claim to original sources, records omissions, namespace ambiguity, unknowns and abstention, and separately reviews the new v2 outputs. Negative quality is a result, not an invariant waiver. Default remains baseline.

Earlier complete runs are preserved separately as `text-cli-live-*` and `graph-cli-live-*`; results are not merged or retrospectively substituted. V2 was compiled after trace/duplicate fixes but before later malformed/overflow and Unicode-alias guard corrections. Its valid actual traces are independently replayed through the final decoder; separate adversarial tests validate those error-path corrections. We do not assign the final source identity to the earlier compiled executable.

CLI hard input/output bounds and monetary pricing remain unverified/unavailable. Tokens are advisory; zero unpriced ledger units are not zero billing. Supported tools are disabled and observed tool activity rejects responses, but extra context and internal provider dispatch/retries cannot be proven absent. Full returned usage is retained without subtraction. A subprocess deadline does not prove cancellation of remote generation. These are small fixed-fixture consumer comparisons, not isolated retrieval quality, production SLA, meaningful percentiles or statistical significance. Reference hard-budget/call/depth/scope acceptance is tested separately. No additional live run at original token/time profiles is required by the clarified issue.

## Closure and release

After final independent acceptance, this report, the195-atom matrix, capability limits and author migration instructions provide the issue closure material. Issue #3 is closed only on a separate user command, with these implementation/evidence/migration references. The substantial clear break requires `make release-break`; release additionally requires clean `make lint`/`make test` and explicit approval. `make release-patch` is reserved for insignificant changes. No release or issue-state operation is part of this local completion.
