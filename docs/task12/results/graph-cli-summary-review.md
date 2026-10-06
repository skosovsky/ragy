# External consumer review of actual CLI graph summaries

Reviewed against `graph-cli-live-capture.json`, its saved per-call inputs/outputs, and `examples/conformance/graph_comparison/fixture.json`. This is a direct source comparison by the implementation reviewer, not model self-evaluation. Independent auditors must check the review. No statistical/general quality claim is made.

Original references below mean `originalReference(sN)` in the capture: namespace n, source sN, revision v1, original-p1/utf8, a-public. Canonical graph namespaces remain prod/staging and are independently resolved from source snippets. The consumer's configured summary communities C1/C2 use prod sources s1/s2/s4; staging s3 is separately materialized and is not selected into those communities.

| Actual call | Checkable claim | Source/input support | Verdict |
|---|---|---|---|
| Community query | “В предоставленных фрагментах C1 не упоминается.” | The two supplied texts are s1/s2; neither literally names C1. Host membership defines C1, but that ID-to-content association is absent from the model input. | Literally supported for supplied text; insufficient for the configured community question. |
| Community query | The model cannot determine C1's role/dependencies. | Output explicitly abstains, `selected=[]`; all input texts are retained. | Quality failure/omission. Core validation rejects the nonempty summary with no selected support. Captured outcome is failed; no published summary/supports are credited. |
| Global map, C1 | Billing (alias Pay) belongs to Team A and depends on LedgerDB. | s1 explicitly states all three facts. Output selects snippet0 → s1. s2 corroborates dependency. | Supported. |
| Global map, C1 | No Team B service/dependency information is present in the supplied snippets. | The map input contains only s1/s2. | Supported as a local-input absence statement, not a claim about the entire corpus. |
| Global map, C2 | Search belongs to Team B and depends on IndexDB. | s4 explicitly states both facts; selected snippet0 → s4. | Supported. |
| Global map, C2 | No Team A dependencies are provided in this fragment. | The map input is s4 alone. | Supported as a local-input limitation. |
| Global reduce | Team A: Billing (alias Pay) depends on LedgerDB. | Reduce selects map0, whose selected original support is s1. | Supported transitively by s1. Alias preserved. |
| Global reduce | Team B: Search depends on IndexDB. | Reduce selects map1, whose selected original support is s4. | Supported transitively by s4. |

Completeness and ambiguity checklist:

- Community question: expected Billing/Pay, Team A ownership and LedgerDB dependency are all omitted; no claim of successful community answer. The literal C1 label is missing from model inputs, so the model's abstention is understandable. This is retained as the fixed-run result; neither question/prompt nor limits were changed mid-run.
- Global question: both requested configured prod-community dependencies are covered, with original s1/s4 refs and preserved Billing/Pay alias. The duplicate supporting source s2 is omitted, which lowers support Recall@3 to2/3 even though both dependency facts are covered.
- Billing in staging belongs to Team B (s3); it remains a separate canonical entity from prod Billing/Team A. The summaries do not merge that identity or invent a conflicting owner. They are scoped to the configured prod communities, not all possible communities in the corpus.
- Local map absence statements explicitly refer to supplied snippets. The reduce combines the supported positives and does not turn local unknown information into a global absence assertion.
- No unsupported or contradicted positive factual claims were found in these actual summary outputs. This is a review of these outputs only.
- Community insufficient evidence produces an explicit model abstention followed by a captured validation failure. Its empty selection is not silently treated as a supported answer. A generic successful abstention path is not established by this case.

Negative quality outcomes are allowed for opt-in delivery under issue #3's clarification. The failed community case and negative support-recall gains are part of the report, not grounds for promotion. Default remains hybrid baseline. Reference token/deadline budgets are independently tested; this advisory CLI run does not qualify them. Extra executor context and internal provider retries/dispatches remain unverified.

## Separate corrected-retention run (v2)

`graph-cli-v2-capture.json` and `graph-cli-v2-report.json` were reviewed independently against all eight new actual receipts and the original fixture texts. The community response again explicitly states that C1 is not named in the input and abstains with an empty selection; it remains a captured validation failure. The first global map states Billing/Pay→LedgerDB for Team A and explicitly limits unknown Team B information to its supplied snippets. The second states Team B's Search→IndexDB and the local lack of Team A data. The reduce publishes exactly the two supported positive dependency facts, selecting both maps. Every checkable positive fact is supported by s1 or s4; the same omissions/ambiguity checklist and the original table's verdicts apply to these actual v2 outputs. No unsupported/contradicted positive claim was found. This statement comes from comparing the new saved output text and actual snippet indices, not assuming repeated model behavior.

All eight v2 receipts preserve full raw stdout, stderr diagnostics and returned settlements. No observed tool/reconnecting/retry/failure event marker appears in them; internal provider dispatches remain unverified, not declared absent. Preparation22320input/753output tokens and48.682585834seconds summed attempt time are reported separately from query21598input/221output. Support recall is unchanged:1→1 local,1→0 community,1→2/3 global.

The v2 run was compiled after full trace/duplicate-key fixes but before the later malformed/overflow sticky-state correction. Valid actual settlements in this run do not exercise that error path. Final independent revalidation with the corrected parser and adversarial sticky-state tests is required; no final-source identity is falsely assigned to the compiled earlier executable.
