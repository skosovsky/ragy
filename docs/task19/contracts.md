# TASK19 final conformance and retrieval evaluation contract

Baseline implementation: `13c37b67689be736bb05d575b41f2446d2b90c50`. Original TASK19 and common execution conditions remain mandatory. Correctness, performance, quality, transport fixtures and live-service verification are separate evidence classes. Library scope remains retrieval/context/index lifecycle; IAM, prompts, models, tokenizers, prices, agents and retention decisions remain host-owned.

## Frozen evaluation protocol before execution

Use a versioned, explicitly described consumer-domain corpus with independently authored development and holdout query/qrels splits. Corpus rows retain original source/revision/support identities, scope and conflicting/stale facts; qrels use graded source/document relevance. Dataset author does not tune strategies. Strategy implementation may inspect development data, but must freeze its configuration/code digest before running or inspecting holdout labels. Preserve both authoring and execution provenance; failure/unknown outcomes remain visible.

Compare baseline BM25, hybrid/RRF plus an explicit host reranker, single-rewrite/multi-query/decomposition text recipes, admitted graph expansion and candidate-only tensor MaxSim against the same eligible corpus and resource policy. Reuse the existing recipe_comparison, graph_comparison and tensor_comparison execution/scoring paths; shared consumer helpers are allowed, a second general evaluation framework is not. Concrete domains and model simulators remain in examples/docs. Deterministic local/scripted model/embedding profiles must be named explicitly: they measure consumer/library behavior, not LLM/provider effectiveness. Live execution is opt-in, never inferred from transport fixtures.

Predeclared hypothesis for promoting a more complex profile: holdout absolute Recall@K gain >=0.05 over baseline, no MRR/nDCG degradation, no additional no-answer or execution failures, zero scope/source/citation correctness violations, and fully confirmed equal resource-policy compliance for both profiles. A deterministic local experiment alone does not qualify a new universal library default. Failed or unknown accounting cannot satisfy compliance. Retain baseline if qualification is absent; report losses without changing qrels or thresholds. No final-answer judge or agent prompt-injection immunity is claimed.

Policy must declare finite retrieval/model-call slots, full formatted-context bytes/tokenizer units, candidate universe and deadlines. Record retrieved/delivered IDs and original contributors, source citation validity, Recall/MRR/nDCG where applicable, abstention/no-answer errors, actual calls, known/unknown token and cost accounting, elapsed observations and failures. Separate metric denominators for answerable/no-answer cases; never drop failed rows from their applicable denominator. Zero model calls and unavailable billed usage are different states. No p95/p99 from isolated samples. Deterministic repeats assess repeatability only; stochastic provider results require repeated observations and honest uncertainty.

| ID | Mandatory requirement |
|---|---|
| E01 | One documented runner covers all 14 modules with passed/failed/skipped, actual cache/race/live modes, versions and raw outcomes; real PDF engine and unavailable live services distinguished |
| E02 | Public contract helpers and actual external GOWORK=off BYOT composition verify decorators/fusion/space/provenance/packing/coverage/shared budgets/evidence privacy v2/lifecycle pins and maintenance |
| E03 | Versioned reproducible corpus, independently authored dev/holdout qrels and provenance cover lookup, multihop/decomposition, no-answer, harmful rewrite, duplicates, conflicting/stale facts, limited scopes, multilingual and malicious retrieved instructions |
| E04 | Actual baseline/hybrid/rerank/text/graph/tensor comparisons reuse existing consumers with identical eligible corpus/resource policy and explicit candidate-only limitations |
| E05 | Quality, delivered/citation/abstention and resource metrics preserve all relevant denominators, failures, unknown usage and raw elapsed results; no invented billing or final-answer quality |
| E06 | Corpus/split/config/prompt/model/provider/seed/tokenizer/usage/enforcement identities and code/data digests make execution reproducible; local simulator and live-provider evidence separated |
| E07 | Hypotheses and thresholds precede execution, strategy freeze precedes holdout use, repeats/uncertainty match actual deterministic/stochastic execution; no post-result tuning |
| E08 | Official provider wire fixtures are checked independently of client implementation; opt-in live checks and actual external DB requirements never replaced by mocks |
| E09 | Adversarial actual composition proves scope, original provenance, revocation/retirement, budgets and evidence privacy; retrieved instructions remain data and external-agent injection success is separate |
| E10 | README/capability matrix match current code and distinguish implemented versus real-service-verified; dated historical evidence retained without readiness-percentage claims |
| E11 | Meaningful AAA/race/consumer checks and all-module final run pass; experiment/domain/OTel/provider dependencies do not leak into universal core |
| E12 | Default promotion follows predeclared confirmed qualification; local fixture gains/unknown compliance cannot change universal default, and losses/limitations remain reported |

Live credentials, service provisioning and third-party billing are not prerequisites to verifying the explicitly labeled local profile. Unsupported or unexecuted real-service profiles must stay unavailable/skipped. This does not permit skipping any required implemented local contract or substituting fabricated observations.
