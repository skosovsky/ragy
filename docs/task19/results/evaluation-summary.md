# Frozen holdout evaluation — 2026-10-06

All seven profiles completed actual local retrieval, source admission and context packing. Every profile had **zero execution failures, scope violations, stale-source violations and mechanical citation violations**, with confirmed compliance to the same finite local resource policy. All three deterministic repetitions reproduced the same candidate/ranked/delivered IDs. No complex profile met the predeclared numeric promotion criterion. **BM25 remains the default.**

These are deterministic host scripts, authored graph facts and SHA256 token vectors, not LLM/provider quality measurements. Live provider execution, billed tokens, prices and final-answer quality are unverified. No provider was called. Local operation/input/output accounting is known in explicitly declared serialized-byte units; it is neither provider-token telemetry nor a monetary bill. Unknown provider pricing is not recorded as free service usage.

## Independent data, freeze and commands

The holdout has 18 independent queries: 15 answerable and three no-answer cases. Three deterministic repetitions give denominators of 45 answerable executions and nine no-answer executions per profile; they do not create 54 independent experimental cases or stochastic confidence intervals. Corpus: 40 fictional operations documents, of which 36 current documents enter the host publication, with public and private scopes. Graded source/document qrels were independently authored.

- Source baseline revision: `13c37b67689be736bb05d575b41f2446d2b90c50` plus the exact portable 536-file snapshot in [eval-freeze.json](eval-freeze.json).
- Freeze SHA256: `2dd3a05970eac8999a17140dc8a5911e644c0a63308aa6fa3c102af24b8a8fb6`.
- Code/module file-map digest: `4365423bdc1edf89bdfa01cfa805a9932cef0963b353edd13e7d047c50347090`; configuration digest: `f60e9e41965ea5d481c56832203da45a65c32e8f57ef9826fcb9e25854098b76`.
- Corpus SHA256: `0d2d66ea73e6aaa76c4ccfc6b8d132e5643746fee0209a04b838111bc6d295fa`; development split SHA256: `b91349406bd152fafbccde11901703c4ca7e98aabcf3c90803c6ef5707005bd8`.
- Independently sealed holdout SHA256: `06eef67ae1b61a647bd40cf7984f54175f8a8454e5d37131baba6582d0c6cbcc`.
- Publication authorization and exact sealed-copy proof: [holdout-publication.json](holdout-publication.json). Publication followed root validation of every frozen file. All 536 hashes were checked again after execution and were unchanged.

Each existing comparison command ran with `GOWORK=off`, `GOCACHE=/tmp/ragy-task19-evaluation-cache`, `RAGY_TASK19_FREEZE=../../docs/task19/results/eval-freeze.json`, corpus `datasets/task19/corpus.json` and split `datasets/task19/holdout.json`, from `examples/conformance`. Reports retain actual command arguments, the full freeze manifest, scoped publication identities, candidate/ranked/delivered IDs, original source locators, resource counters and every raw elapsed observation. No frozen strategy/library source, configuration, qrel or threshold changed after holdout publication. Historical TASK12 evidence files and fixtures are unchanged.

Raw reports: [text](eval-text-holdout.json), [graph](eval-graph-holdout.json), [tensor](eval-tensor-holdout.json). Development reports: [text](eval-text-dev.json), [graph](eval-graph-dev.json), [tensor](eval-tensor-dev.json). Local protocol and limitations: [evaluation.md](../evaluation.md).

## Holdout quality

| Profile | Recall@3 | MRR@3 | Graded nDCG@3 | Retrieved ID coverage | Delivered ID coverage | No-answer errors |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.9667 | 1.0000 | 0.9919 | 1.0000 | 0.9667 | 9/9 |
| Hybrid + host reranker | 0.9833 | 0.9667 | 0.9766 | 1.0000 | 0.9833 | 9/9 |
| Single rewrite | 0.8833 | 0.9333 | 0.8992 | 1.0000 | 0.8833 | 6/9 |
| Multi-query | 0.9833 | 1.0000 | 0.9946 | 1.0000 | 0.9833 | 6/9 |
| Decomposition | 0.9667 | 1.0000 | 0.9919 | 1.0000 | 0.9667 | 6/9 |
| Graph expansion | 0.9833 | 0.9333 | 0.9145 | 1.0000 | 0.9833 | 9/9 |
| Candidate-only tensor MaxSim | 0.8833 | 0.9333 | 0.8859 | 0.9500 | 0.8833 | 9/9 |

All metrics retain failed/missing executions as empty rankings in their applicable denominator. No-answer failures are not counted as successful abstention. In this execution there were no failures. Citation validity is mechanical original source/revision/artifact/location identity; factual entailment and statement-level grounding are unmeasured. Delivered coverage is qrels document/source-ID recall over actual packed contributors, including any partial snippets. All executed rows reported `partial_document_delivery=false` and `delivery_uncertain=false`; renderer packing status is recorded independently.

The maximum possible macro Recall@3 for this independently authored holdout is 0.9833 because one answerable case has four atomic relevant source documents. BM25 starts at 0.9667, leaving only 0.0167 headroom. The fixed gain threshold of 0.05 therefore cannot be satisfied by this fixture. This limitation was discovered after publication and is reported without altering qrels, TopK or the hypothesis. A larger independent domain evaluation is needed to assess that promotion hypothesis.

Multi-query gains 0.0167 Recall, preserves MRR and improves nDCG slightly, but falls short of the fixed Recall threshold. Hybrid and graph gain the same Recall while degrading MRR/nDCG. Single rewrite and tensor lose Recall and ranking quality; compression can discard constraints, and tensor can only rerank the dense candidate set. Tensor candidate coverage of 0.9500 exposes missed gold sources before MaxSim. Decomposition matches BM25 quality here; its development loss remains visible in the preserved development report. None of these observations is a universal default recommendation.

Scripted text recipes abstained on one of the three independent no-answer cases (three repeated successful abstentions), but delivered evidence on the other two. BM25/hybrid/graph/tensor delivered evidence on all three no-answer cases. Mechanical scope/citation correctness does not imply that retrieved facts answer the question. Retrieved malicious instructions remained untrusted source data; no external-agent injection-success experiment was performed.

## Local resources and preparation

Shared query bounds are six retrieval slots, three combined scripted-planner/assessor/encoder/reranker slots, at most twelve materialized candidates per dispatch, at most 36 unique cumulative original candidate IDs, 262,144 local serialized input bytes, 65,536 local serialized output bytes, 1,800 full formatted-context UTF8 bytes and a five-second deadline. Every actual dispatch count and candidate universe is retained. Unknown local accounting or signed/overflowed/duplicate/missing accounting cannot qualify compliance.

| Profile | Retrieval calls | Scripted model-port calls | Local encoder calls | Local reranker calls | Local input bytes | Local output bytes | Full context bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 1 | 0 | 0 | 0 | 0 | 0 | 485–668 |
| Hybrid + host reranker | 2 | 0 | 1 | 1 | 12843–13922 | 2581–4629 | 483–694 |
| Single rewrite | 2 | 2 | 0 | 0 | 14241–30469 | 79–106 | 78–862 |
| Multi-query | 3 | 2 | 0 | 0 | 25387–50410 | 129–274 | 78–646 |
| Decomposition | 1–2 | 2 | 0 | 0 | 8703–27391 | 93–166 | 78–660 |
| Graph expansion | 2 | 0 | 0 | 0 | 0 | 0 | 483–668 |
| Candidate-only tensor MaxSim | 2 | 0 | 1 | 1 | 65088–77099 | 2520–4568 | 485–729 |

Model-port counts describe in-process deterministic planner/assessor invocations. Native hash encoding and MaxSim/overlap reranking have separate counters. Provider calls are zero for every row. Input/output counts include complete serialized text evidence for scripted ports, complete reranker request/response payloads, and declared float32 vector/tensor payload bytes for local encoding/MaxSim. CPU, filesystem storage and provider monetary prices are unpriced; no financial savings claim is made.

Index preparation is explicit and separate from timed query execution, with a 60-second setup bound and raw setup observations. Dense/tensor setup records 36 local encoder invocations, 5,742 text input bytes and 208,640 float32 output payload bytes. Captured elapsed preparation was 13.998s for the text command and 14.137s for the tensor command. Graph prepared two scoped immutable publications in 9.748s; its setup report distinguishes the 40 input corpus rows from source admission. The three command preparations ran concurrently, so these timings are observations of this execution, not isolated comparative performance benchmarks. Each raw query timing remains in its report; no p95/p99 or stochastic uncertainty is fabricated.

## Verification and boundary

Before freeze, shared/text full tests passed with `-race`; graph/tensor actual scoped-source tests passed with `-race`; all changed consumers/helpers passed lint. Meaningful AAA checks cover independently hand-calculated graded ranking metrics, failure/missing denominators, failed no-answer abstention, unsupported dataset versions, wrong scopes, stale/corrupted citations, unknown/overflow/negative accounting, candidate bounds/duplicates, actual bounded recipe dispatch and valid-versus-corrupted freeze file digests. Independent dataset/schema and all-module correctness checks are reported separately by the TASK19 acceptance runner.

This report confirms the explicitly executed local consumer/library composition. It provides no live database/provider verification, stochastic model-quality confidence interval, final-answer judge, external-agent prompt-injection immunity or production-readiness percentage.
