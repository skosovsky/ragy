# TASK19 retrieval evaluation

This is a deterministic local consumer comparison over fictional operations documentation. It exercises actual BM25, scoped persistent dense/tensor publication, RRF, bounded text recipes, managed graph traversal and source-aware context packing. It does not measure a language model's ability to rewrite, assess or answer questions. Host-authored graph relations and local hash vectors are explicit fixtures. Live provider quality and monetary billing remain unexecuted.

The strategy implementation may read corpus and development queries. Holdout authoring is independent; holdout publication follows the complete implementation/configuration digest freeze. The author corrected a missing mirrored procedural support before inspecting any ranking; the dataset manifest records this author-only correction. No holdout ranking, qrel or threshold influences a strategy.

The fixed policy is TopK=3, per-dispatch candidate limit=12 and whole-attempt unique candidate universe=36, six retrieval slots, three combined local planner/assessor/encoder/reranker dispatch slots, 262,144 serialized input bytes, 65,536 serialized output bytes (UTF8 text and float32 tensor/vector payloads), full formatted context at most 1,800 UTF8 bytes, and five seconds per query. Provider tokens are unavailable because no provider runs. Local counters use declared serialized-byte units; zero provider dispatches are distinct from unknown provider prices. Packing measures the exact serialized envelope and citation labels using the library artifact renderer. Scope includes public documents plus the caller's explicit scope. Host publishes current corpus revisions; obsolete revisions cannot enter that publication. Mechanical citation validity verifies original source/revision/artifact identity and source locations; it is not a factual entailment judgment.

Predeclared strategies:

- Baseline: actual pinned readonly BM25, k1=1.2, b=0.75, top three documents.
- Hybrid: actual scoped persistent dense candidates from SHA256 token sign vectors (64 dimensions), BM25 candidates, library RRF k=60, then a host overlap reranker over at most twelve candidates. Embedding and reranker are deterministic local computations, with separate dispatch counters.
- Single rewrite: remove a fixed English stopword list and retain the first six unique query terms. This intentionally simple compression can lose constraints; original and rewritten evidence are assessed by an explicit scripted host.
- Multi-query: normalized unique terms and a fixed `roll back` → `rollback` spelling variant. Decomposition splits query text on the first ` and `, otherwise comma. Neither planner reads categories, qrels or answerability labels. All recipe attempts use the actual shared library budget ledger.
- Assessor: retain query indices with a document sharing at least two non-stopword terms with the original query. It counts the complete serialized assessment input as local input; semantic sufficiency is only a host heuristic.
- Graph: top one lexical seed, actual managed admitted depth-one expansion of host-authored relations, neighbors before seed, then lexical fallback. Projected targets require the target's own admitted original source; relation mentions alone cannot supply target document evidence.
- Tensor: actual persistent dense candidate generation, then native candidate-only MaxSim with SHA256 token sign vectors, never a full-corpus late-interaction quality claim.

The hypothesis requires an absolute holdout Recall@3 gain of at least 0.05 over BM25, no MRR@3/nDCG@3 degradation, no additional no-answer errors or failures, no scope/stale/citation violations, and confirmed equal known resource compliance. These numeric criteria never promote a universal default using this local fixture alone. Baseline remains the default; losses are valid findings.

Three deterministic repetitions check ranking reproducibility. They are not independent stochastic samples and yield no confidence intervals or latency percentiles. Each raw elapsed observation remains in the capture. Failed or missing executions produce empty rankings and stay in answerable denominators. No-answer cases have a separate denominator; delivery on one is a no-answer error. Recall/MRR/nDCG use independently authored graded qrels. Retrieved coverage measures the union of actual backend candidates; delivered coverage measures document/source-ID qrels recall over actual packed contributors. It does not measure semantic completeness, statement-level grounding or factual entailment; partial snippets still contribute their source IDs. Rows separately record renderer packing status, partial-document delivery and delivery uncertainty. No final-answer judge or external-agent prompt-injection immunity is claimed. Malicious instructions remain untrusted source text inside the rendered boundary.

Dense/tensor and graph stores are prepared explicitly once before timed queries, with separate raw setup elapsed, input/output bytes, document counts and publication identities. Preparation has a separate 60-second bound; query budgets never hide an indexing attempt.

Run each existing command from `examples/conformance` with `GOWORK=off`:

```sh
RAGY_TASK19_FREEZE=../../docs/task19/results/eval-freeze.json go run ./recipe_comparison -task19-corpus datasets/task19/corpus.json -task19-split datasets/task19/holdout.json -task19-output ../../docs/task19/results/eval-text-holdout.json
RAGY_TASK19_FREEZE=../../docs/task19/results/eval-freeze.json go run ./graph_comparison -task19-corpus datasets/task19/corpus.json -task19-split datasets/task19/holdout.json -task19-output ../../docs/task19/results/eval-graph-holdout.json
RAGY_TASK19_FREEZE=../../docs/task19/results/eval-freeze.json go run ./tensor_comparison -task19-corpus datasets/task19/corpus.json -task19-split datasets/task19/holdout.json -task19-output ../../docs/task19/results/eval-tensor-holdout.json
```

Historical TASK12 reports and embedded fixtures remain unchanged. Their CLI captures retain their documented executor context, advisory token bounds and unavailable provider prices; they cannot establish an isolated provider comparison. TASK19 reports have a separate schema and paths. Unit/contract/wire/race evidence is separate from retrieval quality and live-service availability.
