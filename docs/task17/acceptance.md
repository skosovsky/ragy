# TASK-17 — accepted

Candidate `cef29e5b3d7750d1a7b36d867d0b1de0b51435f60008bf46e593136cd7690a81` independently accepted at 100% completeness (10/10) and with no confirmed open defects. Both final reviewers verified all 64 code/spec hashes. Rejected round 1 and round 2 final integration gate reports retained.

Final validation: core-round2-final-race/lint; examples-conformance-round2-final-race and conformance-round2-final-lint; adapters-openai-final-race/lint; adapters-observability-otel-round2-final-race all exited 0. Full actual PDF/parser, planner/resilience suites passed on unchanged consumer sources. Focused observer/retrieval/lifecycle/recipe/evidence races and independent JSON Schema plus relational verifier passed; 18 shared fixtures. Diff whitespace check passed.

Earlier core/conformance lint, initial partial integration and adapters-openai-round2-final-race logs contain failures and remain historical; the named final successful logs supersede them. Tests partially use cache, independent regressions use count=1/10. No live provider or database execution claimed.

Clear break: evidence schema ragy.retrieval-evidence/v2 explicitly rejects old records, which remain historical. Optional core observer is bounded and payload-free, uses cooperative serialized callbacks and no OTel/provider dependencies. Failure/panic never retries or changes dispatch. Decision export is default-deny, with separate named permissions and unavailable identities; no model reasoning. OTel uses an explicitly selected development subset 1.38.0. Runtime byte/depth bounds and independent relational checks complement declarative schema; neither authenticates provenance. Source admission, sink durability, callback promptness and provider billing remain host responsibilities.
