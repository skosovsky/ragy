# TASK19 fictional engineering dataset v1

Dataset `ragy-fictional-ops-v1`, schema `1.0.0`, contains 40 authored, short fictional engineering runbook documents and 36 independently authored queries. No external source, personal information, provider output or strategy ranking informed the labels. This small controlled consumer experiment measures retrieval behavior; it does not establish production-domain quality or final-answer correctness.

Public data live in `examples/conformance/datasets/task19/`: `corpus.json`, `dev.json`, `schema.json`, `manifest.json`, and eventually `holdout.json`. Before strategy freeze, holdout is withheld in `/tmp/ragy-task19-holdout-author/holdout.json`; its immutable SHA-256 is committed to the manifest before execution. Publication is permitted only after the evaluation implementation/configuration digest is frozen and root authorizes publication. No query, label or threshold changes may follow observed holdout results. Changes require a new version and a new experiment; the old evidence stays intact.

## Executable wire contract

`schema.json` is JSON Schema draft 2020-12 with an exact object contract and no additional fields. `verify.py` is a standard-library-only interpreter for the exact schema subset used here plus semantic referential checks. It validates IDs, relation targets, grades, category balance, answerability, split separation, source/revision uniqueness, qrel eligibility and SHA-256. It does not implement arbitrary JSON Schema drafts. A conforming full draft 2020-12 validator can validate these same files without conversion.

```sh
python3 examples/conformance/datasets/task19/verify.py
```

Before publication this checks corpus/development and explicitly reports only the present split. The dataset author additionally runs the verifier with the withheld path; implementation authors must not use that option before freeze. After publication, the same default command checks both splits.

Corpus root fields: `schema_version`, `dataset_id`, `documents`. Each document has `id`, `source_id`, `revision`, `scope`, `current`, `text`, and `relations`. Relation fields are `target_id`, `type` (`depends_on` or `owned_by`). These domain fields and admission decisions belong only to this consumer; they must not enter universal library types. Relation facts are host authored traversal hints, with document text as the evidence.

Split root fields: `schema_version`, `dataset_id`, `split`, `queries`. Each query has `id`, `case_id`, `category`, `text`, `scope`, `answerable`, `qrels`, `notes`. Each qrel contains `document_id`, `grade`. Relevance grades: 3 directly answers a requested atomic fact; 2 supplies necessary explanatory/intermediate evidence; 1 supplies procedure/default/prerequisite context useful to resolve the question. Missing IDs imply grade 0. Grades judge source relevance, not quality of a generated answer.

## Source identities and host admission

The original citation locator is deterministically `source_id@revision`; the artifact ID and document ID are `id`. Each row is a complete original `text/plain` representation, spanning UTF-8 bytes `[0,len(text.encode("utf-8")))`; no transformed summary or chunk is supplied. Duplicate text from a mirror retains its own source, revision and document identity. A consumer must preserve both citations; deduplicating content does not merge original support identities.

The host eligible corpus for a query includes rows with `current == true` and `scope == public` or `scope == query.scope`. Query scope is one of public/blue/red, not a claim that a user has access to every workspace. `current` is this fixture's publication decision and must be bound through the consumer's actual admission/lifecycle interface, rather than treated as a library freshness or IAM guarantee. Obsolete rows remain in the source corpus as adversarial evidence but are ineligible before retrieval. Requests for a private fact from public scope intentionally have empty gold and require abstention. No gold asks for an unavailable cross-scope source.

Scoped exceptions coexist with public defaults: both can be graded evidence when explaining precedence; the relevant override receives grade 3 and the applicable public default grade 1. Obsolete conflicting revisions never become qrels. Multi-hop Recall counts each relevant atomic document, including intermediate supports. A relation edge alone cannot replace a document citation or reduce that denominator.

## Coverage and independence

Each split has 18 cases, exactly two for each category: lookup, multihop, no_answer, harmful_rewrite, duplicates, conflicting_stale, limited_scope, multilingual, malicious_instructions. Each has three no-answer cases: two unsupported facts and one public request for an inaccessible private fact. The category `limited_scope` includes both positive admitted access and negative inaccessible evidence. Four source rows are stale; six are workspace private. Injection-bearing imported notes remain current public data so factual retrieval is measurable; successful execution of those instructions is outside this retrieval dataset and is not an agent-immunity claim.

Development and holdout case IDs and texts are disjoint. Development uses Atlas checkpoint rollback, Beacon readiness, Cedar index publishing, Orchid export freezes and Moss diagnostics. Holdout uses Kestrel lease drain/Quartz ownership, Raven replay timing, Saffron snapshot/Opal checksums, Tulip payload parking and Willow quarantine/diagnostics. Holdout changes the actual entities, required facts, preservation operations, intermediate dependencies, failure codes and workspace positive/negative cases. It was authored as a separate source-to-question exercise, not by copying development qrels or paraphrasing development texts. Shared categories and preservation constraints are intentional test coverage, not tuning material.

No embedding vectors are persisted. Any deterministic local hash embeddings, tokenization, scripted rewriting or scoring are separate implementation profiles whose revisions, usage source and enforcement must be recorded by evaluation. This dataset does not claim model calls, provider effectiveness, seed support or billed cost.
