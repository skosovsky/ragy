# Lifecycle batch reservation correction

Independent round-1 repro found that ValidateReplacement reserved new manifests
only against the prior snapshot. Two distinct new payload plans in one empty or
nonempty namespace CAS could share an exact target/reference identity.

The correction builds one occupied-reference and digest-fence index only when
there are new rows. It reserves current inventories first, then each new inventory
before admitting the following row. Ordinary CAS with no new manifests does not
allocate the artifact ownership index. Existing immutable identity/fence guards
remain. Snapshot.Validate and all managed read paths are unchanged.

AAA regressions exercise public ValidateReplacement and actual filestore CAS,
empty and nonempty namespaces, different content/payload identities, complete
atomic rejection, and unchanged durable generations/rows. Positive coverage
allows shared support provenance and identical references on distinct targets.
The old abandoned-plan cleanup fixture now assigns its independent plans distinct
artifact IDs while retaining shared original support; it no longer relies on the
previously accepted invalid batch. Benchmark harness/workloads were not edited.

Final frozen validation:

- lifecycle-round2-frozen-race.txt: lifecycle 4.865s, filestore 3.210s,
  materialization 2.011s; all PASS.
- lifecycle-round2-final-lint.txt: compatible Homebrew lint, 0 issues.
- independent-batch-round2-repro.txt: the unchanged independent original repro
  PASS; both public validation and actual CAS return ErrConflict, namespace stays
  generation 0 with zero manifests.
- git diff --check PASS.

Historical round-1 repro, prior measurements and initial round-2 fixture/lint
failures remain separate. Root reruns affected history/retirement benchmarks for
the final candidate; other execution paths and benchmark workloads are unchanged.
No source processes remain running. No commits created.
