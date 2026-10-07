# Task21 delivery status

The external published API blocker is resolved: the optional consumer pins memy v0.3.1 (tag commit `27b788e8d56ee33ccb3dd5a8fc99d5dee75fedff`) and contexty v0.13.1. No sibling development replace remains. Core API and root dependencies are unchanged.

Both fresh checkout and published semantic/race/demo runners have passed. `results/checkout.json` and `results/published.json` record dependency/source identities, commands, exit codes and source hashes. Published mode removes all replaces and uses GOWORK=off. The earlier failed published run is retained in `results/published-blocked.log`; `results/dependency-wait.json` is a historical blocker observation, not current state.

The full final `make acceptance` completed successfully (exit 0, `results/acceptance-final.log`). Independent completeness is 100%: AC1–AC10 PASS. Independent adversarial review of the published-dependency composition found no remaining confirmed defects in the checked scope. Final reports are in `reviews/completeness.md` and `reviews/correctness.md`. This does not certify live providers, distributed purge or absolute error absence. Initial acceptance and subsequent optional checks are retained under `results/`.

Delivery remains pending: exact reviewed source commit, CI, `make release-patch`, released source inspection, a fresh published consumer run against the new ragy tag, explicit migration notification and closure of https://github.com/skosovsky/ragy/issues/4. Patch is justified because the diff adds only optional consumer/docs/CI and does not alter core API or behavior. The source tag must contain the optional consumer even though a nested example module is not in the root Go module zip.

The follow-up issue response https://github.com/skosovsky/ragy/issues/4#issuecomment-6035141390 is incorporated: verify the concrete required presence-aware contract, keep absence/native evidence intact, remove sibling replaces, record tag/commit identities, and retest the newly released ragy tag rather than only the previous tag. The earlier blocker comment is historical: https://github.com/skosovsky/ragy/issues/4#issuecomment-6035093927.
