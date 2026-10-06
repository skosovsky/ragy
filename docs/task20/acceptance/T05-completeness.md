# T05 completeness acceptance

Baseline: `f282696c8fdf15e13bcbba70fca8e6bb146e6b14`. Independent reviewer did not implement the change and did not read correctness report before verdict.

Initial verdict: NOT ACCEPTED. Completeness **2/4 = 50%**. No blocked criteria; two incomplete criteria share one reproduced gap.

| Criterion | Status | Evidence |
|---|---|---|
| T05.C01 | Not completed | Parent/shared/local remaining-time context composition and independent ledger clock pass, but local deadline is sampled before host authority callback in `read.Check`. Final delivery authority clock leap returns expired entities with nil error. |
| T05.C02 | Not completed | Existing 22 callback boundary cases, controlled model barriers, equality, shared just-before, zero entry callbacks and accounting pass. Coverage omits local leap inside final authority callback; independent overlay reproduces missing suppression. |
| T05.C03 | Completed | Config GoDoc/README explicitly source-text MaxInputBytes and actual provider-envelope CountInputTokens. Scripted counter/model serialize identical full request including provider instruction/limits; over-limit rejects without reservation/model. No real-provider tokenizer claim. |
| T05.C04 | Completed | Independent race suite six packages PASS; independent targeted lint 0 issues; source has one Model dispatch, one settlement per acquired lease and no retry. |

Original coverage: F04/G-F01 not fully resolved due final local gate gap; D28/graph:12 completed. Normative contract requires independent local deadline validation at callback/delivery boundaries and expiry equality. A cooperative host freshness callback may advance fake local time; this is admitted controllable-clock behavior, not hostile code.

## Independent commands

- `GOCACHE=/private/tmp/ragy-t05-completeness-cache go test -race -count=1 ./recipe/... ./graphingest/extraction/...` — exit 0, six packages PASS.
- `GOCACHE=/private/tmp/ragy-t05-completeness-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-t05-completeness-lint golangci-lint run --allow-serial-runners ./recipe/... ./graphingest/extraction/...` — exit 0, 0 issues; installed exhaustruct deprecation warning.
- `GOCACHE=/private/tmp/ragy-t05-completeness-cache go test -race -overlay=/private/tmp/ragy-t05-completeness-adversarial/overlay.json -run '^TestCompleteness' -count=1 ./graphingest/extraction` — exit 1. Diagnostic: `final authority leap escaped: checks=45 total=45 entities=2 err=<nil>`.

Adversarial file: `/private/tmp/ragy-t05-completeness-adversarial/boundary_test.go`. It discovers final successful freshness callback count, repeats with independent fake epochs, and advances local clock to exact expiry during final authority callback. Desired zero payload + DeadlineExceeded assertion fails. No substantive repository files changed by reviewer.

## Initial reviewed SHA256

| File | SHA256 |
|---|---|
| `graphingest/extraction/extraction.go` | `848f99824a01153efc01aaad774a3281910513ba861bcf103ce7eeb1487295b3` |
| `graphingest/extraction/contracts.go` | `e35ed0289f1a4861aa6fa83881a3c7df2824e5d1905f5f7bf5eea39a3561daab` |
| `graphingest/extraction/README.md` | `39e66b41f36dda0a2923571e378c6dbbaf2b22ff316900b404c2471e6d019e9f` |
| `graphingest/extraction/shared_deadline_test.go` | `e0cd558aa87cb23482231e5f8d94f9a54b4a1c2a092fc0324ea69b549e62df98` |
| `recipe/budget/budget.go` | `ab6d50234ca54405868f40e2e446cfc73bb1c1839f8dbaa975c0c8327299c2e3` |
| `recipe/budget/deadline_check_test.go` | `bf6d79230fe64190553f34d02fda39ac4854c62520eb7190ac95a531e7de8eb2` |
| `docs/contracts/remediation.md` | `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df` |

## Reacceptance on corrected latest diff

Final verdict: ACCEPTED. Completeness **4/4 = 100%**; no incomplete or blocked criteria. Initial finding resolved and reproduced as PASS. No correctness report read before this verdict.

| Criterion | Status | Current evidence |
|---|---|---|
| T05.C01 | Completed | `ledger.Context` composes remaining shared time with real parent, local remaining interval composes child without assuming clock epochs. `checkExtractionClocks` uses each scope's own clock. Composed gate checks before and after authority; Prepare completion and structural returns retain clock errors. Final authority expiry cannot escape delivery. |
| T05.C02 | Completed | Three earliest-context cases, 22 shared/local callback boundaries, known/unknown settlement and cancellation barriers, equality/expired-entry/clone-cause cases all pass. Permanent final-authority and Prepare-failure+leap tests cover both clocks, no further authority call on failure. Independent original adversarial test now PASS; additional local just-before (+1ns remaining) and real parent/shared/local timer after backwards clocks PASS. Zero payload, one model call and zero outstanding leases asserted. |
| T05.C03 | Completed | Field GoDoc/README state MaxInputBytes sums source text only; instructions/ontology/framing/limits belong in host CountInputTokens full actual provider envelope. Scripted host test counter/model agreement and over-limit pre-reservation rejection pass. No undocumented byte-envelope cap or tokenizer implementation. |
| T05.C04 | Completed | Fresh independent six-package race scope PASS; lint 0 issues. Exact one dispatch and settlement path retained, no retry/publication additions. |

Original coverage: **4/4 = 100%**: F04, graph G-F01, D28, graph:12 each resolved with inspected implementation/contracts and passing acceptance evidence. Clock-only gates do not mutate capacity; accounting remains legal after expiry, with conservative unknown usage and no new lease after expiry. Protection causes and callback causes retain their classifications; no attempt to weaken authorization/pinned publication rules. BYOT ports remain unchanged apart from documented clock/count requirements and Ledger.Check gate.

Fresh independent commands:

- `GOCACHE=/private/tmp/ragy-t05-completeness-cache go test -race -count=1 ./recipe/... ./graphingest/extraction/...` — exit 0; recipe, budget, graphexpand, graphsummary, recording, extraction PASS.
- `GOCACHE=/private/tmp/ragy-t05-completeness-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-t05-completeness-lint golangci-lint run --allow-serial-runners ./recipe/... ./graphingest/extraction/...` — exit 0, 0 issues (installed exhaustruct deprecation only).
- `GOCACHE=/private/tmp/ragy-t05-completeness-cache go test -race -overlay=/private/tmp/ragy-t05-completeness-adversarial/overlay.json -run '^TestCompleteness' -count=1 ./graphingest/extraction` — exit 0; previous final authority repro, local just-before, and three backward-clock real timer probes PASS. These temporary probes use scripted clients, no paid/live provider. Parent/shared/local timer cases wait on ctx.Done without sleeps.

## Final substantive SHA256

Admin backlog/journal/trace/report files excluded; docs contract included as normative reviewed evidence.

| File | SHA256 |
|---|---|
| `graphingest/extraction/extraction.go` | `6c97cc7d7a0eafe3d74484461c6f56640363a15647f3b7ff450a5e6f2996d677` |
| `graphingest/extraction/contracts.go` | `e35ed0289f1a4861aa6fa83881a3c7df2824e5d1905f5f7bf5eea39a3561daab` |
| `graphingest/extraction/README.md` | `39e66b41f36dda0a2923571e378c6dbbaf2b22ff316900b404c2471e6d019e9f` |
| `graphingest/extraction/shared_deadline_test.go` | `ea9ff486c5c2c1c1c099345a07c3d1fcf9958e37a8653cdcc6a4844fcad65ffa` |
| `recipe/budget/budget.go` | `ab6d50234ca54405868f40e2e446cfc73bb1c1839f8dbaa975c0c8327299c2e3` |
| `recipe/budget/deadline_check_test.go` | `bf6d79230fe64190553f34d02fda39ac4854c62520eb7190ac95a531e7de8eb2` |
| `docs/contracts/remediation.md` | `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df` |
