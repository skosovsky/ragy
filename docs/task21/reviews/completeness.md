# Independent task21 completeness review

Read-only review of the current implementation. Workspace source was not modified. Result: **100% (10 fully confirmed AC / 10)**. Partial, blocked and skipped criteria are not counted. Implementation and delivery are assessed separately.

## AC matrix

| AC | Current evidence | Verdict |
|---|---|---|
| AC1 | docs/context-bridge.md, typed Bridge/Input/Sidecar contracts, runtime-compiled sidecar.schema.json; separate optional consumer; no core integration imports/API changes; runner compiles consumer independently. | PASS |
| AC2 | Real canonical Recall, DefaultArtifactRenderer and context codec; TestCanonicalPublicationAndDurableOwnership; scope/namespace/missing/stale/source/denial/protocol cases reject without writes. | PASS |
| AC3 | Deterministic Forget and authority-revocation barriers after Recall; concurrent Forget reaches observable real Store.Update while publication owns fence; managed purge, stale publication/replay, cleanup retry and actual lexical lifecycle cleanup. | PASS |
| AC4 | Absent, zero, negative, incompatible native scores, native observations and stable policy/tie-break; TestAdversarialUnrankedDocuments now preserves native Rank=0 while renderer positional rank is separately validated. | PASS |
| AC5 | Cyrillic/emoji/repeated fragments/JSON escaping/wrapper fixture; spans bind final decoded text; rewriting and truncation remove final exact spans and preserve source-only support with unavailable precision. | PASS |
| AC6 | Fresh registry and actual durable file roundtrip, ownership mutation checks, codec/schema/sidecar corruption failures; inventory/native/contributor/source relations independent of checksum; unique contributors and consistent unavailable/truncated flags. Input always retains canonical extractor, losses and uncertainties; TestCanonicalUncertaintyWithoutHostExtension proves losslessness when host U callback is absent. | PASS |
| AC7 | Empty/partial/omissions/packing/truncation/unknown remain separate; cancellation/deadline errors.Is at cooperative boundaries, stable policy cause and privacy checks. | PASS |
| AC8 | Independent final bytes/runes/durable JSON/model-token budgets including escaping and wrapping; explicit rejection with no publication; host tokenizer and data-role policy. | PASS |
| AC9 | Portable checkout/published runner and CI semantic matrix implemented. Current checkout and published semantic/race/demo lanes both passed against final source; published lane resolves memy v0.3.1 and contexty v0.13.1 with GOWORK=off and no replaces. | PASS |
| AC10 | Fresh checkout and published race/demo lanes passed; final full make acceptance finished exit0 and is saved in docs/task21/results/acceptance-final.log. | PASS |

## Authoritative verification evidence

- docs/task21/results/checkout.json currently status=passed. All recorded consumer source SHA256 values match current files (zero mismatches). go test -race -count=1 ./... and go run ./cmd/demo have exit_code=0.
- docs/task21/results/checkout.log contains final successful race output and executable demo roundtrip/Forget output, matching the updated checkout record.
- docs/task21/results/acceptance-optional.log records fresh optional lint, race and build, each exit_code=0. Final full make acceptance is retained in docs/task21/results/acceptance-final.log and completed exit0; both modes have fresh independent race execution.
- Checkout dependency source identities: ragy be98dba21faceb900c8afa4611238ae86fa6ea9a; memy 9719bc7967031c56daba9ef302edc4994ec8554a; contexty 912c0413994b2b3a1a7d3849a0aa09c71350f915. The runner retains dirty source and individual input hashes, so HEAD alone is not presented as the full source identity.
- Tests run with GOWORK=off, fresh -count=1 and race detection. Scope is offline canonical/renderer/codec plus managed in-process lifecycle profiles; no live services or distributed purge claim.

Final published record was inspected: docs/task21/results/published.json has status=passed, consumer source hashes match all current files, and fresh race tests plus demo have exit_code=0. Published resolution is contexty v0.13.1, memy v0.3.1 and ragy v0.8.0, without replacement directives. The external missing Score/ScoreOf API blocker is resolved. Consumer retains only the ragy development replacement; published runner removes it. Root authoritatively confirmed published session48875 and checkout session91649 finished exit0.

Independent correctness verdict is recorded at docs/task21/reviews/correctness.md: no open confirmed defects found in reviewed checkout implementation scope after repairs. That verdict explicitly excludes successful published delivery and overall completion.

## Implementation verdict

Previous confirmed gaps were fixed: contributor/canonical/source association, lost evidence/lineage/policy classification, revocation coverage, observable concurrent Forget admission, native Rank=0 handling, duplicated contributors, unavailable/truncated flag consistency, canonical extractor uncertainty independent of host U, stale runner hashes. No remaining confirmed implementation defect was found in this completeness review. This does not imply absolute correctness; the independent adversarial reviewer provides its separate verdict.

## Implementation completion

No implementation AC remains incomplete. The root authoritatively confirmed full make acceptance session25854 exit0 and published origin-recording runner session15491 exit0; complete logs are saved in docs/task21/results/acceptance-final.log and published.log. The external published contract blocker is resolved.

## Delivery and closeout verdict

**Implementation 100%; release and published delivery VERIFIED; issue closeout PENDING.** Final authoritative records supersede the historical pending states below. make release-patch on exact reviewed source730e94137f5aee0e0213be1a1e90e0c405e54f97 finished exit0. Published v0.8.1 candidateb12fedb529edfa7304fa713e823d114e496254ef has complete matching observed/expected 11 remote refs; release-state status=complete and no last_error/observation_error. release.log records the successful publication.

CI source730e94137f5aee0e0213be1a1e90e0c405e54f97 has all34 jobs success. released-source.json records the fresh tag clone with all13 optional consumer files, zero local mismatches. I independently compared all13 SHA256 values against current local files and published-release.json runner source files: zero mismatches in both comparisons.

The runner from the released source explicitly applied -require=github.com/skosovsky/ragy@v0.8.1, removed local replaces, used GOWORK=off, and passed fresh go test -race -count=1 ./... plus executable demo (exit0). published-release.json records ragy originb12fedb529edfa7304fa713e823d114e496254ef refs/tags/v0.8.1 and published memy/contexty origins and checksums. This meets the additional post-release --ragy-ref NEW_RELEASE_TAG gate requested at https://github.com/skosovsky/ragy/issues/4#issuecomment-6035141390.

Remaining requirement before overall goal completion: explicitly notify the issue author with actual APIs, migration/before-after snippets and implementation/reviews/CI/release links, then close issue4 and independently verify comment URL/body and closed state. No issue closeout is credited yet.

Artifacts inspected: docs/task21/results/ci.json, release.log, release-state.json, released-source.json, published-release.json and published-release.log. No tests were repeated because source/evidence checks exposed no new risk.


Final published origin evidence: ragy v0.8.0 -> 3a06cb47a91463badb5ec1e240bb72cca6f82a83 (refs/tags/v0.8.0); memy v0.3.1 -> 27b788e8d56ee33ccb3dd5a8fc99d5dee75fedff (refs/tags/v0.3.1); contexty v0.13.1 -> 3ce901169b09b7136b126d9baa98e1875b33fa82 (refs/tags/v0.13.1). Each includes module Sum and GoModSum in published.json. The runner's logging-only change adds identity evidence and preserves semantic/race/demo gates.

Completion is implementation-only: current published lane uses existing ragy v0.8.0 and proves composition semantics, but does not prove publication of this newly implemented optional example. Delivery therefore still requires release/CI and a post-release published runner using --ragy-ref NEW_RELEASE_TAG, followed by the explicit author notification and issue closure.


Source identity audit: current HEAD 42062c9b00c6d6f9bf3b9a1de9f9ffbba206e14a on the implementation branch. Its only commit diff is CI push branch filter addition of codex/** (one workflow line); consumer source hashes still match both passed records with zero mismatches. This change enables branch CI and does not alter implementation semantics or invalidate prior consumer acceptance. CI run https://github.com/skosovsky/ragy/actions/runs/37605363190 and exact-source make release-patch are pending, not credited as successful. Final delivery audit must verify release tag/source identity, published runner with that new ragy tag, and issue notification/closure.


CI dependency-profile repair audit: source 730e94137f5aee0e0213be1a1e90e0c405e54f97 explicitly checks out reviewed memy9719bc7967031c56daba9ef302edc4994ec8554a and contexty912c0413994b2b3a1a7d3849a0aa09c71350f915, exactly matching previously accepted checkout identities. The prior CI failed because remote default branches predated the required contracts, not because the consumer semantics failed. The README documents deliberate ref selection and published tag independence. No consumer Go/schema semantics changed. Current prerelease records /tmp/task21-checkout-prerelease.json and /tmp/task21-published-prerelease.json both status=passed with zero current source-file hash mismatches; the older repository records differ only on the README documentation change. Implementation acceptance remains 100%.

Delivery remains pending: previous release attempt was interrupted exit1 during acceptance before publication and is not credited. Its retained artifact is docs/task21/results/release-initial-interrupted.log. Failed initial CI evidence is retained in docs/task21/results/ci-checkout-initial-failed.log. Corrected CI run https://github.com/skosovsky/ragy/actions/runs/37605796008 is pending. Release should proceed only against the corrected source and successful gates, then verify new-tag published composition and issue closeout.


Corrected CI completion audit: inspected /tmp/task21-ci-final.json. Run status completed/conclusion success on exact source730e94137f5aee0e0213be1a1e90e0c405e54f97; all 34 jobs concluded success, including context-bridge checkout and published semantic lanes, optional consumer lint/tests, all module checks and release-consumer. URL https://github.com/skosovsky/ragy/actions/runs/37605796008. This supersedes the prior pending CI state; implementation AC1–AC10 remains 100%.

Release session35401 is still live and is not credited as successful by this review. Delivery remains pending until exact-source release success, published check using the new release tag and author notification/issue closeout are independently audited. CI success alone does not complete the overall goal.
