# Independent adversarial review — task21

Verdict: no open confirmed defects found in the reviewed checkout implementation scope after the final repairs. This is not a claim of absolute error absence or successful published delivery.

All independently reproduced findings were repaired and checked: missing lineage/snippet inventory, contributor/canonical association, safe policy-denial classification, native Rank=0 versus effective renderer rank, duplicate snippet, unavailable/truncated delivery completeness flags, and renderer ErrArtifactLimit classification. Canonical Extractor/Losses/Uncertainties are retained independently of optional typed U; required schema fields and public roundtrip regression cover the behavior. Checksums detect accidental corruption; relational tests recompute checksums and still reject invalid evidence.

Fresh full checkout-module test before the final whitelist-only repair:
`GOPATH=/tmp/ragy-task21-go GOCACHE=/tmp/ragy-task21-go-cache GOMODCACHE=/tmp/ragy-task21-modcache GOWORK=off go test -race -count=1 ./...`
Exit 0; package reported 22.281s.

Fresh targeted race tests after the final repair:
`GOPATH=/tmp/ragy-task21-go GOCACHE=/tmp/ragy-task21-go-cache GOMODCACHE=/tmp/ragy-task21-modcache GOWORK=off go test -race -count=1 -run 'TestAdversarial|TestRelationalValidationWithRecomputedChecksum|TestCancellationAndDeadlineAtCooperativeBoundaries|TestCanonicalPublicationAndDurableOwnership|TestManagedSinkRejectsLatePublishAndPreservesNewEpoch|TestAuthorityRevokedAfterRecallBeforePublication' ./...`
Exit 0; package reported 7.946s. The selected suite includes the actual renderer-byte-budget failure/no-publication regression, stable errors.Is preservation, cancellation/deadline classification and private error-text exclusion, corruption with recomputed checksums, unranked retrieval, managed sink late replay fences and authority revocation.

Scope: independent read-only review and public API adversarial consumer fixtures covering explicit identity/scope, source revision binding, canonical epoch admission and Forget exclusion, native evidence/ranks, UTF-8 final spans, codec corruption/ownership, public projection/privacy, independent resource budgets and cooperative cancellation. Runner/CI inspected. No workspace files were edited by the reviewer; temporary reproduction modules and this report reside in /tmp.

Limitations: published composition remains genuinely failed pending publication of the required dependency contract; it is not counted as passed. Release, closeout and overall goal completion are not certified by this implementation review. Live providers, distributed purge, external authority atomicity and authenticity of arbitrary host source storage are not verified. Green tests alone are not proof of error absence.

## Published dependency re-review

Independent disposable consumer `/tmp/ragy-task21-published-review`: copied the current consumer, removed every replace, set GOWORK=off, ran `go mod tidy` (exit 0), and inspected `go list -m -json all`: no Replace entries. The initial direct test before tidy reported missing ragy sums; this was resolved by the same explicit tidy step used by the semantic runner, not by local replacements.

Resolved published sources:
- memy v0.3.1: 27b788e8d56ee33ccb3dd5a8fc99d5dee75fedff; h1:Fw8vN082CsSGJtF4b+mMZW+lpKGeR1/cFQFg4RapSsw=.
- contexty v0.13.1: 3ce901169b09b7136b126d9baa98e1875b33fa82; h1:xLct8igPiDCcA4Q3eL3D6lIYI/UmZpwekSxNKyOZ3hk=.
- ragy v0.8.0: 3a06cb47a91463badb5ec1e240bb72cca6f82a83; h1:U6+ItzoIbw9rChX4e4/mfwml/hUB8ZmgC/ftHNtz5pg=.

Confirmed the published memy module contains presence-aware Score/ScoreOf in score.go and WithDerivedWrite in forget.go. No compatibility fallback is required or present.

Fresh independent published targeted test (same cache environment as above):
`go test -race -count=1 -run 'TestAdversarial|TestRelationalValidationWithRecomputedChecksum|TestCancellationAndDeadlineAtCooperativeBoundaries|TestCanonicalPublicationAndDurableOwnership|TestCanonicalUncertaintyWithoutHostExtension|TestManagedSinkRejectsLatePublishAndPreservesNewEpoch|TestAuthorityRevokedAfterRecallBeforePublication' ./...`
Exit 0; package reported 6.810s.

Read https://github.com/skosovsky/ragy/issues/4#issuecomment-6035141390 via GitHub API. Its old memy v0.2.1 build blocker is now superseded by the independently verified published dependency above. Its remaining delivery requirement is valid: publish the new ragy source tag containing the consumer, then repeat published verification with `--ragy-ref <new tag>`, and close out with linked implementation/migration/review/CI/release evidence. This review does not certify the not-yet-published new ragy tag or the final issue closeout.

Updated verdict: no open confirmed defects found in the independently tested published-dependency composition scope. The dependency publication blocker is cleared for the tested composition against ragy v0.8.0. Final new-tag release validation and closeout remain separate outstanding delivery gates. All prior live/distributed/source-authenticity limitations remain in force. No workspace edits.

## Final release identity audit

Independent read-only audit confirms v0.8.1 points to b12fedb529edfa7304fa713e823d114e496254ef (`git ls-remote origin refs/tags/v0.8.1`). Fresh source clone `/tmp/ragy-task21-released-source.imiufe` has exactly that HEAD. All 13 optional consumer file SHA256 fingerprints match released-source.json and current workspace, with zero mismatches. Comparing released files to the independently tested disposable consumer found no production Go/schema/test changes; README alone gained an accurate explanation of explicit CI source commits and root-only self-replace. The dependency-selection differences were already independently validated.

GitHub compare API from reviewed source 730e94137f5aee0e0213be1a1e90e0c405e54f97 to release commit reports one release commit ahead and only ten adapter go.mod changes; no unreviewed production source changes. The shallow clone lacks the earlier source commit, and the workspace has no local release tag, so local git-diff attempts were inconclusive; remote compare plus actual file hashes supplied authoritative evidence instead.

release-state.json reports patch, complete, last_error=null, observation_error=null, all 11 expected release refs observed at the candidate commit. New-tag published-release.json/log are passed: actual ragy v0.8.1 origin is b12fedb529edfa7304fa713e823d114e496254ef, sum h1:FMmxH9s8YxfPvv7ajNAwFsGgUmDFIi32uQj5uSPSxm0=. Runner executed from released source; every replace was removed and final manifest contains no Replace. `go test -race -count=1 ./...` exits 0 (14.983s), and `go run ./cmd/demo` exits 0 with preserved roundtrip and complete Forget/zero managed contexts.

CI evidence contains 34 successful jobs, including both context-bridge semantic lanes and release-consumer. Independent `gh run view 37605796008` confirms completed/success on source 730e94137f5aee0e0213be1a1e90e0c405e54f97. Conditional sibling checkout steps in published mode are intentionally skipped; semantic tests themselves passed, and no skipped test gate is treated as success.

Final reviewed-scope verdict: no pending confirmed correctness defects; published new-tag composition and release source identity are verified. Closeout remains the parent's next action and is not yet certified here. Live services, distributed purge and other previously stated unsupported guarantees remain outside scope. No workspace edits by reviewer.
