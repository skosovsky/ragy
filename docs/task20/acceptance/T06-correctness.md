# T06 correctness acceptance

Verdict: **PASS**. Independent reviewer did not implement the change and did not read the completeness report. No blocking correctness findings in T06.C01–C03 / F05 / G-F02.

Baseline: `fd70ce6fc493620ec33448a40aacc7c19011f65e` (verified HEAD). Reviewed the complete resolver/history diff against the normative Identity and Unicode contract, immutable G-F02 report and F05 master finding.

Resolver validates the entire direct identity batch before support/clone/identity callbacks; malformed ff/fe IDs/names/namespaces/endpoints fail ErrInvalidArgument and return the full zero Result. Constructors reject malformed ontology/policy identities. Returned decisions are validated before either canonical hash invocation; relation keys before relation hashing/grouping. Invalid host identities are ErrProtocol with zero result, without dispatching later policy/group callbacks. Ambiguous optional fields remain explicitly empty. U+FFFD and all valid UTF-8 remain lawful; no case, whitespace or Unicode normalization is added. Hash framing is unchanged JSON tuple + SHA256; distinct tuple boundaries and equal tuples retain their previous behavior.

History applies the same domain to all declared metadata/input/result/group/trace identities before serialization and support admission, and again when decoded payload inventory is admitted. Raw invalid UTF-8 JSON is rejected rather than repaired. Required and state-dependent optional fields are coherent with the resolver. BYOT kind/attribute codecs remain host-owned and schema-faithful; no new host type requirement or normalization is imposed. Malformed legacy bytes already repaired to U+FFFD cannot be recovered automatically; documentation explicitly requires host quarantine/migration. Valid legacy snapshot bytes/digests retain compatibility.

Fresh verification:

- `go test -race -count=1 ./graphingest/...`: PASS, five packages including resolver and history.
- `GOCACHE=/private/tmp/ragy-task20-go-cache GOLANGCI_LINT_CACHE=/private/tmp/ragy-t06-correctness-lint golangci-lint run --allow-parallel-runners ./graphingest/...`: PASS, exit 0, 0 issues. The initial default-cache attempt was denied by the filesystem sandbox and a subsequent ordinary run encountered another linter's lock; the successful run used a writable private cache and permitted independent runners. No checks skipped.
- `git diff --check`: PASS.
- Independent persisted-record overlay: `GOCACHE=/private/tmp/ragy-task20-go-cache go test -race -count=1 -overlay=/private/tmp/ragy-t06-correctness-probes/overlay.json ./graphingest/resolution/history -run TestT06IndependentPersistedMalformedAndValidRecords -v`: PASS, five cases. Fixture writes exact self-consistent SHA-addressed payloads with truthful authorized locator inventory: raw ff, raw fe, invalid decision state and empty ontology identity all return ErrProtocol and zero Snapshot; valid U+FFFD restores precisely. Each case separately asserts normal Capture retains the exact SHA256 of the historical `json.Marshal(record)` bytes. Overlay/probe source are in `/private/tmp/ragy-t06-correctness-probes`; no implementation files modified. An initial probe build typo (unused import) was fixed only in the temporary probe before this successful run.

Reviewed substantive file SHA256:

- `docs/contracts/remediation.md`: `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df`
- `graphingest/resolution/resolver.go`: `bce4c70152f32d7f991dbe31004401eae731fa24049d73e179096e3ddb50aec9`
- `graphingest/resolution/contracts.go`: `cf72e2c4b63066780ecec8352932b72e37ba203471a2619fc37652e896394fd0`
- `graphingest/resolution/README.md`: `4d0618166c4b8c9bca6b82d134767ad0aa35ab22ea3f9ddb74131f27a621e0fc`
- `graphingest/resolution/identity_test.go`: `613d1e3c11069a26fea94b2f3fbde87d879f34fe931e62d922794910490a2e10`
- `graphingest/resolution/history/history.go`: `b5bcc7e81aac9aa465ae64e90249485712fd7a66887354f292eb19a57afb7887`
- `graphingest/resolution/history/identity.go`: `879692430920615863abcdea235eca5a1aa5d1775207933458fc627490b2227c`
- `graphingest/resolution/history/README.md`: `48fbe57d3ca4437fa3ec06ccf47bc4ce1fb2ae821c15c8e298ccfb04808198f4`
- `graphingest/resolution/history/identity_test.go`: `8c6d6a38fcb18f7c1f1d381b86e806fef090bba5371323a55822bbd6181c98f7`
