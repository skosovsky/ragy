# T01 — independent correctness acceptance

Reviewer role: correctness; did not implement the contract and did not read the completeness acceptance before this verdict.

Baseline HEAD: `517179b87209a1fb2a41c47057f766dfdce3f959`.
Reviewed source implementation baseline: `b63d5e19a52c7d4e621b1a85a3ce428a92acbcb7`.
Final substantive file: `docs/contracts/remediation.md`.
SHA256: `65d31729d0f80ef8b85ca548fce419ff1019400bbd640592a013bb9a837b45df`.

Final verdict: **PASS** — no remaining actionable correctness findings in this contract-first scope. This is target-contract acceptance, not evidence that F01–F11 are already fixed.

## Findings and independent recheck

The initial file at digest `3bc7bdd4f8280f4b15d8d31be0c41932c813b412f2de787455c0bc25e53f66cf` contained two P2 ambiguities:

1. “RawAttributes nil ... rejected” could reject a whole nil map, despite `NormalizeRawAttributes` and `Schema.NormalizeAttributes` admitting an empty attribute set. This conflicts with optional fields unless a breaking rejection is deliberately specified. Requested distinction: nil/empty whole map is all absent; present nil/null scalar is rejected. **Resolved** in the final filter section; independently reread and compared with `filter/rawattributes.go` and `filter/filter.go`.
2. The blanket nonempty extraction namespace rule could reject the supported ambiguity path: `TestExtractionMixedNamespacesRemainAmbiguous` returns an entity with empty namespace, and resolution Config says absent namespace is never guessed. Requested explicit optional namespace with valid UTF-8 for every nonempty namespace. **Resolved** in the final identity section; required resolved namespace/key/name stay nonempty.

The final digest was calculated after both fixes and after addition of the provider endpoint section.

## Contract checks

| Criterion | Independent assessment |
|---|---|
| T01.C01 | PASS. Callback protection is unconditionally suppressive; joined ordinary/protocol and usage causes are retained. Child-timer provenance is explicitly separate from parent/read authority. Unestablished provenance fails closed. RunObserved journal is only an ordinary failed record with empty Selected. |
| T01.C02 | PASS. Deadline composition uses remaining durations per clock domain, never fake absolute timestamps in real timers. Ledger and adapter evaluate their own Now; equality expires; leaps require boundary checks. Settlement remains exact-once accounting even after denied dispatch/expiry. |
| T01.C03 | PASS. Four scalar domains and numeric-only order match portable matcher intent. Absent leaves have two-valued semantics before grouping; nil/null/wrong-kind fields are admission errors, empty maps are all absent. int64 parity explicitly includes neighbors above 2^53. Real PG query/deletion parity remains an implementation gate. |
| T01.C04 | PASS. Required identities, optional namespace/history fields and callback-versus-input errors are distinguished. Valid JSON tuple + SHA256 framing remains unchanged; invalid bytes are rejected before hashing, and history cannot claim reconstruction after JSON repair. |
| T01.C05 | PASS. Page text Reference uniqueness cannot be substituted by physical-page hashing. Media/cell sharing depends on complete distinct locators and retained host authority; already authorized input and callback/delivery gates are explicit. |
| T01.C06 | PASS. Isolated candidate checkout, reviewed source, manifest/file/ref allowlists and exact object collision policy preserve caller state. Persistent candidate/atomic push plus none/complete/partial/unknown/collision recovery prohibit silent version increment and blind deletion. Clean consumer does not require remote release. |

Provider endpoint section also passes source comparison: shared transport already checks ForceQuery, whereas structured constructor currently omits it. The target requires preflight rejection and parsed path composition, preserving custom `/v1` and trailing slash semantics without a new public framework.

Independent source inspection covered `recipe/run.go`, `recipe/budget/budget.go`, filter matcher/builder/raw attributes/schema, extraction namespace admission/tests, resolution decisions/hash framing/history, layout Document/Page validation, `scripts/release.sh`, structured constructor and shared provider transport. Compared with master F01/F04/F05/F06/F08/F09/F10/F11 and storage/graph/retrieval/ingestion/provider review statements. No owner license choice, host ontology implementation, universal workflow framework or remote release is introduced.

## Commands and evidence

- `git rev-parse HEAD`: baseline SHA above.
- `shasum -a 256 docs/contracts/remediation.md`: final exact digest above.
- `git diff --check`: PASS, exit 0.
- Initial `go test ./filter ./graphingest/extraction ./graphingest/resolution ./recipe/budget`: could not access the default macOS Go cache under sandbox; setup failure, not test failure or PASS.
- Retried `GOCACHE=/private/tmp/ragy-t01-correctness-cache go test ./filter ./graphingest/extraction ./graphingest/resolution ./recipe/budget`: PASS, exit 0 for all four packages. These independently verify baseline public semantics used in the contract review; they do not certify future remediation or live backends.

No implementation files were edited by this reviewer. No mandatory T01 check remains skipped.
