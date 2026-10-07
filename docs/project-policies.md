# Project policy inventory

Inventory at remediation T20, 2026-10-07. These statements describe current repository evidence and pending owner decisions; they do not grant a software license or establish a security-reporting service.

| Policy | Current evidence and decision boundary |
|---|---|
| License | No root LICENSE is present. The owner must choose the license and permitted distribution terms before publication requires them. A license header on third-party tooling configuration does not license ragy. This remediation deliberately does not select a license. |
| Versioning | Release tool derives versions from remote root tags: patch increments patch; incompatible v0 change increments minor. v2+ is rejected until a reviewed semantic import-version migration. Adapter tags share the candidate version with directory prefixes. |
| Changelog | No maintained root CHANGELOG exists. Signed commit history and dated remediation evidence record these changes; they are not a promised public release-note policy. Owner selects the maintained release-note convention. |
| Contribution | Local working contract is spec/contract first, typed host payloads, AAA regression tests and scoped verification. Required repository instructions apply. No external contribution/license-assignment policy is asserted. |
| Security reporting | No owner-approved private reporting address/channel or response SLA is present. Do not infer one from commit email or invent one. Owner must establish the channel before offering public security reporting. |

Missing policy templates do not block runtime conformance. Owner publication/maintenance decisions remain explicit and are not implied by 100% remediation acceptance. Real publication and license choice are outside this remediation goal.

For local contributions inspect affected contracts and run fresh scoped checks with `GOWORK=off`, including relevant nested adapter/example modules. The current core uses the `go 1.27.1` directive; record the actual compiler and linter used. Use `go test -count=1 -race ./...` from each affected module, build affected examples, and retain commands/exit status/profile information. A cached developer run is not fresh acceptance. Use `make test` for cached developer checks and `make acceptance` for fresh lint/race/examples verification. The validated toolchain is recorded in `scripts/toolchain.json`; CI uses those exact pins and standalone GOWORK=off. Each of the fourteen inventoried modules is tested once. Final remediation acceptance and applicable actual backend/parser profiles remain separate gates; no readiness claim follows from this guide.

The [release runbook](release/runbook.md) defines operational publication and exact recovery. Release commands publish refs; ordinary local contributor verification must use disposable local remotes for release fixtures. Do not discard unknown candidate state or select another version to hide an inconclusive publication.

Run `make fuzz FUZZ_SECONDS=2` for a bounded per-function fuzz campaign (default 30 seconds, accepted range 1–300). Every listed Fuzz function is selected independently; listing failures, subprocess failures and timeouts fail the command. `make examples`/`make test-examples` compile examples without repeating their test suites. See [verification commands](verification.md) for the module/profile distinction.
