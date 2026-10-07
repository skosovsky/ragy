# T19 independent correctness acceptance

Verdict: **PASS**, no unresolved correctness findings. Baseline `c6986c5cbc62834e9b66b30b2cb2029be564735d`; current candidate includes the final corrected Classify GoDoc and README spacing. Reviewer did not implement changes or read counterpart acceptance.

## Contract and diff assessment

Only GoDoc/public Markdown, one AAA classification regression and one executable host queue example change. No production statements, dependency, worker, retry or framework change. T19 target contract was checked against `.cursor/tasks/task20-ragy-review-remediation.md` D60 and `docs/task20/reviews/arch-docs.md` A-D03–06, and both T19 acceptance criteria.

- Events counts attempts before callback entry, including errors/panics; Dropped counts rejected operation pairs for capacity or invalid stage. MaxEvents reserves two lifetime callbacks, odd remainder unused. Abandoned/canceled spans receive no automatic End. Failures means failed callback attempts, separate from host queue event losses or asynchronous SDK/export health.
- Finite payload-free by-value enums remain bounded; invalid outcomes/classes normalize, invalid stages reject. Unknown counters clear values, known zero stays known; stage count, token usage and provider billed units are distinct. No raw Error text, scope identity, remote cancellation/refund/price assertion or ordinal metric labels added.
- Serialized synchronous cooperative observers preserve safe Stats access and disabled ordinary nested instrumentation. Explicit same-session reentry remains forbidden because callback mutex serialization would deadlock. Cancellation cannot forcibly terminate callbacks. Observer return errors/panics are isolated; custom classification Is/As/Unwrap methods execute cooperatively and their panics are not sandboxed.
- Host queue example uses copied fixed Events, finite capacity and a distinct atomic event-drop count. Host closes/drains only after producers finish; no worker or lifetime policy enters core. Executed expected output `2 0 0 1 1` establishes two core callback attempts, zero core pair drops/failures, one retained host event and one host event loss.
- Optional OTel ignores starts and exports one span per actual completion; unknown numeric attributes omitted, known zero explicit and signed saturation annotated. SDK asynchronous health is host-owned. Pinned conventions remain dated selection without asserting current upstream release versions.

## Independent executed validation

Go `/opt/homebrew/Cellar/go/1.26.5/bin/go`; `GOTOOLCHAIN=local GOWORK=off GOCACHE=/tmp/ragy-t19-correctness-cache` throughout. Each command exited zero:

| Scope | Command from module directory | Evidence |
|---|---|---|
| Core affected scope + executable queue example | `go test -race -count=1 ./observation ./retrieval ./recipe ./lifecycle` | `T19-correctness-root-race.log`, all four packages PASS |
| Optional standalone OTel module, observer and wrappers | `go test -race -count=1 ./...` | `T19-correctness-otel-race.log`, PASS |
| External public consumer using actual local BM25/cache/pipeline | `go test -race -count=1 ./observation_contract` | `T19-correctness-conformance-race.log`, PASS |
| Independent adversarial public-API probe | From `/tmp/ragy-t19-correctness-probe`, `go test -mod=mod -race -count=1 -v ./...` | `T19-correctness-probe.log`, three tests PASS |

Probe assertions: invalid stages 0/255 increment pair drops without callbacks; odd capacity 3 accepts one operation only; callback entry sees incremented Events and can safely read Stats; failed callbacks count Events/Failures; nested Begin on callback context is disabled; invalid completion enums normalize and unknown count clears while known zero survives. Cancellation while callback blocks does not force Begin to return; releasing host callback completes Begin, and abandonment leaves exactly one start with no fabricated completion. Custom `As` panic escapes Classify without formatting Error(), proving narrower cooperative classification boundary. The probe owns no product files and releases its goroutine/channel deterministically.

Reviewed root author lint evidence `T19-root-lint-final.log`: zero issues with linter deprecation warning only. Independently `git diff --check` passed. No broad all-root PASS asserted: existing production-doc wording blacklist is assigned T21. No SKIP counted as PASS. No changed remote backend/parser profile or optimization requires native service/benchmark evidence here.

## Frozen candidate SHA256 manifest

Mutable backlog/plan/traceability, acceptance reports/logs and unrelated iCloud duplicate are excluded. Manifest covers every changed implementation, test, public document and immutable T19 contract:

```text
13ccccb58a82fc66dcddc6bc008de7859f65002cfe924db8ab6bad573d7c4020  adapters/observability/otel/README.md
ab937c1cfa5901383e65b8ba1b0336c951d0ae5a8796f151e2a199a323f5fd31  docs/task20/T19.md
f8b8b587f70452bf561f9b639bddb0c8878f712c93f93495c9c727b1beeb7ea5  examples/conformance/observation_contract/README.md
0175f3be601b5f3e27e1853f6f5e1a2f5e6fe8396ecd800af5835c79d6551924  observation/README.md
06d7d32c0e378bfe8263ef86866b80097575aa33bb69bde26cf24466e37c3acf  observation/error_boundary_test.go
832670951ca81c334084b55d64f6ca065a1d76d11f3b2d95c272374290d3beec  observation/example_test.go
d286155431ea5b26319925293f600f953f21b539491905252c720c0905cdcd08  observation/observation.go
```
