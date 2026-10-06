# Text recipe comparison scoring

This consumer command captures and scores observations against the embedded fixed
four-document corpus and five query/qrels cases from task12. Offline scoring performs
no retrieval/model calls. Explicit live capture uses actual scoped BM25 and the
configured planner/assessor adapter. The live acceptance run remains outstanding. Contract-fixture captures test the
scorer; they do not establish model quality.

```sh
cd examples/conformance
GOWORK=off go run ./recipe_comparison -input capture.json -output report.json
```

Supply exactly one observation for each query and each profile: baseline,
single-rewrite, multi-query and decomposition. Record original ordered IDs,
execution outcome/stop/failure, actual retrieval/model calls, usage and whether it
is known, raw elapsed nanoseconds and scope/publication identities. Capture also
names corpus/qrels, actual adapter, model, tokenizer and configuration identities,
and explicitly labels live-provider or contract-fixture execution. The scorer
preserves all observations; it does not verify their claimed provenance.

Recall@3 averages per-query relevant-document recall over the four answered queries.
MRR@3 uses the first relevant rank on those same queries. The no-answer query is
excluded from both denominators and separately counts an error when evidence is
selected. Failed answered executions contribute empty ranking metrics and a failure
count; failures of either baseline or recipe prevent threshold qualification.
Duplicate rows/hits, unknown query/profile/document IDs and incomplete grids fail.
Unknown usage cannot satisfy the budget gate. No missing observation becomes zero
usage or an empty successful ranking.

Numeric thresholds require absolute Recall gain at least 0.10, no MRR regression,
no increased no-answer errors, no failed executions and budgets respected by both
profiles. The scorer retains baseline as default regardless of those numbers:
changing a default requires reviewed actual-adapter evidence. Each raw timing is
preserved; five queries are insufficient for production validation or meaningful
latency percentiles. Cost uses the declared mock fixture units, not actual billing.
No adapter credentials or authorization predicates belong in capture/report files.

The executable comparison path is not complete until actual adapter observations
have been generated and reviewed. Live capture requires separately configured
model credentials and an exact host tokenizer for that model; a length estimate or
scripted completion cannot substitute for either.

Model bindings in model_ports.go connect the optional structured planner/assessor
adapter to a trusted local hostCounter. Before construction the host supplies an
attempt context with a deadline, credentials, explicit model and tokenizer
identity, endpoint/client and absolute counter executable path. The executable
receives the complete serialized provider request on stdin and returns one object:

```json
{"model":"host-model","tokenizer_identity":"host-qualified-tokenizer","input_tokens":123}
```

The counter has a two-second computation bound capped by the parent deadline,
a 32 KiB request and 512-byte response cap, no shell/retry and no inherited
credentials. Its process environment is limited to PATH and LANG. Unknown,
malformed, empty or mismatched receipts fail; child errors are sanitized and parent
cancellation is preserved. The host must separately qualify actual model framing
and schema/token counts; naming an identity is not qualification. The executable
is trusted host configuration and must perform only bounded local tokenization.
The local process launch has a specific G204 annotation explaining this boundary,
consistent with the optional parser adapter's host executable contract.

Planner/assessor instructions and executable strict validators are fixed in the
consumer example. Reference call cost is 30 fixture units. HTTP and executable
fixtures verify bindings/usage/cancellation with scripted responses and a fixed
counter count. They do not constitute a live capture or token accuracy result.
Full BM25→recipes→capture orchestration is implemented and tested with actual BM25,
HTTP/subprocess protocol fixtures. Live comparison and tokenizer qualification
remain outstanding.

To capture actual executions, configure OPENAI_API_KEY in the environment and select
the model and a separately qualified trusted tokenizer explicitly:

```sh
GOWORK=off go run ./recipe_comparison -capture capture.json \
  -model "$RAGY_EXPERIMENT_MODEL" \
  -counter "$RAGY_EXPERIMENT_TOKENIZER" \
  -tokenizer-id "$RAGY_EXPERIMENT_TOKENIZER_ID"
GOWORK=off go run ./recipe_comparison -input capture.json -output report.json
```

Live capture uses the configured provider endpoint, with no fixture mode in the CLI.
Missing credentials/model/tokenizer configuration fails before any capture file is
written. The shared corpus is an owned readonly pinned BM25 snapshot, not a claim
of persistent storage. Five baseline and fifteen recipe attempts run once each.
A foreign adversarial document is filtered before payload projection. Each sample
retains selected IDs and exact original references; offline scoring checks their
association with the fixed corpus. Only synthetic caller-authorized IDs/content
are eligible for this experiment's model/capture paths. Arbitrary auth metadata and
credentials are not exported. No final answer generation is part of the example.

Calls are observed at actual retrieval and HTTP boundaries. Raw model stage usage
is retained independently; errors are failed observations with no selected payload.
If accounting cannot be confirmed, usage is explicitly unknown and recommendation
is refused. Capturing failures never turns them into successful retrieval outcomes.
Every recipe has its own five-second context and ledger; the entire grid is bounded
by two minutes. Capture errors abort; individual execution failures remain in the
complete grid for honest scoring. There is no automatic retry or default promotion.

Capture embeds the full fixed reference configuration: corpus digest, planner and
assessor instruction/schema digests, explicit BM25 K1/B, TopK/fusion, per-strategy
query/retrieval/model limits, token/cost caps, per-call reservations, timing and
byte bounds. Config identity is its SHA-256; the offline evaluator requires both
the digest and the actual supported profile. A self-consistent hash of different
limits cannot qualify an observation. Credentials and executable paths are absent.
Model/tokenizer identities are recorded separately. No seed is sent to the provider;
seed policy explicitly says provider-default-no-seed-requested. Captures therefore
do not claim deterministic model sampling or identical repeat-run results. Cost
units are the declared fixture accounting units, not a monetary billing estimate.

## CLI calibration under the clarified acceptance

The author accepted a separate live CLI host profile in issuecomment-6011559475. Reference deterministic/conformance limits remain unchanged. The consumer binding in internal/codexcall launches one trusted CLI process per model port, retains exact port input/instructions/schema, structured output, JSON events, reported token usage and end-to-end timing. Model working directories are temporary; shell/web/app tools are disabled where supported, tool activity observed in JSON events rejects the call. This does not prove that all additional agent context or undisclosed provider dispatches can be excluded; the live report must explicitly describe those limits. It does not establish a hard generation token cap or monetary billing. Missing usage is unavailable, never zero.

Calibration exercises planner → actual scoped BM25 → assessor, without sending qrels/expected IDs to the model:

```sh
GOWORK=off go run ./recipe_comparison \
  -calibrate-codex ../../docs/task12/results/codex-text-calibration.json \
  -codex-bin /absolute/path/to/codex -model configured-model
```

Each calibration model invocation has a45-second subprocess deadline. This is calibration-only, not a complete grid or reference-budget certification. The CLI profile and complete comparison path must be fixed after calibration, before the full run; do not retrospectively raise limits on failed cases. Unknown provider price and advisory input/output bounds must be kept separate from observed tokens and reference hard-budget conformance.

After successful calibration, run the complete fixed grid:

```sh
go run ./recipe_comparison -executor codex \
  -capture /absolute/text-capture.json -calibration /absolute/calibration.json \
  -codex-bin /absolute/codex -model configured-model
go run ./recipe_comparison -input /absolute/text-capture.json -output /absolute/report.json
```

The `.profile.json` is written before comparative calls. Each recipe has a120-second
attempt deadline; CLI invocations retain the calibrated45-second direct subprocess
deadline; the whole grid has30minutes. Reservations are20000input/1024output per
call, advisory, with unknown price. Original retrieval/model call caps stay in
place. These are scheduling reservations, not verified generation limits.
`budgets_honored` continues to score the unchanged reference profile.

Executor arguments and fixed instruction prefix are in `internal/codexcall/call.go`
and the graph consumer README. Reasoning effort is `low`; no seed/temperature is
set. Direct process termination does not prove cancellation of already dispatched
remote generation. Reported usage is retained in full without subtracting cached
or agent input. Provider internal dispatch/retries and additional context remain
unverified, so this is not an isolated retrieval comparison. Unknown provider
price is never inferred from the ledger's zero unpriced units. CLI capture and
provider reference capture are separate modes; core packages do not import CLI.
