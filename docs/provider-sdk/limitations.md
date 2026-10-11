# Accepted dependency limitations

These are accepted semantics, not upstream fixes. No upstream issue/PR link was
found for these exact probes during this task; versioned implementation links
below are evidence, not claims that an issue was filed. No live certification.

| ID | SDK / operation | Reproducing input and observed behavior | Accepted impact / characterization |
|---|---|---|---|
| SDK-L01 | OpenAI v3.76.0, genai v1.73.0, Cohere v2.19.0 / all | Very large JSON whitespace, discarded/unknown fields: public DTO serialization differs from wire; genai request serialization is not public | MaxRequest/ResponseBytes bound public representations, not wire/peak allocation. Existing bounded tests test representation admission. Host owns transport resource policy, no capture parser |
| SDK-L02 | OpenAI v3.76.0 / embeddings & chat | Valid first envelope followed by `{}`; duplicate `choices` | SDK accepts first/trailing envelope, duplicate members normalized. Valid exposed usage survives domain rejection. `TestAdversarialResponses/trailing`, `TestResponseFailuresPreserveKnownUsageWithoutRetry/duplicate-envelope`; no strict provider-envelope promise. Domain content/schema still reject duplicates |
| SDK-L03 | genai v1.73.0 / text & multimodal | `usageMetadata.promptTokenCount:7`, contradictory `model`, embedding `shape` | Converter discards nonpublic fields. Tokens always Unknown for Gemini API embeddings; no echo/shape admission possible. `TestSDKL03DiscardedFields`, `TestMaterializationCancellationPreservesAccounting`; model/revision remains host attestation |
| SDK-L04 | Cohere v2.19.0 / rerank | search_units=9007199254740993 | float64 DTO observes 9007199254740992. No lossless accounting guarantee; reject negative/fractional/nonfinite/int64 overflow, retain observable integral units. `TestSDKL04BillingPrecision` |
| SDK-L05 | Cohere v2.19.0 / rerank | results:[{}] | Scalar absence becomes index=0/score=0; SDK exposes no public presence metadata. Zero may be valid, adapter cannot distinguish. `TestSDKL05AbsentScalars`; other cardinality/index/finite checks remain enforced |
| SDK-L06 | OpenAI v3.76.0 / all | Cancel context in transport just before returning response headers | SDK leaves body unclosed on this path. `TestStructuredContextTransport/headers_cancel`; no claim that upstream leak is fixed. Host owns transport lifecycle and cancellation cannot undo a remote effect |
| SDK-L07 | OpenAI v3.76.0 / all | custom/prefix/// and custom%2F base URLs | SDK preserves triple slashes; escaped slash base resolves to root chat path. `TestProviderBaseURLPolicy`; host must use ordinary trailing-slash URL and verify custom routing |

Sources (public pinned SDK source):
[OpenAI request execution](https://github.com/openai/openai-go/blob/v3.76.0/internal/requestconfig/requestconfig.go),
[OpenAI response decoder](https://github.com/openai/openai-go/tree/v3.76.0/packages/respjson),
[genai response converter](https://github.com/googleapis/go-genai/blob/v1.73.0/models.go),
[Cohere DTOs](https://github.com/cohere-ai/cohere-go/blob/v2.19.0/v_2.go).

Upgrade procedure: choose a published exact version, inspect public API/release
notes, update go.mod/go.sum, run all characterization and local HTTP fixtures
with race (including retries/cancellation/concurrency), then all Make checks and
GOWORK=off consumer tests. Any changed characterization MUST trigger review of
this entry, contract/matrix/docs and mapping. Replace with corrected-behavior
assertions only after reproducing upstream correction. Never Skip/xfail a probe
or reconstruct precision by marshaling a rounded DTO. Never use private SDK APIs.

SDK-L08 — Cohere v2.19.0 rerank typed decode barrier. Input:
`{"results":[{"index":0,"relevance_score":"bad"}],"meta":{"billed_units":{"search_units":7}}}`.
The SDK fails typed decoding and exposes no completed response DTO, so billing is
Unknown; ragy cannot salvage the counter without a forbidden envelope parser.
Characterization: TestSDKL08DecodeBarrier. Source: versioned Cohere DTO link above;
no exact issue/PR found. Recheck this fixture on every SDK upgrade. Valid decoded
billing still survives all subsequent adapter/domain/delivery rejection.
