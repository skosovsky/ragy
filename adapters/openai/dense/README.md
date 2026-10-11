# OpenAI dense embeddings

Official SDK pin: openai-go/v3 v3.76.0. Inject `Config.SDK *openai.Client`; host owns credentials, endpoint, HTTP client and lifecycle. Supports text-embedding-3-small (1–1536), large (1–3072), ada-002 (1536). Dimensions/float encoding are explicit; purpose is symmetric. Index/cardinality/model/dimension/finite validation remains local; prompt_tokens presence controls Known. No normalization.

MaxInputs/MaxInputBytes are local admission. MaxRequestBytes/MaxResponseBytes now
bound public SDK representations, not retained network bytes or peak allocation.
Timeout and caller cancellation apply. Byte caps are not token/billing caps;
RequireRemoteTokenBound embedding requests reject before dispatch.

Retries are disabled per SDK operation through public options. Host must disable
redirects/hidden transport retries. Borrowed SDK clients are never closed or
mutated; concurrent use requires a concurrency-safe host client and immutable inputs.
Errors preserve SDK causes behind a sanitized Error string; private SDK diagnostic
payloads must not be publicly logged.

See [migration](../../../docs/provider-sdk/migration.md), [contract](../../../docs/provider-sdk/contract.md),
[capability matrix](../../../docs/provider-sdk/matrix.md), and [accepted limitations / upgrade checks](../../../docs/provider-sdk/limitations.md).
Local HTTP fixtures are offline evidence, not paid live certification.
