# Cohere reranking

Official SDK pin: cohere-go/v2 v2.19.0. Inject `Config.SDK *client.Client`; host owns token/endpoint/transport. V2.Rerank requests top_n=document count. Complete finite ranking preserves original documents/metadata/resolver/score history and protected delivery. Partial replies reject with error and independently observed billing. Scores are model-native, not probabilities. Search units use SDK float64 precision; input tokens remain unknown.

MaxInputs/MaxInputBytes are local admission. MaxRequestBytes/MaxResponseBytes now
bound public SDK representations, not retained network bytes or peak allocation.
Timeout and caller cancellation apply. Byte caps are not token/billing caps;
RequireRemoteTokenBound embedding requests reject before dispatch.

Retries are disabled per SDK operation through public options. Host must disable
redirects/hidden transport retries. Borrowed SDK clients are never closed or
mutated; concurrent use requires a concurrency-safe host client and immutable inputs.
Errors preserve SDK causes behind a sanitized Error string; private SDK diagnostic
payloads must not be publicly logged.

See [migration](../../docs/provider-sdk/migration.md), [contract](../../docs/provider-sdk/contract.md),
[capability matrix](../../docs/provider-sdk/matrix.md), and [accepted limitations / upgrade checks](../../docs/provider-sdk/limitations.md).
Local HTTP fixtures are offline evidence, not paid live certification.
