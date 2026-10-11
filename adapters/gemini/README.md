# Gemini text and multimodal embeddings

Official SDK pin: genai v1.73.0. Inject `Config.SDK *genai.Client` configured with BackendGeminiAPI. Dense supports gemini-embedding-001 task types and gemini-embedding-2 text prefixes. Multimodal model2 supports text/inline PNG/JPEG, six images per input, preserving MIME/bytes/order. URL/file/audio/video/PDF and Vertex profiles are unsupported before I/O. One SDK batch operation; no text-only substitution. Usage/echo fields are SDK-limited and remain unknown.

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
