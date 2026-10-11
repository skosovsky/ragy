# Migration to official provider SDKs

This breaks adapter constructors. Core, BYOT ports and execution/delivery APIs
are unchanged. Do not reuse a vector space identity for a different revision,
preprocessing or dimension. The SDK model string is not compatibility evidence.

Before: `dense.New(dense.Config{APIKey:key, BaseURL:url, HTTPClient:httpClient,
Space:space})`. After (OpenAI embeddings; the same SDK binds structured):

```go
httpClient := &http.Client{
    CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
}
sdk := openai.NewClient(option.WithAPIKey(key), option.WithHTTPClient(httpClient),
    option.WithMaxRetries(0), option.WithBaseURL("https://api.openai.com/v1/"))
encoder, err := dense.New(dense.Config{SDK:&sdk, Space:space, Limits:limits})
```

Structured: replace credentials/URL/HTTPClient with `SDK:&sdk`; retain Model,
Instructions, SchemaName, Schema, CountTokens, Validate, byte/duration limits.
CountTokens receives the SDK parameter MarshalJSON envelope, including schema and
messages; qualify tokenizer/model framing again rather than assuming byte identity
with the old serializer. Refusal/incomplete remain separate errors. Content JSON
still uses UseNumber and host validation; envelope source fidelity is SDK-limited.

Gemini (explicit Gemini API backend; constructor rejects Vertex):

```go
attempts := int32(1)
sdk, err := genai.NewClient(ctx, &genai.ClientConfig{APIKey:key,
    Backend:genai.BackendGeminiAPI, HTTPClient:httpClient,
    HTTPOptions:genai.HTTPOptions{BaseURL:"https://generativelanguage.googleapis.com/",
        APIVersion:"v1beta", RetryOptions:&genai.HTTPRetryOptions{Attempts:&attempts}}})
encoder, err := dense.New(dense.Config{SDK:sdk, Space:space, Limits:limits})
```

Use the same Config for multimodal with model `gemini-embedding-2`. URL now excludes
`/v1beta`: SDK adds APIVersion. Gemini `DefaultBaseURL` now equals the SDK origin. Inline bytes/MIME/order are preserved. Tokens are
Unknown because the public SDK embedding response discards usageMetadata.

Cohere:

```go
sdk := cohereclient.NewClient(cohereclient.WithToken(key),
    cohereclient.WithHTTPClient(httpClient), cohereclient.WithMaxAttempts(1),
    cohereclient.WithBaseURL("https://api.cohere.com"))
ranker, err := rerank.New[Meta](rerank.Config{SDK:sdk, Model:model, Limits:limits})
```

URL excludes `/v2`: V2.Rerank adds it. `rerank.DefaultBaseURL` now equals the SDK origin. All-document top_n is explicit; host selects
final top-N. Partial responses reject and preserve observable billing. Scores are
model-native, not probabilities. SDK billing float precision is limited.

Host owns credentials, SDK/HTTP client, endpoints, transport and lifecycle. Do not
mutate borrowed clients while calls run. Adapter overrides SDK retries per call
but cannot disable a retrying transport or undo server effects after cancellation.
Error().text is safe; SDK error causes exposed by errors.As can contain private
payloads and must not enter public logs. Local examples never use paid calls:
`go test ./dense`, the structured integration fixtures, and
`GOWORK=off go run ./retry_embedder` in examples/resilience.

See [contract](contract.md), [operation matrix](matrix.md), and
[limitations + upgrade procedure](limitations.md). Byte caps now cover public
representations; retained network byte/peak allocation guarantees changed.

Runnable offline examples for every migrated operation are Go Example tests:
OpenAI dense and structured, Gemini dense and multimodal, Cohere rerank. Run
`GOWORK=off go test -count=1 -run Example ./...` in each adapter module. Real-SDK
HTTP operation fixtures also have `TestIntegrationSDK` names and execute under
`make test-integration`; this is local integration, not live certification.
