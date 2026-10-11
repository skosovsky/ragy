# Provider SDK contract (migration decision, 2026-10-11)

Baseline fde95a0a25c597a6f7e168a98cef4a62e8166548. Implement before accepting
fixtures. SDK pins selected from the public Go module registry: OpenAI v3.76.0,
google.golang.org/genai v1.73.0, cohere-go/v2 v2.19.0.

Host supplies the official SDK client; credentials, endpoint, transport, redirect
policy and lifecycle remain host-owned. Adapters neither close nor mutate it.
OpenAI and Cohere set per-operation public retry options to zero retries/one
attempt; genai sets per-operation Attempts=1. Hosts must disable HTTP redirects
and hidden transport retries. Earlier context deadlines win over adapter timeout.
Core imports no SDK types. No fallback, private SDK API or provider parser.

Space remains host-owned. Ordered inputs produce ordered vectors; cardinality,
indices (where SDK exposes presence), dimensions and finite values are checked.
No normalization, truncation, rebatching or model switch. Gemini uses the Gemini
API backend, batch EmbedContent, model-001 task types and model-2 text prefixes;
multimodal keeps inline MIME/bytes and part order. Vertex is unavailable here.
Cohere requests all documents (top_n=input count), accepts only a complete ranking,
preserves metadata/resolver/history and model-native finite scores. Partial ranks
are protocol errors with observed billing; top-N selection remains host-side.
Structured uses chat/completions with explicit strict schema, store=false,
max_completion_tokens and host CountTokens/Validate. Local schema/content JSON
validation is domain validation, not a replacement provider envelope parser.
Refusal precedes content validation; length finish is incomplete. Delivery gates
remain authoritative and retain prior callback/recovered error identities.

SDK decoding semantics replace the old envelope admission policy. Available
presence metadata is checked, unknown counters remain unknown; known valid usage
survives domain rejection. No lossless numeric/source guarantee is made where
SDK decoding has already normalized data. Dependency limitations require entries
in limitations.md with fixtures, version, source link and upgrade recheck.

MaxInputs/MaxInputBytes retain exact local admission. Request/response byte limits
now bound SDK parameter serialization/retained public DTO representation where
available, not network bytes or peak allocation. genai lacks public exact request
serialization; its request limit bounds the content representation. Structured counts
the public SDK MarshalJSON request bytes. No capture transport/parser is permitted.
Byte admission is neither a token nor billing cap. Remote token-bound embedding
requests fail unsupported before dispatch. SDK errors retain cause behind a safe
Error string; host must treat errors.As SDK payloads as private diagnostics.
