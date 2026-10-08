package dense

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/gemini/internal/wire"
	root "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

func profile(model string) embedding.Space {
	return embedding.Space{
		Model:         model,
		ModelRevision: "host-pinned",
		Configuration: "gemini-default-v1",
		VectorSpace:   "test",
		Dimension:     2,
		Metric:        embedding.Cosine,
	}
}

// Official fixtures verified 2026-10-06: https://ai.google.dev/api/embeddings
// and https://ai.google.dev/gemini-api/docs/embeddings . These are protocol tests.
func TestBatchProtocol(t *testing.T) {
	for _, tc := range []struct {
		model      string
		purpose    embedding.Purpose
		task, text string
	}{
		{"gemini-embedding-001", embedding.Query, "RETRIEVAL_QUERY", "hello"},
		{"gemini-embedding-001", embedding.Document, "RETRIEVAL_DOCUMENT", "hello"},
		{"gemini-embedding-001", embedding.Similarity, "SEMANTIC_SIMILARITY", "hello"},
		{"gemini-embedding-2", embedding.Query, "", "task: search result | query: hello"},
		{"gemini-embedding-2", embedding.Document, "", "title: none | text: hello"},
		{"gemini-embedding-2", embedding.Similarity, "", "task: sentence similarity | query: hello"},
	} {
		t.Run(tc.model+string(tc.purpose), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				checkBatchRequest(t, r, tc.model, tc.task, tc.text)
				_, _ = w.Write(
					[]byte(`{"embeddings":[{"values":[3,4]},{"values":[4,3]}],"usageMetadata":{"promptTokenCount":7}}`),
				)
			}))
			defer server.Close()
			client, err := New(Config{APIKey: "secret", Space: profile(tc.model), BaseURL: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			result, err := client.Embed(
				context.Background(),
				root.Request{Inputs: []string{"hello", "other"}, Purpose: tc.purpose},
			)
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Embeddings) != 2 || result.Embeddings[0].Vector[0] != 3 ||
				result.Embeddings[1].Vector[0] != 4 ||
				result.Embeddings[0].Space != client.Space() ||
				!result.Usage.InputTokensKnown ||
				result.Usage.InputTokens != 7 ||
				result.Usage.BilledUnitsKnown {
				t.Fatalf("result=%+v", result)
			}
		})
	}
}
func TestMalformedResponses(t *testing.T) {
	for _, body := range []string{`{}`, `{"embeddings":[]}`, `{"embeddings":[{"values":[1]}]}`, `{"embeddings":[{"values":[0,0]}]}`, `{"embeddings":[{"values":[1e100,1]}]}`, `{"embeddings":[{"values":[1,2],"shape":[2]}]}`, `{"embeddings":[{"values":[1,2]}],"model":"other"}`, `{"embeddings":[{"values":[1,2]},{"values":[1,2]}]}`, `{"embeddings":[{"values":[1,2]}],"usageMetadata":{"promptTokenCount":-1}}`, `{"error":{"message":"secret"}}`} {
		t.Run(body, func(t *testing.T) {
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(body)) }),
			)
			defer server.Close()
			client, err := New(Config{APIKey: "secret", Space: profile("gemini-embedding-001"), BaseURL: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			_, err = client.Embed(context.Background(), root.Request{Inputs: []string{"hi"}, Purpose: embedding.Query})
			if !errors.Is(err, ragy.ErrProtocol) || strings.Contains(err.Error(), "secret") {
				t.Fatalf("err=%v", err)
			}
		})
	}
}
func TestUnknownUsage(t *testing.T) {
	for _, usage := range []string{"", `,"usageMetadata":{}`, `,"usageMetadata":{"promptTokenCount":0}`} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"embeddings":[{"values":[1,2]}]` + usage + `}`))
		}))
		client, err := New(Config{APIKey: "key", Space: profile("gemini-embedding-001"), BaseURL: server.URL})
		if err != nil {
			t.Fatal(err)
		}
		result, err := client.Embed(
			context.Background(),
			root.Request{Inputs: []string{"hi"}, Purpose: embedding.Document},
		)
		server.Close()
		if err != nil {
			t.Fatal(err)
		}
		if result.Usage.InputTokensKnown != (strings.Contains(usage, "promptTokenCount")) {
			t.Fatalf("usage=%+v", result.Usage)
		}
	}
}
func TestLocalRejection(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) { calls++ }))
	defer server.Close()
	client, err := New(
		Config{
			APIKey:  "key",
			Space:   profile("gemini-embedding-001"),
			BaseURL: server.URL,
			Limits:  embedding.Limits{MaxInputBytes: 4, MaxInputs: 2},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, request := range []root.Request{{Inputs: []string{"hi"}, Purpose: embedding.Query, RequireRemoteTokenBound: true}, {Inputs: []string{"hi"}, Purpose: "unknown"}, {Inputs: []string{"12345"}, Purpose: embedding.Query}, {Inputs: []string{" "}, Purpose: embedding.Query}, {Inputs: []string{"a", "b", "c"}, Purpose: embedding.Query}, {Inputs: []string{string([]byte{255})}, Purpose: embedding.Query}} {
		if _, err := client.Embed(context.Background(), request); err == nil {
			t.Fatal("expected rejection")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := client.Embed(
		ctx,
		root.Request{Inputs: []string{"hi"}, Purpose: embedding.Query},
	); !errors.Is(
		err,
		context.Canceled,
	) {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("calls=%d", calls)
	}
	bad := profile("unknown")
	if _, err := New(Config{APIKey: "key", Space: bad}); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal(err)
	}
	if _, err := New(
		Config{APIKey: "", Space: profile("gemini-embedding-001")},
	); !errors.Is(
		err,
		ragy.ErrInvalidArgument,
	) {
		t.Fatal(err)
	}
}

func checkBatchRequest(t *testing.T, r *http.Request, model, task, text string) {
	t.Helper()
	if r.Method != http.MethodPost || r.URL.Path != "/models/"+model+":batchEmbedContents" ||
		r.URL.RawQuery != "" ||
		r.Header.Get("X-Goog-Api-Key") != "secret" {
		t.Errorf("wrong request: %s %s", r.Method, r.URL.Path)
	}
	var body wire.Batch
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Error(err)
	}
	if len(body.Requests) != 2 {
		t.Errorf("requests=%d", len(body.Requests))
		return
	}
	item := body.Requests[0]
	if item.Model != "models/"+model || item.TaskType != task || item.Dimension != 2 ||
		len(item.Content.Parts) != 1 ||
		item.Content.Parts[0].Text != text {
		t.Errorf("body=%+v", body)
	}
}
func TestBoundedResponseAndHTTPError(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		class  error
	}{
		{"oversized", 200, strings.Repeat("x", 65), ragy.ErrProtocol},
		{"refused", 403, "secret request and response", ragy.ErrInvalidArgument},
		{"rate", 429, "secret request and response", ragy.ErrUnavailable},
		{"redirect", 302, "secret request and response", ragy.ErrUnavailable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls++
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			client, err := New(
				Config{
					APIKey:  "secret",
					Space:   profile("gemini-embedding-001"),
					BaseURL: server.URL,
					Limits:  embedding.Limits{MaxResponseBytes: 64},
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			_, err = client.Embed(context.Background(), root.Request{Inputs: []string{"hi"}, Purpose: embedding.Query})
			if err == nil || strings.Contains(err.Error(), "secret") || calls != 1 {
				t.Fatalf("err=%v calls=%d", err, calls)
			}
			// Status mapping is the shared provider transport contract.
			if tc.status != 302 && !errors.Is(err, tc.class) {
				t.Fatalf("err=%v", err)
			}
		})
	}
}
func TestConstructorRejectsConflictingModel(t *testing.T) {
	_, err := New(Config{APIKey: "key", Model: "gemini-embedding-2", Space: profile("gemini-embedding-001")})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal(err)
	}
}
