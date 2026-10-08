package dense

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	rootdense "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

// Wire contract verified 2026-10-06: https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/
// These fixtures exercise documented structure; they do not attest a live service.
func testSpace() embedding.Space {
	return embedding.Space{
		Model:         "jina-embeddings-v3",
		ModelRevision: "host-pinned",
		Configuration: "fixture",
		VectorSpace:   "retrieval-pair",
		Dimension:     2,
		Metric:        embedding.Dot,
	}
}
func testClient(t *testing.T, url string, limits embedding.Limits) *Client {
	t.Helper()
	c, err := New(Config{APIKey: "secret", Space: testSpace(), BaseURL: url, Limits: limits})
	if err != nil {
		t.Fatal(err)
	}
	return c
}
func TestDocumentedWire(t *testing.T) {
	for _, purpose := range []embedding.Purpose{embedding.Query, embedding.Document} {
		t.Run(string(purpose), func(t *testing.T) {
			server := httptest.NewServer(wireHandler(t, purpose))
			defer server.Close()
			c := testClient(t, server.URL, embedding.Limits{})
			result, err := c.Embed(
				context.Background(),
				rootdense.Request{Inputs: []string{"hello", "world"}, Purpose: purpose},
			)
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Embeddings) != 2 || result.Embeddings[0].Space != c.Space() ||
				result.Usage.InputTokens != 7 ||
				!result.Usage.InputTokensKnown ||
				result.Usage.BilledUnitsKnown {
				t.Fatalf("bad result: %+v", result)
			}
			if result.Embeddings[0].Vector[0] != 1 || result.Embeddings[1].Vector[1] != 1 {
				t.Fatal("output order")
			}
		})
	}
}
func TestAdversarialResponses(t *testing.T) {
	cases := map[string]string{
		"missing": "{}", "missing-index": `{"data":[{"embedding":[1,0]}]}`,
		"out-of-range": `{"data":[{"index":1,"embedding":[1,0]}]}`,
		"extra":        `{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[1,0]}]}`,
		"empty":        `{"data":[{"index":0,"embedding":[]}]}`,
		"wrong-shape":  `{"data":[{"index":0,"embedding":[1]}]}`,
		"overflow":     `{"data":[{"index":0,"embedding":[1e999,0]}]}`,
		"nan":          `{"data":[{"index":0,"embedding":[NaN,0]}]}`,
		"model":        `{"model":"wrong","data":[{"index":0,"embedding":[1,0]}]}`,
		"usage":        `{"data":[{"index":0,"embedding":[1,0]}],"usage":{"total_tokens":-1}}`,
		"trailing":     `{"data":[{"index":0,"embedding":[1,0]}]} {}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(body)) }),
			)
			defer server.Close()
			c := testClient(t, server.URL, embedding.Limits{})
			_, err := c.Embed(
				context.Background(),
				rootdense.Request{Inputs: []string{"hello"}, Purpose: embedding.Query},
			)
			if !errors.Is(err, ragy.ErrProtocol) {
				t.Fatalf("error %v", err)
			}
		})
	}
}
func TestDuplicateAndMissingOutput(t *testing.T) {
	for _, body := range []string{`{"data":[{"index":0,"embedding":[1,0]},{"index":0,"embedding":[1,0]}]}`, `{"data":[{"index":0,"embedding":[1,0]}]}`} {
		server := httptest.NewServer(
			http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(body)) }),
		)
		c := testClient(t, server.URL, embedding.Limits{})
		_, err := c.Embed(
			context.Background(),
			rootdense.Request{Inputs: []string{"hello", "world"}, Purpose: embedding.Document},
		)
		server.Close()
		if !errors.Is(err, ragy.ErrProtocol) {
			t.Fatalf("error %v", err)
		}
	}
}
func TestUnknownUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[1,0]}]}`))
	}))
	defer server.Close()
	c := testClient(t, server.URL, embedding.Limits{})
	result, err := c.Embed(context.Background(), rootdense.Request{Inputs: []string{"hello"}, Purpose: embedding.Query})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.InputTokensKnown || result.Usage.BilledUnitsKnown {
		t.Fatal("invented usage")
	}
}
func TestLocalRejectionBeforeDispatch(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			calls.Add(1)
			w.WriteHeader(http.StatusInternalServerError)
		}),
	)
	defer server.Close()
	c := testClient(t, server.URL, embedding.Limits{MaxInputBytes: 4, MaxInputs: 1})
	cases := []rootdense.Request{
		{Inputs: []string{"hi"}, Purpose: embedding.Query, RequireRemoteTokenBound: true},
		{Inputs: []string{"hi"}, Purpose: "bad"},
		{Inputs: []string{"long text"}, Purpose: embedding.Query},
		{Inputs: []string{"a", "b"}, Purpose: embedding.Query},
		{Inputs: []string{" "}, Purpose: embedding.Query},
	}

	for _, request := range cases {
		if _, err := c.Embed(context.Background(), request); err == nil {
			t.Fatal("accepted unsupported request")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Embed(
		ctx,
		rootdense.Request{Inputs: []string{"hi"}, Purpose: embedding.Query},
	); !errors.Is(
		err,
		context.Canceled,
	) {
		t.Fatal(err)
	}
	if calls.Load() != 0 {
		t.Fatal("dispatched invalid request")
	}
}
func TestBoundsAndSanitizedErrors(t *testing.T) {
	for _, status := range []int{302, 400, 429, 503, 200} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.Header().Set("Location", "/leak")
				w.WriteHeader(status)
				_, _ = w.Write([]byte(strings.Repeat("secret-private-input", 100)))
			}))
			defer server.Close()
			c := testClient(t, server.URL, embedding.Limits{MaxResponseBytes: 10})
			_, err := c.Embed(
				context.Background(),
				rootdense.Request{Inputs: []string{"private-input"}, Purpose: embedding.Query},
			)
			if err == nil || strings.Contains(err.Error(), "secret") ||
				strings.Contains(err.Error(), "private-input") ||
				strings.Contains(err.Error(), server.URL) {
				t.Fatalf("unsafe error %v", err)
			}
			if calls.Load() != 1 {
				t.Fatal("retry or redirect")
			}
		})
	}
}
func TestTimeout(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) { <-release }))
	defer server.Close()
	defer close(release)
	c := testClient(t, server.URL, embedding.Limits{Timeout: 10 * time.Millisecond})
	_, err := c.Embed(context.Background(), rootdense.Request{Inputs: []string{"hi"}, Purpose: embedding.Query})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
}
func TestInvalidConfiguration(t *testing.T) {
	space := testSpace()
	space.Model = "unsupported"
	if _, err := New(Config{APIKey: "key", Space: space}); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal(err)
	}
	if _, err := New(
		Config{APIKey: "key", Model: "mismatch", Space: testSpace()},
	); !errors.Is(
		err,
		ragy.ErrInvalidArgument,
	) {
		t.Fatal(err)
	}
	if _, err := New(Config{Space: testSpace()}); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal(err)
	}
}

func wireHandler(t *testing.T, purpose embedding.Purpose) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/embeddings" ||
			r.Header.Get("Authorization") != "Bearer secret" ||
			r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("wrong HTTP request")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["model"] != "jina-embeddings-v3" || body["dimensions"] != float64(2) {
			t.Errorf("wrong profile: %v", body)
		}
		if inputs, ok := body["input"].([]any); !ok || len(inputs) != 2 || inputs[0] != "hello" ||
			inputs[1] != "world" {
			t.Errorf("wrong inputs: %v", body)
		}
		wantTask := "retrieval.query"
		if purpose == embedding.Document {
			wantTask = "retrieval.passage"
		}
		if body["task"] != wantTask || body["embedding_type"] != "float" || body["late_chunking"] != false {
			t.Errorf("wrong task: %v", body)
		}
		_, _ = w.Write(
			[]byte(
				`{"model":"jina-embeddings-v3","data":[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}],"usage":{"total_tokens":7}}`,
			),
		)
	}
}

func TestUnsupportedDimension(t *testing.T) {
	space := testSpace()
	space.Dimension = 1025
	if _, err := New(Config{APIKey: "key", Space: space}); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal(err)
	}
}
