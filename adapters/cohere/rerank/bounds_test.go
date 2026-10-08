package rerank

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/retrieval"
)

func fixtureDocs() retrieval.ResultSet[struct{}] {
	return retrieval.NewResultSet(
		[]retrieval.Document[struct{}]{{ID: "a", Content: "alpha"}, {ID: "b", Content: "beta"}},
		nil,
	)
}

// Official wire source: https://docs.cohere.com/v2/reference/rerank
// Verified 2026-10-06. Fixtures test protocol; they do not claim a live call.
func TestOfficialWireAndObservedSearchUnits(t *testing.T) {
	// Arrange.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v2/rerank" ||
			r.Header.Get("Authorization") != "Bearer fixture-key" ||
			r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("wrong request envelope")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if len(body) != 3 || body["model"] != "fixture-model" || body["query"] != "query" {
			t.Errorf("wrong body: %#v", body)
		}
		docs, ok := body["documents"].([]any)
		if !ok || len(docs) != 2 || docs[0] != "alpha" || docs[1] != "beta" {
			t.Error("wrong documents")
		}
		_, _ = io.WriteString(
			w,
			`{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}],"meta":{"billed_units":{"search_units":1}}}`,
		)
	}))
	defer server.Close()
	client, err := New[struct{}](Config{APIKey: "fixture-key", Model: "fixture-model", BaseURL: server.URL + "/v2"})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := client.RerankWithUsage(context.Background(), retrieval.UnrestrictedRead(), "query", fixtureDocs())
	// Assert.
	if err != nil || !result.Usage.BilledUnitsKnown || result.Usage.BilledUnits != 1 || result.Usage.InputTokensKnown ||
		result.Documents.Documents()[0].ID != "b" {
		t.Fatalf("result=%#v err=%v", result, err)
	}
}

func TestAdversarialResponse(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		valid      bool
		known      bool
	}{
		{"unknown-usage", `{"results":[{"index":0,"relevance_score":0},{"index":1,"relevance_score":1}]}`, true, false},
		{"known-zero", `{"results":[{"index":0,"relevance_score":0},{"index":1,"relevance_score":1}],"meta":{"billed_units":{"search_units":0}}}`, true, true},
		{"negative-usage", `{"results":[],"meta":{"billed_units":{"search_units":-1}}}`, false, false},
		{"wrong-model", `{"model":"other","results":[]}`, false, false},
		{"duplicate", `{"results":[{"index":0,"relevance_score":0},{"index":0,"relevance_score":1}]}`, false, false},
		{"missing-index", `{"results":[{"relevance_score":0},{"index":1,"relevance_score":1}]}`, false, false},
		{"missing-score", `{"results":[{"index":0},{"index":1,"relevance_score":1}]}`, false, false},
		{"out-of-range", `{"results":[{"index":-1,"relevance_score":0},{"index":1,"relevance_score":1}]}`, false, false},
		{"nonfinite", `{"results":[{"index":0,"relevance_score":1e1000},{"index":1,"relevance_score":1}]}`, false, false},
		{"refusal", `{"message":"secret"}`, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange.
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, tc.body) }),
			)
			defer server.Close()
			client, err := New[struct{}](Config{APIKey: "key", Model: "fixture-model", BaseURL: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			out, err := client.RerankWithUsage(
				context.Background(),
				retrieval.UnrestrictedRead(),
				"query",
				fixtureDocs(),
			)
			// Assert.
			if tc.valid {
				if err != nil || out.Usage.BilledUnitsKnown != tc.known {
					t.Fatalf("out=%#v err=%v", out, err)
				}
			} else if !errors.Is(err, ragy.ErrProtocol) || out.Documents.Len() != 2 {
				t.Fatalf("err=%v docs=%v", err, out.Documents)
			}
		})
	}
}

func TestLocalLimitsPreventDispatch(t *testing.T) {
	for _, limits := range []embedding.Limits{{MaxInputs: 2}, {MaxInputBytes: 12}, {MaxRequestBytes: 10}} {
		// Arrange.
		var calls atomic.Int64
		server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) { calls.Add(1) }))
		client, err := New[struct{}](Config{APIKey: "secret-key", Model: "model", BaseURL: server.URL, Limits: limits})
		if err != nil {
			t.Fatal(err)
		}
		// Act.
		out, err := client.Rerank(context.Background(), retrieval.UnrestrictedRead(), "query", fixtureDocs())
		// Assert.
		server.Close()
		if !errors.Is(err, ragy.ErrInvalidArgument) || calls.Load() != 0 || out.Len() != 2 {
			t.Fatalf("err=%v calls=%d", err, calls.Load())
		}
	}
}

func TestTransportBoundsAndSanitization(t *testing.T) {
	for _, mode := range []string{"oversized", "redirect", "error", "timeout", "canceled"} {
		t.Run(mode, func(t *testing.T) {
			// Arrange.
			var calls atomic.Int64
			released := make(chan struct{})
			server := httptest.NewServer(transportHandler(mode, &calls, released))
			defer server.Close()
			client, err := New[struct{}](
				Config{
					APIKey:  "secret-key",
					Model:   "model",
					BaseURL: server.URL,
					Limits:  embedding.Limits{MaxResponseBytes: 32, Timeout: 20 * time.Millisecond},
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if mode == "canceled" {
				cancel()
			}
			// Act.
			_, err = client.Rerank(ctx, retrieval.UnrestrictedRead(), "secret-input", fixtureDocs())
			close(released)
			// Assert.
			if err == nil || strings.Contains(err.Error(), "secret") || strings.Contains(err.Error(), server.URL) {
				t.Fatalf("unsafe error: %v", err)
			}
			want := int64(1)
			if mode == "canceled" {
				want = 0
			}
			if calls.Load() != want {
				t.Fatalf("calls=%d want=%d", calls.Load(), want)
			}
			assertTimeout(t, mode, err)
		})
	}
}

func transportHandler(mode string, calls *atomic.Int64, released <-chan struct{}) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		switch mode {
		case "oversized":
			_, _ = io.WriteString(w, strings.Repeat("secret-input", 100))
		case "redirect":
			http.Redirect(w, r, "/secret-key", http.StatusTemporaryRedirect)
		case "error":
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, "secret-key secret-input")
		case "timeout":
			<-released
		}
	}
}

func assertTimeout(t *testing.T, mode string, err error) {
	t.Helper()
	if mode == "timeout" && !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("timeout=%v", err)
	}
}
