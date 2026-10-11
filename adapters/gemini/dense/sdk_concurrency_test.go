package dense

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	ragy "github.com/skosovsky/ragy"
	root "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

func TestSDKConcurrentRequestsAndRetryAttempts(t *testing.T) {
	// Arrange: one real SDK shared by concurrent callers; HTTP status is fixture-owned.
	var calls atomic.Int32
	var failing atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if failing.Load() {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"private fixture"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"embeddings":[{"values":[1,2]}]}`)
	}))
	t.Cleanup(server.Close)
	client, err := New(Config{SDK: newSDK("fixture", server.URL, nil), Space: profile("gemini-embedding-001")})
	if err != nil {
		t.Fatal(err)
	}
	const workers = 8
	problems := make(chan error, workers)
	var wait sync.WaitGroup
	// Act.
	for range workers {
		wait.Go(func() {
			result, callErr := client.Embed(
				context.Background(),
				root.Request{Inputs: []string{"text"}, Purpose: embedding.Query},
			)
			if callErr != nil || len(result.Embeddings) != 1 {
				problems <- fmt.Errorf("concurrent result: %w", callErr)
			}
		})
	}
	wait.Wait()
	close(problems)
	// Assert.
	for problem := range problems {
		t.Error(problem)
	}
	if calls.Load() != workers {
		t.Fatal("unexpected SDK attempts", calls.Load())
	}
	failing.Store(true)
	_, err = client.Embed(context.Background(), root.Request{Inputs: []string{"text"}, Purpose: embedding.Query})
	if _, ok := errors.AsType[genai.APIError](err); !ok {
		t.Fatal("SDK error type lost", err)
	}
	if !errors.Is(err, ragy.ErrUnavailable) || calls.Load() != workers+1 {
		t.Fatal("retry attempts", calls.Load(), err)
	}
}

type failingSDKTransport struct {
	calls atomic.Int32
	cause error
}

func (t *failingSDKTransport) RoundTrip(*http.Request) (*http.Response, error) {
	t.calls.Add(1)
	return nil, t.cause
}

func TestSDKTransportFailureClassification(t *testing.T) {
	// Arrange: host-owned transport fails before any provider response exists.
	cause := errors.New("private host transport failure")
	transport := &failingSDKTransport{calls: atomic.Int32{}, cause: cause}
	client, err := New(Config{
		SDK:   newSDK("private-key", "https://fixture.invalid", &http.Client{Transport: transport}),
		Space: profile("gemini-embedding-001"),
	})
	if err != nil {
		t.Fatal(err)
	}
	// Act: call the actual public SDK through the adapter.
	result, err := client.Embed(context.Background(), root.Request{
		Inputs: []string{"private input"}, Purpose: embedding.Query,
	})
	// Assert: one attempt, safe classification and full SDK diagnostic chain.
	_, sdkCause := errors.AsType[*url.Error](err)
	if !errors.Is(err, ragy.ErrUnavailable) || errors.Is(err, ragy.ErrProtocol) ||
		!errors.Is(err, cause) || !sdkCause || transport.calls.Load() != 1 ||
		strings.Contains(err.Error(), "private") || len(result.Embeddings) != 0 || result.Usage.InputTokensKnown {
		t.Fatalf("result=%#v error=%v attempts=%d SDK cause=%v", result, err, transport.calls.Load(), sdkCause)
	}
}
