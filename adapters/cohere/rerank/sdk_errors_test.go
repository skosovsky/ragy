package rerank

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/cohere-ai/cohere-go/v2/core"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
)

func TestSDKFailurePreservesTypeWithoutRetry(t *testing.T) {
	// Arrange.
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"message":"private failure"}`)
	}))
	t.Cleanup(server.Close)
	client, err := New[struct{}](Config{SDK: newSDK("fixture", server.URL, nil), Model: "fixture-model"})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, err = client.Rerank(context.Background(), retrieval.UnrestrictedRead(), "query", fixtureDocs())
	// Assert.
	if _, ok := errors.AsType[*core.APIError](err); !ok {
		t.Fatal("SDK error type lost", err)
	}
	if !errors.Is(err, ragy.ErrUnavailable) || calls.Load() != 1 {
		t.Fatal("unexpected class or retry", err, calls.Load())
	}
}

func TestSDKConcurrentReranks(t *testing.T) {
	// Arrange.
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, `{"results":[{"index":0,"relevance_score":0.2},{"index":1,"relevance_score":0.9}]}`)
	}))
	t.Cleanup(server.Close)
	client, err := New[struct{}](Config{SDK: newSDK("fixture", server.URL, nil), Model: "fixture-model"})
	if err != nil {
		t.Fatal(err)
	}
	problems := make(chan error, 8)
	var wait sync.WaitGroup
	// Act.
	for range 8 {
		wait.Go(func() {
			result, callErr := client.Rerank(context.Background(), retrieval.UnrestrictedRead(), "query", fixtureDocs())
			if callErr != nil || result.Len() != 2 {
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
	if calls.Load() != 8 {
		t.Fatal("unexpected SDK attempts", calls.Load())
	}
}
