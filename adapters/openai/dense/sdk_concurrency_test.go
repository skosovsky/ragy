package dense

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

	"github.com/openai/openai-go/v3"

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
		_, _ = io.WriteString(w, `{"data":[{"index":0,"embedding":[1,2]}],"usage":{"prompt_tokens":7}}`)
	}))
	t.Cleanup(server.Close)
	client, err := New(Config{SDK: newSDK("fixture", server.URL, nil), Space: testSpace()})
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
	if _, ok := errors.AsType[*openai.Error](err); !ok {
		t.Fatal("SDK error type lost", err)
	}
	if !errors.Is(err, ragy.ErrUnavailable) || calls.Load() != workers+1 {
		t.Fatal("retry attempts", calls.Load(), err)
	}
}
