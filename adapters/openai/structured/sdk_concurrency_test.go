package structured_test

import (
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
	"github.com/skosovsky/ragy/adapters/openai/structured"
)

func TestSDKConcurrentCallsAndProviderType(t *testing.T) {
	// Arrange.
	var calls atomic.Int32
	var failing atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if failing.Load() {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"private failure"}}`)
			return
		}
		_, _ = io.WriteString(w, envelope(`{"value":1}`, "stop"))
	}))
	t.Cleanup(server.Close)
	client, err := structured.New[payload](config(server.URL))
	if err != nil {
		t.Fatal(err)
	}
	problems := make(chan error, 8)
	var wait sync.WaitGroup
	// Act.
	for range 8 {
		wait.Go(func() {
			result, usage, callErr := client.Call(
				t.Context(),
				"text",
				structured.Limits{InputTokens: 100, OutputTokens: 10},
			)
			if callErr != nil || result.Value != "1" || !usage.Known {
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
	failing.Store(true)
	_, _, err = client.Call(t.Context(), "text", structured.Limits{InputTokens: 100, OutputTokens: 10})
	if _, ok := errors.AsType[*openai.Error](err); !ok {
		t.Fatal("SDK type lost", err)
	}
	if !errors.Is(err, ragy.ErrUnavailable) || calls.Load() != 9 {
		t.Fatal("retry attempts", calls.Load(), err)
	}
}
