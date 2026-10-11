// Command retry_embedder demonstrates retrying dense.Embedder.Embed on transient errors (HTTP 429).
package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/openai/dense"
	rootdense "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

const (
	successAfterAttempts = 3
	maxRetryAttempts     = 5
	retryBackoff         = 10 * time.Millisecond
)

func main() {
	var attempts int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		attempts++
		if attempts < successAfterAttempts {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"rate_limited","type":"rate_limit_error"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.25,0.75]}]}`))
	}))
	defer server.Close()

	sdk := openai.NewClient(option.WithAPIKey("example"), option.WithBaseURL(server.URL), option.WithMaxRetries(0))
	emb, err := dense.New(dense.Config{
		SDK:   &sdk,
		Model: "text-embedding-3-small",
		Space: embedding.Space{
			Model:         "text-embedding-3-small",
			ModelRevision: "host-v1",
			Configuration: "no-preprocessing",
			VectorSpace:   "example",
			Dimension:     2,
			Metric:        embedding.Dot,
		},
	})
	if err != nil {
		panic(err)
	}

	wrapped := retryEmbedder{inner: emb, max: maxRetryAttempts, backoff: retryBackoff}

	ctx := context.Background()
	encoded, err := wrapped.Embed(ctx, rootdense.Request{Inputs: []string{"hello"}, Purpose: embedding.Query})
	if err != nil {
		panic(err)
	}
	if len(encoded.Embeddings) != 1 || len(encoded.Embeddings[0].Vector) != 2 {
		panic("unexpected embedding shape")
	}
	fmt.Printf("ok after %d server hits: dim=%d\n", attempts, len(encoded.Embeddings[0].Vector))
}

type retryEmbedder struct {
	inner   rootdense.Embedder
	max     int
	backoff time.Duration
}

func (r retryEmbedder) Space() rootdense.Space { return r.inner.Space() }

func (r retryEmbedder) Embed(ctx context.Context, texts rootdense.Request) (rootdense.Result, error) {
	var last error
	for range r.max {
		out, err := r.inner.Embed(ctx, texts)
		if err == nil {
			return out, nil
		}
		if !errors.Is(err, ragy.ErrUnavailable) {
			return rootdense.Result{}, err
		}
		last = err
		select {
		case <-ctx.Done():
			return rootdense.Result{}, ctx.Err()
		case <-time.After(r.backoff):
		}
	}
	return rootdense.Result{}, last
}
