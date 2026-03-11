// Package cohere provides a ragy.Reranker using the Cohere Rerank API (e.g. rerank-multilingual-v3).
// Retries on 429 and 5xx with exponential backoff and jitter.
package cohere

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"

	coheregov2 "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/cohere-ai/cohere-go/v2/core"
	"github.com/skosovsky/ragy"
)

const (
	defaultModel      = "rerank-multilingual-v3"
	cohereBatchSize   = 50
	defaultMaxRetries = 5
)

// Reranker implements ragy.Reranker using Cohere Rerank API.
type Reranker struct {
	client     *cohereclient.Client
	model      string
	maxRetries int
}

// Option configures the Reranker.
type Option func(*Reranker)

// WithModel sets the model name (e.g. "rerank-multilingual-v3").
func WithModel(m string) Option {
	return func(r *Reranker) { r.model = m }
}

// WithMaxRetries sets the number of retries on 429/5xx (default 5).
func WithMaxRetries(n int) Option {
	return func(r *Reranker) { r.maxRetries = n }
}

// New returns a new Cohere Reranker. Token is the Cohere API key.
func New(token string, opts ...Option) *Reranker {
	r := &Reranker{
		client:     cohereclient.NewClient(cohereclient.WithToken(token)),
		model:      defaultModel,
		maxRetries: defaultMaxRetries,
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

// NewWithClient returns a Reranker using an existing Cohere client.
func NewWithClient(client *cohereclient.Client, opts ...Option) *Reranker {
	r := &Reranker{client: client, model: defaultModel, maxRetries: defaultMaxRetries}
	for _, o := range opts {
		o(r)
	}
	return r
}

// Rerank implements ragy.Reranker. Batches docs in chunks of 50 (API limit), then concatenates and sorts by score (absolute scores 0–1), returns topK.
// Retries each batch on 429/5xx with exponential backoff and jitter.
func (r *Reranker) Rerank(ctx context.Context, query string, docs []ragy.Document, topK int) ([]ragy.Document, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	var allScored []ragy.Document
	for start := 0; start < len(docs); start += cohereBatchSize {
		end := start + cohereBatchSize
		if end > len(docs) {
			end = len(docs)
		}
		batch := docs[start:end]
		scored, err := r.rerankBatchWithRetry(ctx, query, batch)
		if err != nil {
			return nil, err
		}
		allScored = append(allScored, scored...)
	}
	sort.Slice(allScored, func(i, j int) bool {
		return allScored[i].Score > allScored[j].Score
	})
	if topK <= 0 || topK > len(allScored) {
		topK = len(allScored)
	}
	return allScored[:topK], nil
}

func (r *Reranker) rerankBatchWithRetry(ctx context.Context, query string, batch []ragy.Document) ([]ragy.Document, error) {
	var lastErr error
	for attempt := 0; attempt <= r.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(math.Pow(2, float64(attempt-1))) * 100 * time.Millisecond
			backoff += backoff / 4
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}
		docItems := make([]*coheregov2.RerankRequestDocumentsItem, len(batch))
		for i := range batch {
			docItems[i] = &coheregov2.RerankRequestDocumentsItem{String: batch[i].Content}
		}
		topN := len(batch)
		request := &coheregov2.RerankRequest{
			Query:     query,
			Model:     &r.model,
			Documents: docItems,
			TopN:      &topN,
		}
		resp, err := r.client.Rerank(ctx, request)
		if err == nil {
			out := make([]ragy.Document, 0, len(resp.Results))
			for _, result := range resp.Results {
				idx := result.Index
				if idx >= 0 && idx < len(batch) {
					doc := batch[idx]
					doc.Score = float32(result.RelevanceScore)
					out = append(out, doc)
				}
			}
			return out, nil
		}
		lastErr = err
		if !isRetryable(err) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("cohere rerank after %d retries: %w", r.maxRetries, lastErr)
}

func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	var apiErr *core.APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == http.StatusTooManyRequests ||
			(apiErr.StatusCode >= http.StatusInternalServerError && apiErr.StatusCode < 600)
	}
	// Fallback for non-APIError (e.g. wrapped or transport errors) to avoid missing retryable cases
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "429") || strings.Contains(s, "too many requests") ||
		strings.Contains(s, "500") || strings.Contains(s, "502") || strings.Contains(s, "503")
}

var _ ragy.Reranker = (*Reranker)(nil)
