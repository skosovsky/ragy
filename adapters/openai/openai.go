// Package openai provides a ragy.DenseEmbedder implementation using the OpenAI Embeddings API.
// Supports batching, optional dimensions (text-embedding-3), and retry with backoff on 429.
package openai

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	openaiapi "github.com/sashabaranov/go-openai"
	"github.com/skosovsky/ragy"
)

// DefaultBatchSize is the default number of texts sent per API request when batching.
const DefaultBatchSize = 100

// DefaultMaxRetries is the default number of retries on 429/5xx.
const DefaultMaxRetries = 5

// Embedder implements ragy.DenseEmbedder using the OpenAI Embeddings API.
type Embedder struct {
	client         *openaiapi.Client
	model          openaiapi.EmbeddingModel
	dimensions     int
	batchSize      int
	maxRetries     int
	requestTimeout time.Duration
}

// Option configures the Embedder.
type Option func(*Embedder)

// WithModel sets the embedding model (e.g. openaiapi.SmallEmbedding3).
func WithModel(m openaiapi.EmbeddingModel) Option {
	return func(e *Embedder) { e.model = m }
}

// WithDimensions sets the optional dimensions for text-embedding-3 models.
func WithDimensions(d int) Option {
	return func(e *Embedder) { e.dimensions = d }
}

// WithBatchSize sets the max number of texts per API request.
func WithBatchSize(n int) Option {
	return func(e *Embedder) { e.batchSize = n }
}

// WithMaxRetries sets the max retries on 429/5xx.
func WithMaxRetries(n int) Option {
	return func(e *Embedder) { e.maxRetries = n }
}

// WithRequestTimeout sets the timeout for each API request.
func WithRequestTimeout(d time.Duration) Option {
	return func(e *Embedder) { e.requestTimeout = d }
}

// New returns a new OpenAI embedder. apiKey can be empty if the client config carries it.
// If client is nil, NewClient(apiKey) is used.
func New(apiKey string, opts ...Option) *Embedder {
	var client *openaiapi.Client
	if apiKey != "" {
		client = openaiapi.NewClient(apiKey)
	} else {
		client = openaiapi.NewClient("dummy")
	}
	return NewWithClient(client, opts...)
}

// NewWithClient returns a new Embedder using the given OpenAI client (e.g. for custom BaseURL/HTTPClient).
func NewWithClient(client *openaiapi.Client, opts ...Option) *Embedder {
	e := &Embedder{
		client:         client,
		model:          openaiapi.SmallEmbedding3,
		batchSize:      DefaultBatchSize,
		maxRetries:     DefaultMaxRetries,
		requestTimeout: 30 * time.Second,
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

// Embed implements ragy.DenseEmbedder. It batches texts and retries on 429 with exponential backoff.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += e.batchSize {
		end := start + e.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[start:end]
		vecs, err := e.embedBatchWithRetry(ctx, batch)
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (e *Embedder) embedBatchWithRetry(ctx context.Context, texts []string) ([][]float32, error) {
	var lastErr error
	for attempt := 0; attempt <= e.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(math.Pow(2, float64(attempt-1))) * 100 * time.Millisecond
			backoff += backoff / 4 // jitter: add 25% to spread retries
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}
		reqCtx := ctx
		if e.requestTimeout > 0 {
			var cancel context.CancelFunc
			reqCtx, cancel = context.WithTimeout(ctx, e.requestTimeout)
			vecs, err := e.embedBatch(reqCtx, texts)
			cancel()
			if err == nil {
				return vecs, nil
			}
			lastErr = err
			if !isRetryable(err) {
				return nil, err
			}
			continue
		}
		vecs, err := e.embedBatch(reqCtx, texts)
		if err != nil {
			lastErr = err
			if !isRetryable(err) {
				return nil, err
			}
			continue
		}
		return vecs, nil
	}
	return nil, fmt.Errorf("embed batch after %d retries: %w", e.maxRetries, lastErr)
}

func isRetryable(err error) bool {
	var apiErr *openaiapi.APIError
	if errors.As(err, &apiErr) {
		return apiErr.HTTPStatusCode == 429 || (apiErr.HTTPStatusCode >= 500 && apiErr.HTTPStatusCode < 600)
	}
	var reqErr *openaiapi.RequestError
	if errors.As(err, &reqErr) {
		return reqErr.HTTPStatusCode == 429 || (reqErr.HTTPStatusCode >= 500 && reqErr.HTTPStatusCode < 600)
	}
	return false
}

func (e *Embedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	req := openaiapi.EmbeddingRequestStrings{
		Input:      texts,
		Model:      e.model,
		User:       "",
		Dimensions: e.dimensions,
	}
	resp, err := e.client.CreateEmbeddings(ctx, req)
	if err != nil {
		return nil, err
	}
	// Preserve order by index (API returns data in order but we sort by index to be safe)
	vecs := make([][]float32, len(resp.Data))
	for i := range resp.Data {
		idx := resp.Data[i].Index
		if idx < 0 || idx >= len(vecs) {
			return nil, fmt.Errorf("openai: embedding index %d out of range", idx)
		}
		vecs[idx] = resp.Data[i].Embedding
	}
	return vecs, nil
}

// Ensure Embedder implements ragy.DenseEmbedder.
var _ ragy.DenseEmbedder = (*Embedder)(nil)
