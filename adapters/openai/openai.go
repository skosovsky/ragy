// Package openai provides a ragy.DenseEmbedder implementation using the OpenAI Embeddings API.
// Supports batching and optional dimensions (text-embedding-3). Retry/backoff is the caller's responsibility.
package openai

import (
	"context"
	"fmt"
	"time"

	openaiapi "github.com/sashabaranov/go-openai"

	"github.com/skosovsky/ragy"
)

// DefaultBatchSize is the default number of texts sent per API request when batching.
const DefaultBatchSize = 100

// defaultRequestTimeout is the default per-request timeout when none is set via WithRequestTimeout.
const defaultRequestTimeout = 30 * time.Second

// Embedder implements ragy.DenseEmbedder using the OpenAI Embeddings API.
type Embedder struct {
	client         *openaiapi.Client
	model          openaiapi.EmbeddingModel
	dimensions     int
	batchSize      int
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
		dimensions:     0,
		batchSize:      DefaultBatchSize,
		requestTimeout: defaultRequestTimeout,
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

// Embed implements ragy.DenseEmbedder. It batches texts per request.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += e.batchSize {
		end := min(start+e.batchSize, len(texts))
		batch := texts[start:end]
		vecs, err := e.embedOneBatch(ctx, batch)
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (e *Embedder) embedOneBatch(ctx context.Context, texts []string) ([][]float32, error) {
	reqCtx := ctx
	if e.requestTimeout > 0 {
		var cancel context.CancelFunc
		reqCtx, cancel = context.WithTimeout(ctx, e.requestTimeout)
		defer cancel()
	}
	return e.embedBatch(reqCtx, texts)
}

func (e *Embedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	req := openaiapi.EmbeddingRequestStrings{
		Input:          texts,
		Model:          e.model,
		User:           "",
		EncodingFormat: "",
		Dimensions:     e.dimensions,
		ExtraBody:      nil,
	}
	resp, err := e.client.CreateEmbeddings(ctx, req)
	if err != nil {
		return nil, err
	}
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
