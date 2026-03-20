// Package cohere provides a ragy.Reranker using the Cohere Rerank API (e.g. rerank-multilingual-v3).
// Retry and backoff are the caller's responsibility.
package cohere

import (
	"context"
	"fmt"
	"sort"

	coheregov2 "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/skosovsky/ragy"
)

const (
	defaultModel    = "rerank-multilingual-v3"
	cohereBatchSize = 50
)

// Reranker implements ragy.Reranker using Cohere Rerank API.
type Reranker struct {
	client *cohereclient.Client
	model  string
}

// Option configures the Reranker.
type Option func(*Reranker)

// WithModel sets the model name (e.g. "rerank-multilingual-v3").
func WithModel(m string) Option {
	return func(r *Reranker) { r.model = m }
}

// New returns a new Cohere Reranker. Token is the Cohere API key.
func New(token string, opts ...Option) *Reranker {
	r := &Reranker{
		client: cohereclient.NewClient(cohereclient.WithToken(token)),
		model:  defaultModel,
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

// NewWithClient returns a Reranker using an existing Cohere client.
func NewWithClient(client *cohereclient.Client, opts ...Option) *Reranker {
	r := &Reranker{client: client, model: defaultModel}
	for _, o := range opts {
		o(r)
	}
	return r
}

// Rerank implements ragy.Reranker. Batches docs in chunks of 50 (API limit), then concatenates and sorts by score (absolute scores 0–1), returns topK.
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
		scored, err := r.rerankBatch(ctx, query, batch)
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

func (r *Reranker) rerankBatch(ctx context.Context, query string, batch []ragy.Document) ([]ragy.Document, error) {
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
	if err != nil {
		return nil, fmt.Errorf("cohere rerank: %w", err)
	}
	out := make([]ragy.Document, 0, len(resp.Results))
	for _, result := range resp.Results {
		idx := result.Index
		if idx >= 0 && idx < len(batch) {
			doc := batch[idx]
			score, conf := cohereRelevanceToDocScores(result.RelevanceScore)
			doc.Score = score
			doc.Confidence = conf
			out = append(out, doc)
		}
	}
	return out, nil
}

// cohereRelevanceToDocScores maps Cohere RelevanceScore (expected in [0,1]) to Document.Score and Document.Confidence.
func cohereRelevanceToDocScores(relevance float64) (score float32, confidence float64) {
	v := relevance
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return float32(v), v
}

var _ ragy.Reranker = (*Reranker)(nil)
