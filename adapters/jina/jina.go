// Package jina provides ragy.TensorEmbedder (ColBERT v2) and ragy.DenseEmbedder (jina-embeddings-v3) using the Jina API.
package jina

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/skosovsky/ragy"
)

const (
	defaultBaseURL   = "https://api.jina.ai"
	defaultModel     = "jina-colbert-v2"
	defaultBatchSize = 32
)

// Embedder implements ragy.TensorEmbedder using Jina multi-vector (ColBERT) API.
type Embedder struct {
	client    *http.Client
	baseURL   string
	apiKey    string
	model     string
	batchSize int
}

// Option configures the Embedder.
type Option func(*Embedder)

// WithBaseURL sets the API base URL (for tests use httptest.Server.URL).
func WithBaseURL(url string) Option {
	return func(e *Embedder) { e.baseURL = url }
}

// WithModel sets the model name (e.g. "jina-colbert-v2").
func WithModel(m string) Option {
	return func(e *Embedder) { e.model = m }
}

// WithBatchSize sets the max texts per request.
func WithBatchSize(n int) Option {
	return func(e *Embedder) { e.batchSize = n }
}

// WithHTTPClient sets the HTTP client.
func WithHTTPClient(c *http.Client) Option {
	return func(e *Embedder) { e.client = c }
}

// New returns a new Jina TensorEmbedder.
func New(apiKey string, opts ...Option) *Embedder {
	e := &Embedder{
		client:    http.DefaultClient,
		baseURL:   defaultBaseURL,
		apiKey:    apiKey,
		model:     defaultModel,
		batchSize: defaultBatchSize,
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

// EmbedTensors implements ragy.TensorEmbedder. Returns one token matrix per input text.
func (e *Embedder) EmbedTensors(ctx context.Context, texts []string) ([][][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += e.batchSize {
		end := min(start+e.batchSize, len(texts))
		batch := texts[start:end]
		matrices, err := e.embedBatch(ctx, batch)
		if err != nil {
			return nil, err
		}
		out = append(out, matrices...)
	}
	return out, nil
}

type statusError struct {
	status int
	err    error
}

func (e *statusError) Error() string { return e.err.Error() }
func (e *statusError) Unwrap() error { return e.err }

// jinaMultiVectorResponse matches the multi-vector API response: data[].embedding is [][]float32 per input.
type jinaMultiVectorResponse struct {
	Data []struct {
		Index     int         `json:"index"`
		Embedding [][]float32 `json:"embedding"`
	} `json:"data"`
}

func (e *Embedder) embedBatch(ctx context.Context, texts []string) ([][][]float32, error) {
	body := map[string]any{
		"model": e.model,
		"input": texts,
	}
	reqBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	url := e.baseURL + "/v1/multi-vector"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	slurp, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, &statusError{status: resp.StatusCode, err: fmt.Errorf("jina api: %s", string(slurp))}
	}
	var parsed jinaMultiVectorResponse
	if err := json.Unmarshal(slurp, &parsed); err != nil {
		return nil, err
	}
	out := make([][][]float32, len(texts))
	for i := range parsed.Data {
		if i >= len(out) {
			break
		}
		d := &parsed.Data[i]
		idx := d.Index
		if idx >= 0 && idx < len(out) {
			out[idx] = d.Embedding
		} else {
			out[i] = d.Embedding
		}
	}
	return out, nil
}

var _ ragy.TensorEmbedder = (*Embedder)(nil)

const (
	denseModelDefault = "jina-embeddings-v3"
	denseBatchDefault = 64
)

// DenseEmbedder implements ragy.DenseEmbedder using the Jina embeddings v3 API (/v1/embeddings).
type DenseEmbedder struct {
	client    *http.Client
	baseURL   string
	apiKey    string
	model     string
	batchSize int
}

// DenseOption configures the DenseEmbedder.
type DenseOption func(*DenseEmbedder)

// DenseWithBaseURL sets the API base URL (for tests use httptest.Server.URL).
func DenseWithBaseURL(url string) DenseOption {
	return func(d *DenseEmbedder) { d.baseURL = url }
}

// DenseWithModel sets the model name (e.g. "jina-embeddings-v3").
func DenseWithModel(m string) DenseOption {
	return func(d *DenseEmbedder) { d.model = m }
}

// DenseWithBatchSize sets the max texts per request.
func DenseWithBatchSize(n int) DenseOption {
	return func(d *DenseEmbedder) { d.batchSize = n }
}

// DenseWithHTTPClient sets the HTTP client.
func DenseWithHTTPClient(c *http.Client) DenseOption {
	return func(d *DenseEmbedder) { d.client = c }
}

// NewDense returns a new Jina DenseEmbedder (jina-embeddings-v3).
func NewDense(apiKey string, opts ...DenseOption) *DenseEmbedder {
	d := &DenseEmbedder{
		client:    http.DefaultClient,
		baseURL:   defaultBaseURL,
		apiKey:    apiKey,
		model:     denseModelDefault,
		batchSize: denseBatchDefault,
	}
	for _, o := range opts {
		o(d)
	}
	return d
}

// Embed implements ragy.DenseEmbedder. Batches texts into HTTP requests.
func (d *DenseEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += d.batchSize {
		end := min(start+d.batchSize, len(texts))
		batch := texts[start:end]
		vecs, err := d.embedBatch(ctx, batch)
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

type jinaDenseResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (d *DenseEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	body := map[string]any{
		"model": d.model,
		"input": texts,
	}
	reqBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	url := d.baseURL + "/v1/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if d.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+d.apiKey)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	slurp, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, &statusError{status: resp.StatusCode, err: fmt.Errorf("jina embeddings api: %s", string(slurp))}
	}
	var parsed jinaDenseResponse
	if err := json.Unmarshal(slurp, &parsed); err != nil {
		return nil, err
	}
	out := make([][]float32, len(texts))
	for i := range parsed.Data {
		if i >= len(out) {
			break
		}
		item := &parsed.Data[i]
		idx := item.Index
		if idx >= 0 && idx < len(out) {
			out[idx] = item.Embedding
		} else {
			out[i] = item.Embedding
		}
	}
	return out, nil
}

var _ ragy.DenseEmbedder = (*DenseEmbedder)(nil)
