// Package gemini provides ragy.DenseEmbedder and ragy.MultimodalEmbedder using the Google Gemini Embedding API.
// Supports text-embedding-004, optional TaskType, batching, and optional rate limiting (RPM). Retry is the caller's responsibility.
// MultimodalEmbedder supports text + image (Media) for embedding; empty media[i] means text-only.
package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"golang.org/x/time/rate"

	"github.com/skosovsky/ragy"
)

// DefaultBatchSize is the default number of texts per API request.
const DefaultBatchSize = 100

// DefaultBatchesPerMinute limits RPM to smooth traffic (optional).
const DefaultBatchesPerMinute = 30

// defaultRequestTimeout is the per-request HTTP timeout when not overridden.
const defaultRequestTimeout = 30 * time.Second

// Embedder implements ragy.DenseEmbedder using the Gemini Embedding API.
type Embedder struct {
	client         *http.Client
	baseURL        string
	apiKey         string
	model          string
	taskType       string
	batchSize      int
	limiter        *rate.Limiter
	requestTimeout time.Duration
}

// Option configures the Embedder.
type Option func(*Embedder)

// WithBaseURL sets the API base URL (for tests use httptest.Server.URL).
func WithBaseURL(url string) Option {
	return func(e *Embedder) { e.baseURL = url }
}

// WithModel sets the model name (e.g. "text-embedding-004").
func WithModel(m string) Option {
	return func(e *Embedder) { e.model = m }
}

// WithTaskType sets the task type (e.g. "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY").
func WithTaskType(t string) Option {
	return func(e *Embedder) { e.taskType = t }
}

// WithBatchSize sets the max texts per request.
func WithBatchSize(n int) Option {
	return func(e *Embedder) { e.batchSize = n }
}

// WithBatchesPerMinute sets the rate limiter (RPM) to smooth traffic.
func WithBatchesPerMinute(rpm int) Option {
	return func(e *Embedder) {
		e.limiter = rate.NewLimiter(rate.Every(time.Minute/time.Duration(rpm)), 1)
	}
}

// WithRequestTimeout sets timeout per request.
func WithRequestTimeout(d time.Duration) Option {
	return func(e *Embedder) { e.requestTimeout = d }
}

// WithHTTPClient sets the HTTP client.
func WithHTTPClient(c *http.Client) Option {
	return func(e *Embedder) { e.client = c }
}

// New returns a new Gemini embedder. apiKey can be empty if using custom client with auth.
func New(apiKey string, opts ...Option) *Embedder {
	e := &Embedder{
		client:         http.DefaultClient,
		baseURL:        "https://generativelanguage.googleapis.com",
		apiKey:         apiKey,
		model:          "text-embedding-004",
		taskType:       "",
		batchSize:      DefaultBatchSize,
		limiter:        rate.NewLimiter(rate.Every(time.Minute/time.Duration(DefaultBatchesPerMinute)), 1),
		requestTimeout: defaultRequestTimeout,
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

// Embed implements ragy.DenseEmbedder. Batches texts and applies rate limit per request.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += e.batchSize {
		end := min(start+e.batchSize, len(texts))
		batch := texts[start:end]
		vecs, err := e.embedBatchWithOptionalTimeout(ctx, batch)
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (e *Embedder) embedBatchWithOptionalTimeout(ctx context.Context, texts []string) ([][]float32, error) {
	reqCtx := ctx
	if e.requestTimeout > 0 {
		var cancel context.CancelFunc
		reqCtx, cancel = context.WithTimeout(ctx, e.requestTimeout)
		defer cancel()
	}
	return e.embedBatch(reqCtx, texts)
}

// EmbedMultimodal implements ragy.MultimodalEmbedder. len(texts) must equal len(media).
// Empty media[i] means text-only embedding for that index.
func (e *Embedder) EmbedMultimodal(ctx context.Context, texts []string, media [][]ragy.Media) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if len(media) != len(texts) {
		return nil, fmt.Errorf("gemini: len(media)=%d must equal len(texts)=%d", len(media), len(texts))
	}
	out := make([][]float32, 0, len(texts))
	for i := range texts {
		vec, err := e.embedOneMultimodalWithOptionalTimeout(ctx, texts[i], media[i])
		if err != nil {
			return nil, err
		}
		out = append(out, vec)
	}
	return out, nil
}

func (e *Embedder) embedOneMultimodalWithOptionalTimeout(
	ctx context.Context,
	text string,
	media []ragy.Media,
) ([]float32, error) {
	reqCtx := ctx
	if e.requestTimeout > 0 {
		var cancel context.CancelFunc
		reqCtx, cancel = context.WithTimeout(ctx, e.requestTimeout)
		defer cancel()
	}
	return e.embedOneMultimodal(reqCtx, text, media)
}

// embedOneMultimodal sends one embedContent request with optional text and inline media (images).
func (e *Embedder) embedOneMultimodal(ctx context.Context, text string, media []ragy.Media) ([]float32, error) {
	if err := e.limiter.Wait(ctx); err != nil {
		return nil, err
	}
	parts := make([]map[string]any, 0, 1+len(media))
	if text != "" {
		parts = append(parts, map[string]any{"text": text})
	}
	for _, m := range media {
		if m.MimeType == "" || len(m.Data) == 0 {
			continue
		}
		parts = append(parts, map[string]any{
			"inlineData": map[string]any{
				"mimeType": m.MimeType,
				"data":     base64.StdEncoding.EncodeToString(m.Data),
			},
		})
	}
	if len(parts) == 0 {
		return nil, errors.New("gemini: at least one of text or media must be non-empty")
	}
	body := map[string]any{
		"content": map[string]any{"parts": parts},
	}
	if e.taskType != "" {
		body["taskType"] = e.taskType
	}
	reqBody, _ := json.Marshal(body)
	url := e.baseURL + "/v1beta/models/" + e.model + ":embedContent"
	if e.apiKey != "" {
		url += "?key=" + e.apiKey
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	slurp, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, &apiStatusError{status: resp.StatusCode, err: fmt.Errorf("gemini api: %s", string(slurp))}
	}
	var parsed embedResponse
	if err := json.Unmarshal(slurp, &parsed); err != nil {
		return nil, err
	}
	return parsed.Embedding.Values, nil
}

// apiStatusError carries HTTP status from the Gemini API.
type apiStatusError struct {
	status int
	err    error
}

func (e *apiStatusError) Error() string { return e.err.Error() }
func (e *apiStatusError) Unwrap() error { return e.err }

// embedResponse matches Gemini embedContent response shape.
type embedResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
	} `json:"embedding"`
}

func (e *Embedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for _, text := range texts {
		if err := e.limiter.Wait(ctx); err != nil {
			return nil, err
		}
		body := map[string]any{
			"content": map[string]any{"parts": []map[string]any{{"text": text}}},
		}
		if e.taskType != "" {
			body["taskType"] = e.taskType
		}
		reqBody, _ := json.Marshal(body)
		url := e.baseURL + "/v1beta/models/" + e.model + ":embedContent"
		if e.apiKey != "" {
			url += "?key=" + e.apiKey
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := e.client.Do(req)
		if err != nil {
			return nil, err
		}
		slurp, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, &apiStatusError{status: resp.StatusCode, err: fmt.Errorf("gemini api: %s", string(slurp))}
		}
		var parsed embedResponse
		if err := json.Unmarshal(slurp, &parsed); err != nil {
			return nil, err
		}
		out = append(out, parsed.Embedding.Values)
	}
	return out, nil
}

// Ensure Embedder implements ragy.DenseEmbedder and ragy.MultimodalEmbedder.
var (
	_ ragy.DenseEmbedder      = (*Embedder)(nil)
	_ ragy.MultimodalEmbedder = (*Embedder)(nil)
)
