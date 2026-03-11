// Package gemini provides ragy.DenseEmbedder and ragy.MultimodalEmbedder using the Google Gemini Embedding API.
// Supports text-embedding-004, optional TaskType, batching, rate limiting (RPM), and retry on 429.
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
	"math"
	"net/http"
	"time"

	"github.com/skosovsky/ragy"
	"golang.org/x/time/rate"
)

// DefaultBatchSize is the default number of texts per API request.
const DefaultBatchSize = 100

// DefaultBatchesPerMinute limits RPM to avoid hitting quota before 429.
const DefaultBatchesPerMinute = 30

// DefaultMaxRetries is the default number of retries on 429/5xx.
const DefaultMaxRetries = 5

// Embedder implements ragy.DenseEmbedder using the Gemini Embedding API.
type Embedder struct {
	client         *http.Client
	baseURL        string
	apiKey         string
	model          string
	taskType       string
	batchSize      int
	limiter        *rate.Limiter
	maxRetries     int
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

// WithMaxRetries sets retries on 429/5xx.
func WithMaxRetries(n int) Option {
	return func(e *Embedder) { e.maxRetries = n }
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
		batchSize:      DefaultBatchSize,
		limiter:        rate.NewLimiter(rate.Every(time.Minute/time.Duration(DefaultBatchesPerMinute)), 1),
		maxRetries:     DefaultMaxRetries,
		requestTimeout: 30 * time.Second,
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

// Embed implements ragy.DenseEmbedder. Batches texts, applies rate limit, and retries on 429.
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

// EmbedMultimodal implements ragy.MultimodalEmbedder. len(texts) must equal len(media).
// Empty media[i] means text-only embedding for that index. Uses same rate limit and retry as Embed.
func (e *Embedder) EmbedMultimodal(ctx context.Context, texts []string, media [][]ragy.Media) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if len(media) != len(texts) {
		return nil, fmt.Errorf("gemini: len(media)=%d must equal len(texts)=%d", len(media), len(texts))
	}
	out := make([][]float32, 0, len(texts))
	for i := range texts {
		vec, err := e.embedOneMultimodalWithRetry(ctx, texts[i], media[i])
		if err != nil {
			return nil, err
		}
		out = append(out, vec)
	}
	return out, nil
}

func (e *Embedder) embedOneMultimodalWithRetry(ctx context.Context, text string, media []ragy.Media) ([]float32, error) {
	var lastErr error
	for attempt := 0; attempt <= e.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(math.Pow(2, float64(attempt-1))) * 100 * time.Millisecond
			backoff += backoff / 4
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}
		if e.requestTimeout > 0 {
			reqCtx, cancel := context.WithTimeout(ctx, e.requestTimeout)
			vec, err := e.embedOneMultimodal(reqCtx, text, media)
			cancel()
			if err == nil {
				return vec, nil
			}
			lastErr = err
			if !isRetryable(err) {
				return nil, err
			}
			continue
		}
		vec, err := e.embedOneMultimodal(ctx, text, media)
		if err != nil {
			lastErr = err
			if !isRetryable(err) {
				return nil, err
			}
			continue
		}
		return vec, nil
	}
	return nil, fmt.Errorf("gemini embed multimodal after %d retries: %w", e.maxRetries, lastErr)
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
		return nil, fmt.Errorf("gemini: at least one of text or media must be non-empty")
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
	//nolint:gosec // G704: URL is from config/constant, not user input
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	slurp, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, &errWithStatus{status: resp.StatusCode, err: fmt.Errorf("gemini api: %s", string(slurp))}
	}
	var parsed embedResponse
	if err := json.Unmarshal(slurp, &parsed); err != nil {
		return nil, err
	}
	return parsed.Embedding.Values, nil
}

func (e *Embedder) embedBatchWithRetry(ctx context.Context, texts []string) ([][]float32, error) {
	var lastErr error
	for attempt := 0; attempt <= e.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(math.Pow(2, float64(attempt-1))) * 100 * time.Millisecond
			backoff += backoff / 4
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}
		if e.requestTimeout > 0 {
			reqCtx, cancel := context.WithTimeout(ctx, e.requestTimeout)
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
		vecs, err := e.embedBatch(ctx, texts)
		if err != nil {
			lastErr = err
			if !isRetryable(err) {
				return nil, err
			}
			continue
		}
		return vecs, nil
	}
	return nil, fmt.Errorf("gemini embed batch after %d retries: %w", e.maxRetries, lastErr)
}

// errWithStatus carries HTTP status for retry logic.
type errWithStatus struct {
	status int
	err    error
}

func (e *errWithStatus) Error() string { return e.err.Error() }
func (e *errWithStatus) Unwrap() error { return e.err }

func isRetryable(err error) bool {
	var es *errWithStatus
	if errors.As(err, &es) {
		return es.status == 429 || (es.status >= 500 && es.status < 600)
	}
	return false
}

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
		//nolint:gosec // G704: URL is from config/constant, not user input
		resp, err := e.client.Do(req)
		if err != nil {
			return nil, err
		}
		slurp, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, &errWithStatus{status: resp.StatusCode, err: fmt.Errorf("gemini api: %s", string(slurp))}
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
