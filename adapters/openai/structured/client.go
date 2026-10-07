// Package structured provides bounded, single-dispatch structured completions.
// Domain schemas, validation, tokenization and prices belong to the host.
package structured

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/providerhttp"
	"github.com/skosovsky/ragy/recipe/budget"
)

var (
	ErrRefused    = errors.New("structured completion refused")
	ErrIncomplete = errors.New("structured completion incomplete")
)

// Config uses a host-supplied executable schema validator and exact tokenizer.
// Callbacks must be pure, bounded and safe for concurrent use. CountTokens counts
// the complete serialized provider request, including instructions and schema.
type Config struct {
	APIKey           string
	Model            string
	BaseURL          string
	HTTPClient       *http.Client
	Instructions     string
	SchemaName       string
	Schema           json.RawMessage
	MaxRequestBytes  int
	MaxResponseBytes int
	Duration         time.Duration
	Validate         func(json.RawMessage) error
	CountTokens      func([]byte) (uint64, error)
}

type Client[T any] struct {
	config   Config
	http     *http.Client
	endpoint string
}

type Limits struct {
	InputTokens  uint64
	OutputTokens uint64
}
type Usage struct {
	InputTokens  uint64
	OutputTokens uint64
	Known        bool
}

func New[T any](cfg Config) (*Client[T], error) {
	if providerhttp.ValidateAPIKey(cfg.APIKey) != nil ||
		strings.TrimSpace(cfg.Model) == "" || cfg.Instructions == "" || !validName(cfg.SchemaName) ||
		cfg.MaxRequestBytes <= 0 || cfg.MaxResponseBytes <= 0 || cfg.MaxResponseBytes == math.MaxInt || cfg.Duration <= 0 ||
		cfg.Validate == nil || cfg.CountTokens == nil {
		return nil, ragy.ErrInvalidArgument
	}
	if !validSchema(cfg.Schema, cfg.MaxRequestBytes) {
		return nil, ragy.ErrInvalidArgument
	}
	base := cfg.BaseURL
	if base == "" {
		base = "https://api.openai.com/v1"
	}
	endpoint, err := providerhttp.Endpoint(base, "/chat/completions")
	if err != nil {
		return nil, err
	}
	client := http.Client{}
	if cfg.HTTPClient != nil {
		client = *cfg.HTTPClient
	}
	// Redirects cannot dispatch a second call or forward credentials elsewhere.
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	cfg.Schema = bytes.Clone(cfg.Schema)
	return &Client[T]{config: cfg, http: &client, endpoint: endpoint}, nil
}

func validSchema(schema json.RawMessage, limit int) bool {
	trimmed := bytes.TrimSpace(schema)
	return len(schema) <= limit && len(trimmed) > 0 && trimmed[0] == '{' && validJSON(schema) == nil
}

func validName(value string) bool {
	if len(value) == 0 || len(value) > 64 {
		return false
	}
	for _, char := range value {
		if (char < 'a' || char > 'z') && (char < 'A' || char > 'Z') &&
			(char < '0' || char > '9') && char != '_' && char != '-' {
			return false
		}
	}
	return true
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type schemaFormat struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict"`
	Schema json.RawMessage `json:"schema"`
}
type responseFormat struct {
	Type       string       `json:"type"`
	JSONSchema schemaFormat `json:"json_schema"`
}
type request struct {
	Model               string         `json:"model"`
	Messages            []message      `json:"messages"`
	ResponseFormat      responseFormat `json:"response_format"`
	MaxCompletionTokens uint64         `json:"max_completion_tokens"`
	Store               bool           `json:"store"`
}

func (c *Client[T]) body(input any, outputTokens uint64) ([]byte, error) {
	if c == nil || outputTokens == 0 || outputTokens > math.MaxInt64 {
		return nil, ragy.ErrInvalidArgument
	}
	content, err := json.Marshal(input)
	if err != nil {
		return nil, ragy.ErrInvalidArgument
	}
	if len(content) > c.config.MaxRequestBytes {
		return nil, ragy.ErrInvalidArgument
	}
	body, err := json.Marshal(request{
		Model:    c.config.Model,
		Messages: []message{{Role: "system", Content: c.config.Instructions}, {Role: "user", Content: string(content)}},
		ResponseFormat: responseFormat{Type: "json_schema", JSONSchema: schemaFormat{
			Name: c.config.SchemaName, Strict: true, Schema: c.config.Schema,
		}},
		MaxCompletionTokens: outputTokens, Store: false,
	})
	if err != nil || len(body) > c.config.MaxRequestBytes {
		return nil, ragy.ErrInvalidArgument
	}
	return body, nil
}

// CountInputTokens performs no I/O. The host tokenizer must account for the exact
// model's message framing and schema, rather than approximating by string length.
func (c *Client[T]) CountInputTokens(input any, outputTokens uint64) (uint64, error) {
	body, err := c.body(input, outputTokens)
	if err != nil {
		return 0, err
	}
	tokens, err := c.config.CountTokens(body)
	if err != nil || tokens == 0 {
		return 0, ragy.ErrInvalidArgument
	}
	return tokens, nil
}

// Call sends one POST. Usage remains available on refusal, truncation, validation
// failure and token overrun. Errors never contain request, response or credentials.
func (c *Client[T]) Call(ctx context.Context, input any, limits Limits) (T, Usage, error) {
	return c.call(ctx, input, limits, nil)
}

// call admits immediately before transport, after the host token counter. A
// policy change inside that callback must prevent sending admitted source text.
func (c *Client[T]) call(
	ctx context.Context,
	input any,
	limits Limits,
	admit func(context.Context) error,
) (T, Usage, error) {
	var empty T
	if ctx == nil || limits.InputTokens == 0 {
		return empty, Usage{}, ragy.ErrInvalidArgument
	}
	if c == nil {
		return empty, Usage{}, ragy.ErrInvalidArgument
	}
	ctx, cancel := context.WithTimeout(ctx, c.config.Duration)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return empty, Usage{}, err
	}
	body, err := c.body(input, limits.OutputTokens)
	if err != nil {
		return empty, Usage{}, err
	}
	tokens, err := c.config.CountTokens(bytes.Clone(body))
	if err != nil {
		return empty, Usage{}, ragy.ErrInvalidArgument
	}
	if tokens == 0 || tokens > limits.InputTokens {
		return empty, Usage{}, budget.ErrExhausted
	}
	if err = ctx.Err(); err != nil {
		return empty, Usage{}, err
	}
	if err = checkAdmission(ctx, admit); err != nil {
		return empty, Usage{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return empty, Usage{}, ragy.ErrInvalidArgument
	}
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return c.exchange(ctx, req, limits)
}

func (c *Client[T]) exchange(ctx context.Context, req *http.Request, limits Limits) (T, Usage, error) {
	var empty T
	resp, err := c.http.Do(req)
	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}
	if err != nil {
		return empty, Usage{}, providerhttp.SanitizedError(ctx, err, ragy.ErrUnavailable)
	}
	if err = ctx.Err(); err != nil {
		return empty, Usage{}, err
	}
	if resp == nil || resp.Body == nil {
		return empty, Usage{}, ragy.ErrProtocol
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return empty, Usage{}, ragy.ErrorFromHTTPResponse(resp.StatusCode, "openai structured", "")
	}
	payload, err := io.ReadAll(io.LimitReader(resp.Body, int64(c.config.MaxResponseBytes)+1))
	if err != nil {
		return empty, Usage{}, providerhttp.SanitizedError(ctx, err, ragy.ErrProtocol)
	}
	if err = ctx.Err(); err != nil {
		return empty, Usage{}, err
	}
	if len(payload) > c.config.MaxResponseBytes {
		return empty, Usage{}, ragy.ErrProtocol
	}
	output, usage, err := c.decode(payload, limits)
	if ctx.Err() != nil {
		return empty, usage, ctx.Err()
	}
	return output, usage, err
}

func checkAdmission(ctx context.Context, admit func(context.Context) error) error {
	if admit == nil {
		return nil
	}
	return admit(ctx)
}
