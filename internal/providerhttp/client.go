// Package providerhttp implements bounded JSON transport for embedding providers.
package providerhttp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
)

// Doer performs one exchange. Custom implementations must not retry or follow
// redirects internally and must honor request context cancellation.
type Doer interface {
	Do(*http.Request) (*http.Response, error)
}

// Config provides transport and finite local limits. BaseURL cannot contain
// credentials, query parameters or fragments; credentials belong in headers.
type Config struct {
	BaseURL    string
	HTTPClient Doer
	Limits     embedding.Limits
}

// Client is immutable and safe for concurrent requests when its Doer is safe.
type Client struct {
	baseURL   string
	transport Doer
	limits    embedding.Limits
}

// New validates configuration and prevents redirects for standard HTTP clients.
func New(cfg Config) (*Client, error) {
	limits, err := cfg.Limits.Resolve()
	if err != nil {
		return nil, err
	}
	base, err := url.Parse(cfg.BaseURL)
	if err != nil || base.Host == "" || (base.Scheme != "http" && base.Scheme != "https") || base.User != nil ||
		base.RawQuery != "" ||
		base.ForceQuery ||
		base.Fragment != "" ||
		base.Opaque != "" {
		return nil, ragy.ErrInvalidArgument
	}
	transport := cfg.HTTPClient
	if transport == nil {
		transport = &http.Client{CheckRedirect: rejectRedirect}
	}
	if standard, ok := transport.(*http.Client); ok {
		if standard == nil {
			return nil, ragy.ErrInvalidArgument
		}
		clone := *standard
		clone.CheckRedirect = rejectRedirect
		transport = &clone
	}
	return &Client{baseURL: strings.TrimRight(base.String(), "/"), transport: transport, limits: limits}, nil
}

func rejectRedirect(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }

// Limits returns resolved finite limits.
func (c *Client) Limits() embedding.Limits { return c.limits }

// ValidateTexts validates all inputs before a provider request is marshaled.
func (c *Client) ValidateTexts(ctx context.Context, texts []string) error {
	if contextErr := ctx.Err(); contextErr != nil {
		return contextErr
	}
	if len(texts) == 0 || len(texts) > c.limits.MaxInputs {
		return ragy.ErrInvalidArgument
	}
	total := 0
	for _, text := range texts {
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(text) > c.limits.MaxInputBytes-total || !utf8.ValidString(text) || strings.TrimSpace(text) == "" {
			return ragy.ErrInvalidArgument
		}
		total += len(text)
	}
	return ctx.Err()
}

// Post performs exactly one bounded exchange and decodes exactly one JSON
// object. Unknown fields are allowed for provider envelope evolution. Response
// bodies, credentials and raw transport/decode errors never enter returned errors.
func (c *Client) Post(ctx context.Context, path string, headers http.Header, body any, out any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !validPath(path) || out == nil {
		return ragy.ErrInvalidArgument
	}
	requestBody, err := json.Marshal(body)
	if err != nil {
		return ragy.ErrInvalidArgument
	}
	if contextErr := ctx.Err(); contextErr != nil {
		return contextErr
	}
	if int64(len(requestBody)) > c.limits.MaxRequestBytes {
		return ragy.ErrInvalidArgument
	}
	requestCtx, cancel := context.WithTimeout(ctx, c.limits.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, c.baseURL+path, bytes.NewReader(requestBody))
	if err != nil {
		return ragy.ErrInvalidArgument
	}
	req.Header = headers.Clone()
	if req.Header == nil {
		req.Header = make(http.Header)
	}
	req.Header.Set("Content-Type", "application/json")
	if contextErr := requestCtx.Err(); contextErr != nil {
		return contextErr
	}
	return c.exchange(requestCtx, req, out)
}

func (c *Client) exchange(requestCtx context.Context, req *http.Request, out any) error {
	resp, err := c.transport.Do(req)
	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}
	if err != nil {
		return sanitizedError(requestCtx, err, ragy.ErrUnavailable)
	}
	if contextErr := requestCtx.Err(); contextErr != nil {
		return contextErr
	}
	if resp == nil || resp.Body == nil {
		return ragy.ErrProtocol
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return ragy.ErrorFromHTTPResponse(resp.StatusCode, "provider", "")
	}
	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, c.limits.MaxResponseBytes+1))
	if err != nil {
		return sanitizedError(requestCtx, err, ragy.ErrProtocol)
	}
	if err := requestCtx.Err(); err != nil {
		return err
	}
	if int64(len(responseBody)) > c.limits.MaxResponseBytes {
		return ragy.ErrProtocol
	}
	if err := decodeObject(responseBody, out); err != nil {
		return err
	}
	return requestCtx.Err()
}

func validPath(path string) bool {
	if !strings.HasPrefix(path, "/") || strings.HasPrefix(path, "//") {
		return false
	}
	parsed, err := url.Parse(path)
	return err == nil && parsed.Scheme == "" && parsed.Host == "" && parsed.User == nil && parsed.RawQuery == "" &&
		!parsed.ForceQuery &&
		parsed.Fragment == ""
}

func sanitizedError(ctx context.Context, err, errorClass error) error {
	if contextErr := ctx.Err(); contextErr != nil {
		return contextErr
	}
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}
	return errorClass
}

func decodeObject(body []byte, out any) error {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return ragy.ErrProtocol
	}
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	if err := decoder.Decode(out); err != nil {
		return ragy.ErrProtocol
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return ragy.ErrProtocol
	}
	return nil
}
