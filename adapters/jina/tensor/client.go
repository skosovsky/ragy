package tensor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	ragy "github.com/skosovsky/ragy"
	roottensor "github.com/skosovsky/ragy/tensor"
)

// DefaultBaseURL is the default Jina API endpoint.
const DefaultBaseURL = "https://api.jina.ai/v1"
const maxErrorBodyBytes = 4 << 10

// Doer executes HTTP requests.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// Config configures the Jina tensor adapter.
type Config struct {
	APIKey     string
	Model      string
	BaseURL    string
	HTTPClient Doer
}

// Client is a Jina tensor embedder.
type Client struct {
	apiKey  string
	model   string
	baseURL string
	client  Doer
}

// New constructs a tensor embedder.
func New(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.APIKey) == "" {
		return nil, fmt.Errorf("%w: jina api key", ragy.ErrInvalidArgument)
	}

	if strings.TrimSpace(cfg.Model) == "" {
		return nil, fmt.Errorf("%w: jina model", ragy.ErrInvalidArgument)
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	return &Client{
		apiKey:  cfg.APIKey,
		model:   cfg.Model,
		baseURL: strings.TrimRight(baseURL, "/"),
		client:  client,
	}, nil
}

type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Data []struct {
		Index  int         `json:"index"`
		Tensor [][]float32 `json:"tensor"`
	} `json:"data"`
}

// Embed implements tensor.Embedder.
func (c *Client) Embed(ctx context.Context, texts []string) ([]roottensor.Tensor, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("%w: jina texts", ragy.ErrEmptyText)
	}

	body, err := json.Marshal(embedRequest{Model: c.model, Input: texts})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/tensor", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return nil, fmt.Errorf(
			"%w: jina status %d: %s",
			ragy.ErrProtocol,
			resp.StatusCode,
			strings.TrimSpace(string(payload)),
		)
	}

	var decoded embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, err
	}

	if len(decoded.Data) != len(texts) {
		return nil, fmt.Errorf("%w: jina tensor cardinality mismatch", ragy.ErrProtocol)
	}

	out := make([]roottensor.Tensor, len(texts))
	seen := make([]bool, len(texts))
	for _, item := range decoded.Data {
		if item.Index < 0 || item.Index >= len(texts) || seen[item.Index] {
			return nil, fmt.Errorf("%w: jina tensor index %d", ragy.ErrProtocol, item.Index)
		}

		out[item.Index] = append(roottensor.Tensor(nil), item.Tensor...)
		seen[item.Index] = true
	}

	for _, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("%w: jina tensor missing index", ragy.ErrProtocol)
		}
	}

	return out, nil
}

var _ roottensor.Embedder = (*Client)(nil)
