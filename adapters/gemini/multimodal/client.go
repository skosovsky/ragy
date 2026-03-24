package multimodal

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	ragy "github.com/skosovsky/ragy"
	rootmultimodal "github.com/skosovsky/ragy/multimodal"
)

// DefaultBaseURL is the default Gemini API endpoint.
const DefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
const maxErrorBodyBytes = 4 << 10

// Doer executes HTTP requests.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// Config configures the Gemini multimodal adapter.
type Config struct {
	APIKey     string
	Model      string
	BaseURL    string
	HTTPClient Doer
}

// Client is a Gemini multimodal embedder.
type Client struct {
	apiKey  string
	model   string
	baseURL string
	client  Doer
}

// New constructs a multimodal embedder.
func New(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.APIKey) == "" {
		return nil, fmt.Errorf("%w: gemini api key", ragy.ErrInvalidArgument)
	}

	if strings.TrimSpace(cfg.Model) == "" {
		return nil, fmt.Errorf("%w: gemini model", ragy.ErrInvalidArgument)
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
	Inputs []rootmultimodal.Input `json:"inputs"`
}

type embedResponse struct {
	Embeddings []struct {
		Index  int       `json:"index"`
		Vector []float32 `json:"vector"`
	} `json:"embeddings"`
}

// Embed implements multimodal.Embedder.
func (c *Client) Embed(ctx context.Context, inputs []rootmultimodal.Input) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("%w: gemini multimodal inputs", ragy.ErrInvalidArgument)
	}

	for _, input := range inputs {
		if err := input.Validate(); err != nil {
			return nil, err
		}
	}

	body, err := json.Marshal(embedRequest{Inputs: inputs})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		fmt.Sprintf("%s/models/%s:embedMultimodal?key=%s", c.baseURL, c.model, c.apiKey),
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return nil, fmt.Errorf(
			"%w: gemini status %d: %s",
			ragy.ErrProtocol,
			resp.StatusCode,
			strings.TrimSpace(string(payload)),
		)
	}

	var decoded embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, err
	}

	if len(decoded.Embeddings) != len(inputs) {
		return nil, fmt.Errorf("%w: gemini multimodal cardinality mismatch", ragy.ErrProtocol)
	}

	out := make([][]float32, len(inputs))
	seen := make([]bool, len(inputs))
	for _, item := range decoded.Embeddings {
		if item.Index < 0 || item.Index >= len(inputs) || seen[item.Index] {
			return nil, fmt.Errorf("%w: gemini multimodal index %d", ragy.ErrProtocol, item.Index)
		}

		out[item.Index] = append([]float32(nil), item.Vector...)
		seen[item.Index] = true
	}

	for _, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("%w: gemini multimodal missing index", ragy.ErrProtocol)
		}
	}

	return out, nil
}

var _ rootmultimodal.Embedder = (*Client)(nil)
