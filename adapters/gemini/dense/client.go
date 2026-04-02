package dense

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	ragy "github.com/skosovsky/ragy"
	rootdense "github.com/skosovsky/ragy/dense"
)

// DefaultBaseURL is the default Gemini API endpoint.
const DefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
const maxErrorBodyBytes = 4 << 10

// Doer executes HTTP requests.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// Config configures the Gemini dense adapter.
type Config struct {
	APIKey     string
	Model      string
	BaseURL    string
	HTTPClient Doer
}

// Client is a Gemini dense embedder.
type Client struct {
	apiKey  string
	model   string
	baseURL string
	client  Doer
}

// New constructs a dense embedder.
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
	Texts []string `json:"texts"`
}

type embedResponse struct {
	Embeddings []struct {
		Index  int       `json:"index"`
		Vector []float32 `json:"vector"`
	} `json:"embeddings"`
}

// Embed implements dense.Embedder.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("%w: gemini texts", ragy.ErrEmptyText)
	}

	body, err := json.Marshal(embedRequest{Texts: texts})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		fmt.Sprintf("%s/models/%s:embed?key=%s", c.baseURL, c.model, c.apiKey),
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, ragy.WrapTransportError(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return nil, ragy.ErrorFromHTTPResponse(
			resp.StatusCode,
			"gemini dense",
			strings.TrimSpace(string(payload)),
		)
	}

	var decoded embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("%w: gemini dense decode: %w", ragy.ErrProtocol, err)
	}

	if len(decoded.Embeddings) != len(texts) {
		return nil, fmt.Errorf("%w: gemini embedding cardinality mismatch", ragy.ErrProtocol)
	}

	out := make([][]float32, len(texts))
	seen := make([]bool, len(texts))
	for _, item := range decoded.Embeddings {
		if item.Index < 0 || item.Index >= len(texts) || seen[item.Index] {
			return nil, fmt.Errorf("%w: gemini embedding index %d", ragy.ErrProtocol, item.Index)
		}

		out[item.Index] = append([]float32(nil), item.Vector...)
		seen[item.Index] = true
	}

	for _, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("%w: gemini embedding missing index", ragy.ErrProtocol)
		}
	}

	return out, nil
}

var _ rootdense.Embedder = (*Client)(nil)
