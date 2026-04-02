package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/ranking"
)

// DefaultBaseURL is the default Cohere API endpoint.
const DefaultBaseURL = "https://api.cohere.com/v2"
const maxErrorBodyBytes = 4 << 10

// Doer executes HTTP requests.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// Config configures the Cohere reranker.
type Config struct {
	APIKey     string
	Model      string
	BaseURL    string
	HTTPClient Doer
}

// Client is a Cohere query-aware reranker.
type Client struct {
	apiKey  string
	model   string
	baseURL string
	client  Doer
}

// New constructs a reranker.
func New(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.APIKey) == "" {
		return nil, fmt.Errorf("%w: cohere api key", ragy.ErrInvalidArgument)
	}

	if strings.TrimSpace(cfg.Model) == "" {
		return nil, fmt.Errorf("%w: cohere model", ragy.ErrInvalidArgument)
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

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

type rerankResponse struct {
	Results []struct {
		Index int     `json:"index"`
		Score float64 `json:"relevance_score"`
	} `json:"results"`
}

// Rerank implements ranking.QueryReranker.
func (c *Client) Rerank(ctx context.Context, query string, docs []ragy.Document) ([]ragy.Document, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("%w: rerank query", ragy.ErrEmptyText)
	}

	if len(docs) == 0 {
		return nil, nil
	}

	payloadDocs := make([]string, 0, len(docs))
	normalizedDocs := make([]ragy.Document, len(docs))
	for i, doc := range docs {
		normalized, err := ragy.NormalizeDocument(doc)
		if err != nil {
			return nil, err
		}
		normalizedDocs[i] = normalized
		payloadDocs = append(payloadDocs, normalized.Content)
	}

	body, err := json.Marshal(rerankRequest{Model: c.model, Query: query, Documents: payloadDocs})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
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
			"cohere rerank",
			strings.TrimSpace(string(payload)),
		)
	}

	var decoded rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("%w: cohere rerank decode: %w", ragy.ErrProtocol, err)
	}

	if len(decoded.Results) != len(docs) {
		return nil, fmt.Errorf("%w: rerank cardinality mismatch", ragy.ErrProtocol)
	}

	out := make([]ragy.Document, len(docs))
	seen := make([]bool, len(docs))
	for _, result := range decoded.Results {
		if result.Index < 0 || result.Index >= len(docs) || seen[result.Index] {
			return nil, fmt.Errorf("%w: rerank index %d", ragy.ErrProtocol, result.Index)
		}

		doc := normalizedDocs[result.Index]
		doc.Relevance = ragy.ClampRelevance(result.Score)
		out[result.Index] = doc
		seen[result.Index] = true
	}

	for _, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("%w: rerank missing index", ragy.ErrProtocol)
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Relevance == out[j].Relevance {
			return out[i].ID < out[j].ID
		}
		return out[i].Relevance > out[j].Relevance
	})

	return out, nil
}

var _ ranking.QueryReranker = (*Client)(nil)
