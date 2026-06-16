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
	"github.com/skosovsky/ragy/retrieval"
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
type Client[TMeta any] struct {
	apiKey  string
	model   string
	baseURL string
	client  Doer
}

// New constructs a reranker.
func New[TMeta any](cfg Config) (*Client[TMeta], error) {
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

	return &Client[TMeta]{
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

func emptyResultSet[TMeta any](resolver retrieval.IdentityResolver[TMeta]) retrieval.ResultSet[TMeta] {
	if resolver == nil {
		resolver = retrieval.DocumentIDResolver[TMeta]{}
	}
	return retrieval.NewResultSet[TMeta](nil, resolver)
}

func (c *Client[TMeta]) prepareRerankPayload(
	rs retrieval.ResultSet[TMeta],
) ([]retrieval.Document[TMeta], []string, error) {
	docs := rs.Documents()
	payloadDocs := make([]string, 0, len(docs))
	normalizedDocs := make([]retrieval.Document[TMeta], 0, len(docs))
	for _, doc := range docs {
		if err := retrieval.ValidateDocument(doc); err != nil {
			return normalizedDocs, payloadDocs, ragy.WrapProjectionError(err, "rerank validate")
		}
		normalizedDocs = append(normalizedDocs, doc)
		payloadDocs = append(payloadDocs, doc.Content)
	}
	return normalizedDocs, payloadDocs, nil
}

func (c *Client[TMeta]) postRerank(
	ctx context.Context,
	query string,
	payloadDocs []string,
) (rerankResponse, error) {
	body, err := json.Marshal(rerankRequest{Model: c.model, Query: query, Documents: payloadDocs})
	if err != nil {
		return rerankResponse{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return rerankResponse{}, err
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return rerankResponse{}, ragy.WrapTransportError(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return rerankResponse{}, ragy.ErrorFromHTTPResponse(
			resp.StatusCode,
			"cohere rerank",
			strings.TrimSpace(string(payload)),
		)
	}

	var decoded rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return rerankResponse{}, fmt.Errorf(
			"%w: cohere rerank decode: %w",
			ragy.ErrProtocol,
			err,
		)
	}
	return decoded, nil
}

func applyRerankResults[TMeta any](
	normalizedDocs []retrieval.Document[TMeta],
	decoded rerankResponse,
) ([]retrieval.Document[TMeta], error) {
	if len(decoded.Results) != len(normalizedDocs) {
		return nil, fmt.Errorf("%w: rerank cardinality mismatch", ragy.ErrProtocol)
	}

	out := make([]retrieval.Document[TMeta], len(normalizedDocs))
	seen := make([]bool, len(normalizedDocs))
	for _, result := range decoded.Results {
		if result.Index < 0 || result.Index >= len(normalizedDocs) || seen[result.Index] {
			return nil, fmt.Errorf("%w: rerank index %d", ragy.ErrProtocol, result.Index)
		}

		doc := normalizedDocs[result.Index]
		doc.Score = ragy.ClampScore(result.Score)
		out[result.Index] = doc
		seen[result.Index] = true
	}

	for _, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("%w: rerank missing index", ragy.ErrProtocol)
		}
	}

	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Score > out[j].Score
	})

	return out, nil
}

// Rerank implements ranking.QueryReranker.
// Nil or empty result sets are returned unchanged without error.
// Empty query is a validation error and returns an empty ResultSet (input docs are not preserved).
// Runtime and payload errors preserve the input ResultSet via retrieval.PreserveResultOnError.
func (c *Client[TMeta]) Rerank(
	ctx context.Context,
	query string,
	rs retrieval.ResultSet[TMeta],
) (retrieval.ResultSet[TMeta], error) {
	if strings.TrimSpace(query) == "" {
		return emptyResultSet[TMeta](retrieval.ResolverFor(rs)), fmt.Errorf("%w: rerank query", ragy.ErrEmptyText)
	}
	if rs == nil || rs.IsEmpty() {
		return emptyResultSet[TMeta](retrieval.ResolverFor(rs)), nil
	}

	normalizedDocs, payloadDocs, err := c.prepareRerankPayload(rs)
	resolver := retrieval.ResolverFor(rs)
	if err != nil {
		partial := retrieval.NewResultSet(normalizedDocs, resolver)
		return retrieval.PreserveResultOnError(partial, err, resolver)
	}

	decoded, err := c.postRerank(ctx, query, payloadDocs)
	if err != nil {
		return retrieval.PreserveResultOnError(rs, err, resolver)
	}

	out, err := applyRerankResults(normalizedDocs, decoded)
	if err != nil {
		return retrieval.PreserveResultOnError(rs, err, resolver)
	}

	return retrieval.NewResultSet(out, resolver), nil
}

var _ ranking.QueryReranker[any] = (*Client[any])(nil)
