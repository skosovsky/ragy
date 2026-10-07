package rerank

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/internal/providerhttp"
	"github.com/skosovsky/ragy/retrieval"
)

// DefaultBaseURL is the default Cohere API endpoint.
const DefaultBaseURL = "https://api.cohere.com/v2"

// Doer executes one HTTP exchange. Implementations must honor cancellation and
// must not hide retries or redirects.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// Config configures the Cohere reranker.
type Config struct {
	APIKey     string
	Model      string
	BaseURL    string
	HTTPClient Doer
	Limits     embedding.Limits
}

// Client is a Cohere query-aware reranker.
type Client[TMeta any] struct {
	apiKey string
	model  string
	client *providerhttp.Client
}

// New constructs a reranker.
func New[TMeta any](cfg Config) (*Client[TMeta], error) {
	if providerhttp.ValidateAPIKey(cfg.APIKey) != nil {
		return nil, fmt.Errorf("%w: cohere api key", ragy.ErrInvalidArgument)
	}

	if strings.TrimSpace(cfg.Model) == "" {
		return nil, fmt.Errorf("%w: cohere model", ragy.ErrInvalidArgument)
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	transport, err := providerhttp.New(
		providerhttp.Config{BaseURL: baseURL, HTTPClient: cfg.HTTPClient, Limits: cfg.Limits},
	)
	if err != nil {
		return nil, err
	}
	if len(cfg.Model) > transport.Limits().MaxInputBytes || len(cfg.APIKey) > transport.Limits().MaxInputBytes ||
		!utf8.ValidString(cfg.Model) {
		return nil, ragy.ErrInvalidArgument
	}
	return &Client[TMeta]{apiKey: cfg.APIKey, model: cfg.Model, client: transport}, nil
}

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

// Result retains the ordinary retrieval result and observed provider accounting.
// BilledUnits are Cohere search units; input token counts are unavailable.
type Result[TMeta any] struct {
	Documents retrieval.ResultSet[TMeta]
	Usage     embedding.Usage
}

type rerankResponse struct {
	Model string `json:"model"`
	Meta  struct {
		BilledUnits struct {
			SearchUnits *int64 `json:"search_units"`
		} `json:"billed_units"`
	} `json:"meta"`
	Results []struct {
		Index *int     `json:"index"`
		Score *float64 `json:"relevance_score"`
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
	var decoded rerankResponse
	headers := http.Header{"Authorization": []string{"Bearer " + c.apiKey}}
	err := c.client.Post(
		ctx,
		"/rerank",
		headers,
		rerankRequest{Model: c.model, Query: query, Documents: payloadDocs},
		&decoded,
	)
	if err != nil {
		return rerankResponse{}, err
	}
	if decoded.Model != "" && decoded.Model != c.model {
		return rerankResponse{}, ragy.ErrProtocol
	}
	if decoded.Meta.BilledUnits.SearchUnits != nil && *decoded.Meta.BilledUnits.SearchUnits < 0 {
		return rerankResponse{}, ragy.ErrProtocol
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
		if result.Index == nil || result.Score == nil || math.IsNaN(*result.Score) || math.IsInf(*result.Score, 0) {
			return nil, ragy.ErrProtocol
		}
		if *result.Index < 0 || *result.Index >= len(normalizedDocs) || seen[*result.Index] {
			return nil, fmt.Errorf("%w: rerank index", ragy.ErrProtocol)
		}

		doc := normalizedDocs[*result.Index]
		doc.ScoreHistory = doc.ObservedScores()
		doc.Score = *result.Score
		doc.ScoreState = retrieval.ScorePresent
		doc.ScoreSemantics = "rerank.model-native"
		out[*result.Index] = doc
		seen[*result.Index] = true
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

// Rerank implements retrieval.QueryReranker.
// Nil or empty result sets are returned unchanged without error.
// Empty query is a validation error and returns an empty ResultSet (input docs are not preserved).
// Runtime and payload errors preserve the input ResultSet via retrieval.PreserveResultOnError.
func (c *Client[TMeta]) Rerank(
	ctx context.Context,
	read access.Binding,
	query string,
	rs retrieval.ResultSet[TMeta],
) (retrieval.ResultSet[TMeta], error) {
	result, err := c.RerankWithUsage(ctx, read, query, rs)
	return result.Documents, err
}

// RerankWithUsage exposes actual accounting without changing QueryReranker.
// Unknown counters remain zero with Known=false, including calls not dispatched.
func (c *Client[TMeta]) RerankWithUsage(
	ctx context.Context,
	read access.Binding,
	query string,
	rs retrieval.ResultSet[TMeta],
) (Result[TMeta], error) {
	var usage embedding.Usage
	docs, err := c.rerank(ctx, read, query, rs, &usage)
	docs, err = retrieval.DeliverRead(ctx, read, docs, err, retrieval.ResolverFor(rs))
	return Result[TMeta]{Documents: docs, Usage: usage}, err
}

func (c *Client[TMeta]) rerank(
	ctx context.Context,
	read access.Binding, query string,
	rs retrieval.ResultSet[TMeta],
	usage *embedding.Usage,
) (retrieval.ResultSet[TMeta], error) {
	if c == nil || c.client == nil {
		return emptyResultSet[TMeta](retrieval.ResolverFor(rs)), ragy.ErrInvalidArgument
	}
	if err := read.Check(ctx); err != nil {
		return emptyResultSet[TMeta](retrieval.ResolverFor(rs)), err
	}
	if strings.TrimSpace(query) == "" {
		return emptyResultSet[TMeta](retrieval.ResolverFor(rs)), fmt.Errorf("%w: rerank query", ragy.ErrEmptyText)
	}
	if rs == nil || rs.IsEmpty() {
		return emptyResultSet[TMeta](retrieval.ResolverFor(rs)), nil
	}

	if rs.Len() > c.client.Limits().MaxInputs {
		return retrieval.PreserveResultOnError(rs, ragy.ErrInvalidArgument, retrieval.ResolverFor(rs))
	}
	normalizedDocs, payloadDocs, err := c.prepareRerankPayload(rs)
	resolver := retrieval.ResolverFor(rs)
	if err != nil {
		partial := retrieval.NewResultSet(normalizedDocs, resolver)
		return retrieval.PreserveResultOnError(partial, err, resolver)
	}

	texts := append([]string{query}, payloadDocs...)
	// The query participates in aggregate bytes; MaxInputs counts query plus documents.
	if validationErr := c.client.ValidateTexts(ctx, texts); validationErr != nil {
		return retrieval.PreserveResultOnError(rs, validationErr, resolver)
	}
	if gateErr := read.Check(ctx); gateErr != nil {
		return emptyResultSet[TMeta](resolver), gateErr
	}
	decoded, err := c.postRerank(ctx, query, payloadDocs)
	if err != nil {
		return retrieval.PreserveResultOnError(rs, err, resolver)
	}

	if decoded.Meta.BilledUnits.SearchUnits != nil {
		usage.BilledUnits = *decoded.Meta.BilledUnits.SearchUnits
		usage.BilledUnitsKnown = true
	}
	out, err := applyRerankResults(normalizedDocs, decoded)
	if err != nil {
		return retrieval.PreserveResultOnError(rs, err, resolver)
	}

	return retrieval.NewResultSet(out, resolver), nil
}

var _ retrieval.QueryReranker[any] = (*Client[any])(nil)
