package rerank

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/url"
	"sort"
	"strings"
	"unicode/utf8"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/cohere-ai/cohere-go/v2/core"
	"github.com/cohere-ai/cohere-go/v2/option"

	"github.com/skosovsky/ragy/internal/providerpolicy"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/embedding"

	"github.com/skosovsky/ragy/retrieval"
)

// DefaultBaseURL is the default Cohere API endpoint.
const DefaultBaseURL = "https://api.cohere.com"

type Config struct {
	SDK    *cohereclient.Client
	Model  string
	Limits embedding.Limits
}
type Client[TMeta any] struct {
	model  string
	sdk    *cohereclient.Client
	limits embedding.Limits
}

// New constructs a reranker.
func New[TMeta any](cfg Config) (*Client[TMeta], error) {
	if cfg.SDK == nil || cfg.SDK.V2 == nil || strings.TrimSpace(cfg.Model) == "" || !utf8.ValidString(cfg.Model) {
		return nil, ragy.ErrInvalidArgument
	}
	limits, err := cfg.Limits.Resolve()
	if err != nil {
		return nil, err
	}
	if len(cfg.Model) > limits.MaxInputBytes {
		return nil, ragy.ErrInvalidArgument
	}
	return &Client[TMeta]{model: cfg.Model, sdk: cfg.SDK, limits: limits}, nil
}

// Result retains the ordinary retrieval result and observed provider accounting.
// BilledUnits are Cohere search units; input token counts are unavailable.
type Result[TMeta any] struct {
	Documents retrieval.ResultSet[TMeta]
	Usage     embedding.Usage
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
) (*cohere.V2RerankResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, c.limits.Timeout)
	defer cancel()
	topN := len(payloadDocs)
	//nolint:exhaustruct_v5 // SDK options outside the complete-ranking domain profile remain omitted.
	body := &cohere.V2RerankRequest{Model: c.model, Query: query, Documents: payloadDocs, TopN: &topN}
	// SDK-L01: public representation cap, not wire retention.
	if err := providerpolicy.Size(body, c.limits.MaxRequestBytes, ragy.ErrInvalidArgument); err != nil {
		return nil, err
	}
	decoded, err := c.sdk.V2.Rerank(ctx, body, option.WithMaxAttempts(1))
	// SDK-L08: typed decode failure exposes no completed usage DTO.
	if err != nil {
		status := 0
		if apiErr, ok := errors.AsType[*core.APIError](err); ok {
			status = apiErr.StatusCode
		} else {
			if _, ok := errors.AsType[*url.Error](err); !ok {
				status = 200
			}
		}
		return nil, providerpolicy.Failure(ctx, err, status)
	}
	return decoded, nil
}

func applyRerankResults[TMeta any](
	normalizedDocs []retrieval.Document[TMeta],
	decoded *cohere.V2RerankResponse,
) ([]retrieval.Document[TMeta], error) {
	if len(decoded.Results) != len(normalizedDocs) {
		return nil, fmt.Errorf("%w: rerank cardinality mismatch", ragy.ErrProtocol)
	}

	out := make([]retrieval.Document[TMeta], len(normalizedDocs))
	seen := make([]bool, len(normalizedDocs))
	for _, result := range decoded.Results {
		if result == nil || math.IsNaN(result.RelevanceScore) || math.IsInf(result.RelevanceScore, 0) {
			return nil, ragy.ErrProtocol
		}
		if result.Index < 0 || result.Index >= len(normalizedDocs) || seen[result.Index] {
			return nil, fmt.Errorf("%w: rerank index", ragy.ErrProtocol)
		}

		doc := normalizedDocs[result.Index]
		doc.ScoreHistory = doc.ObservedScores()
		doc.Score = result.RelevanceScore
		doc.ScoreState = retrieval.ScorePresent
		doc.ScoreSemantics = "rerank.model-native"
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
	if c == nil || c.sdk == nil {
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

	if rs.Len() > c.limits.MaxInputs {
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
	if validationErr := providerpolicy.ValidateTexts(ctx, texts, c.limits); validationErr != nil {
		return retrieval.PreserveResultOnError(rs, validationErr, resolver)
	}
	if gateErr := read.Check(ctx); gateErr != nil {
		return emptyResultSet[TMeta](resolver), gateErr
	}
	decoded, err := c.postRerank(ctx, query, payloadDocs)
	if err != nil {
		return retrieval.PreserveResultOnError(rs, err, resolver)
	}

	if decoded == nil {
		return retrieval.PreserveResultOnError(rs, ragy.ErrProtocol, resolver)
	}
	observed, usageErr := readRerankUsage(decoded)
	if usageErr != nil {
		return retrieval.PreserveResultOnError(rs, usageErr, resolver)
	}
	*usage = observed
	if gateErr := ctx.Err(); gateErr != nil {
		return retrieval.PreserveResultOnError(rs, gateErr, resolver)
	}
	// SDK-L01: public representation cap, not wire retention.
	if sizeErr := providerpolicy.Size(decoded, c.limits.MaxResponseBytes, ragy.ErrProtocol); sizeErr != nil {
		return retrieval.PreserveResultOnError(rs, sizeErr, resolver)
	}
	if model, ok := decoded.GetExtraProperties()["model"]; ok && model != c.model {
		return retrieval.PreserveResultOnError(rs, ragy.ErrProtocol, resolver)
	}
	// SDK-L05: required scalar absence decodes to zero; no public presence API.
	out, err := applyRerankResults(normalizedDocs, decoded)
	if err != nil {
		return retrieval.PreserveResultOnError(rs, err, resolver)
	}

	return retrieval.NewResultSet(out, resolver), nil
}

var _ retrieval.QueryReranker[any] = (*Client[any])(nil)

func readRerankUsage(decoded *cohere.V2RerankResponse) (embedding.Usage, error) {
	var usage embedding.Usage
	// SDK-L04: float64 counters cannot promise lossless integer precision.
	if decoded.Meta != nil && decoded.Meta.BilledUnits != nil && decoded.Meta.BilledUnits.SearchUnits != nil {
		value := *decoded.Meta.BilledUnits.SearchUnits
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value >= float64(math.MaxInt64) ||
			math.Trunc(value) != value {
			return usage, ragy.ErrProtocol
		}
		usage.BilledUnits = int64(value)
		usage.BilledUnitsKnown = true
	}
	return usage, nil
}
