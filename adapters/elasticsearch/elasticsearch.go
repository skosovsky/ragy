// Package elasticsearch provides a ragy.Retriever for keyword/lexical search using Elasticsearch (match, multi_match).
// Use with EnsembleRetriever for hybrid vector + keyword retrieval.
// Default index is "ragy_docs"; use WithIndex to override (the deprecated "_all" is not used, as it is removed in ES 8.x).
// Search field names (WithSearchFields) are sanitized; only [a-zA-Z0-9_] are allowed. Use trusted config only.
package elasticsearch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"math"
	"regexp"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/elastic/go-elasticsearch/v8/esapi"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

var fieldSanitize = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)

// logisticScoreClamp bounds raw ES _score before mapping to confidence via logistic (same scale as other retrievers).
const logisticScoreClamp = 20.0

// Retriever implements ragy.Retriever using Elasticsearch match/multi_match and filter.Expr → bool query.
type Retriever struct {
	client *elasticsearch.Client
	index  string
	fields []string
}

// Option configures the Retriever.
type Option func(*Retriever)

// WithIndex sets the index name (default "ragy_docs").
func WithIndex(name string) Option {
	return func(r *Retriever) { r.index = name }
}

// WithSearchFields sets the fields for multi_match (default ["content"]).
func WithSearchFields(fields []string) Option {
	return func(r *Retriever) { r.fields = fields }
}

// New returns a new Elasticsearch Retriever.
func New(client *elasticsearch.Client, opts ...Option) *Retriever {
	r := &Retriever{
		client: client,
		index:  "ragy_docs",
		fields: []string{"content"},
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

// sanitizeSearchFields returns only field names that match [a-zA-Z0-9_]. If none valid, returns default ["content"].
func sanitizeSearchFields(fields []string) []string {
	var out []string
	for _, f := range fields {
		if fieldSanitize.MatchString(f) {
			out = append(out, f)
		}
	}
	if len(out) == 0 {
		return []string{"content"}
	}
	return out
}

// Retrieve implements ragy.Retriever. Builds match/multi_match + bool filter from req.Filter.
func (r *Retriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	if len(req.SparseVector) > 0 {
		return nil, fmt.Errorf("elasticsearch: %w", ragy.ErrSparseVectorNotSupported)
	}
	if req.Query == "" {
		return []ragy.Document{}, nil
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	offset := max(req.Offset, 0)
	fields := sanitizeSearchFields(r.fields)
	body := buildSearchBody(req.Query, fields, req.Filter, limit, offset)
	reqES := esapi.SearchRequest{
		Index: []string{r.index},
		Body:  bytes.NewReader(body),
	}
	res, err := reqES.Do(ctx, r.client)
	if err != nil {
		return nil, err
	}
	defer func() { _ = res.Body.Close() }()
	if res.IsError() {
		return nil, fmt.Errorf("elasticsearch: %s", res.String())
	}
	var searchRes struct {
		Hits struct {
			Hits []struct {
				ID     string          `json:"_id"`
				Source json.RawMessage `json:"_source"`
				Score  float64         `json:"_score"`
			} `json:"hits"`
		} `json:"hits"`
	}
	if err := json.NewDecoder(res.Body).Decode(&searchRes); err != nil {
		return nil, err
	}
	docs := make([]ragy.Document, 0, len(searchRes.Hits.Hits))
	for _, h := range searchRes.Hits.Hits {
		doc := ragy.Document{ID: h.ID, Score: float32(h.Score)}
		s := math.Min(logisticScoreClamp, math.Max(-logisticScoreClamp, h.Score))
		doc.Confidence = 1.0 / (1.0 + math.Exp(-s))
		var src map[string]any
		_ = json.Unmarshal(h.Source, &src)
		if c, ok := src["content"].(string); ok {
			doc.Content = c
		}
		doc.Metadata = src
		docs = append(docs, doc)
	}
	return docs, nil
}

// Stream implements ragy.Retriever.
func (r *Retriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := r.Retrieve(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

func buildSearchBody(query string, fields []string, f filter.Expr, limit, offset int) []byte {
	queryClause := map[string]any{}
	if len(fields) == 1 {
		queryClause["match"] = map[string]any{fields[0]: query}
	} else {
		queryClause["multi_match"] = map[string]any{"query": query, "fields": fields}
	}
	boolQ := map[string]any{"must": []any{queryClause}}
	body := map[string]any{
		"query": map[string]any{"bool": boolQ},
		"size":  limit,
		"from":  offset,
	}
	if f != nil {
		filterClause := buildBoolFilter(f)
		if filterClause != nil {
			boolQ["filter"] = []any{filterClause}
		}
	}
	raw, _ := json.Marshal(body)
	return raw
}

//nolint:gocognit,funlen // filter.Expr → ES bool clause is a large switch with nested And/Or handling.
func buildBoolFilter(expr filter.Expr) map[string]any {
	switch e := expr.(type) {
	case filter.Eq:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return map[string]any{"term": map[string]any{e.Field: e.Value}}
	case filter.Neq:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return map[string]any{
			"bool": map[string]any{"must_not": []any{map[string]any{"term": map[string]any{e.Field: e.Value}}}},
		}
	case filter.In:
		if !fieldSanitize.MatchString(e.Field) || len(e.Values) == 0 {
			return nil
		}
		return map[string]any{"terms": map[string]any{e.Field: e.Values}}
	case filter.Gt:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return map[string]any{"range": map[string]any{e.Field: map[string]any{"gt": e.Value}}}
	case filter.Gte:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return map[string]any{"range": map[string]any{e.Field: map[string]any{"gte": e.Value}}}
	case filter.Lt:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return map[string]any{"range": map[string]any{e.Field: map[string]any{"lt": e.Value}}}
	case filter.Lte:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return map[string]any{"range": map[string]any{e.Field: map[string]any{"lte": e.Value}}}
	case filter.And:
		var must []any
		for _, sub := range e.Exprs {
			if c := buildBoolFilter(sub); c != nil {
				must = append(must, c)
			}
		}
		if len(must) == 0 {
			return nil
		}
		return map[string]any{"bool": map[string]any{"must": must}}
	case filter.Or:
		var should []any
		for _, sub := range e.Exprs {
			if c := buildBoolFilter(sub); c != nil {
				should = append(should, c)
			}
		}
		if len(should) == 0 {
			return nil
		}
		return map[string]any{"bool": map[string]any{"should": should}}
	case filter.Not:
		if c := buildBoolFilter(e.Expr); c != nil {
			return map[string]any{"bool": map[string]any{"must_not": []any{c}}}
		}
		return nil
	default:
		return nil
	}
}

var _ ragy.Retriever = (*Retriever)(nil)
