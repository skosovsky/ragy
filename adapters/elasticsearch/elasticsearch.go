package elasticsearch

import (
	"context"
	"fmt"
	"math"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
)

const logisticClamp = 20.0

// Hit is an Elasticsearch hit projection.
type Hit struct {
	ID     string
	Score  float64
	Source map[string]any
}

// Client executes lexical searches.
type Client interface {
	Search(ctx context.Context, index string, body map[string]any) ([]Hit, error)
}

// Config configures the searcher.
type Config struct {
	Index        string
	SearchFields []string
	Schema       filter.Schema
}

// Searcher is an Elasticsearch lexical searcher.
type Searcher struct {
	client Client
	index  string
	fields []string
	schema filter.Schema
}

// New constructs a lexical searcher.
func New(client Client, cfg Config) (*Searcher, error) {
	if client == nil {
		return nil, fmt.Errorf("%w: elasticsearch client", ragy.ErrInvalidArgument)
	}

	if err := filter.ValidateElasticsearchIndexName(cfg.Index); err != nil {
		return nil, err
	}
	if !cfg.Schema.IsFinalized() {
		return nil, fmt.Errorf("%w: elasticsearch schema", ragy.ErrInvalidArgument)
	}

	if len(cfg.SearchFields) == 0 {
		return nil, fmt.Errorf("%w: elasticsearch search fields", ragy.ErrInvalidArgument)
	}

	fields := make([]string, 0, len(cfg.SearchFields))
	seen := make(map[string]struct{}, len(cfg.SearchFields))
	for _, fieldName := range cfg.SearchFields {
		if err := validateSearchField(fieldName); err != nil {
			return nil, err
		}
		if _, exists := seen[fieldName]; exists {
			return nil, fmt.Errorf("%w: duplicate elasticsearch search field %q", ragy.ErrInvalidArgument, fieldName)
		}
		seen[fieldName] = struct{}{}
		fields = append(fields, fieldName)
	}

	return &Searcher{client: client, index: cfg.Index, fields: fields, schema: cfg.Schema}, nil
}

// Search implements lexical.Searcher.
func (s *Searcher) Search(ctx context.Context, req lexical.Request) ([]ragy.Document, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if err := s.Schema().ValidateSchemaIR(req.Filter); err != nil {
		return nil, err
	}

	body, err := s.render(req)
	if err != nil {
		return nil, err
	}

	hits, err := s.client.Search(ctx, s.index, body)
	if err != nil {
		return nil, ragy.WrapBackendError(err, "elasticsearch search")
	}

	if len(hits) == 0 {
		return nil, nil
	}

	docs := make([]ragy.Document, 0, len(hits))
	for _, hit := range hits {
		doc, err := s.projectHit(hit)
		if err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

// Schema returns the finalized filter schema used by the searcher.
func (s *Searcher) Schema() filter.Schema {
	return s.schema
}

func (s *Searcher) render(req lexical.Request) (map[string]any, error) {
	multiMatch := map[string]any{
		"query":  req.Text,
		"fields": slices.Clone(s.fields),
	}

	query := map[string]any{
		"multi_match": multiMatch,
	}

	if req.Filter != nil {
		rendered, err := renderFilter(req.Filter)
		if err != nil {
			return nil, err
		}
		if rendered != nil {
			query = map[string]any{
				"bool": map[string]any{
					"must":   []any{query},
					"filter": []any{rendered},
				},
			}
		}
	}

	body := map[string]any{"query": query}
	if req.Page != nil {
		body["size"] = req.Page.Limit
		body["from"] = req.Page.Offset
	}

	return body, nil
}

func renderFilter(expr filter.IR) (map[string]any, error) {
	walker := &esFilterWalker{stack: nil, result: nil}
	if err := filter.Walk(expr, walker); err != nil {
		return nil, err
	}
	return walker.result, nil
}

func (s *Searcher) projectHit(hit Hit) (ragy.Document, error) {
	contentValue, ok := hit.Source["content"]
	if !ok {
		return ragy.Document{}, fmt.Errorf("%w: elasticsearch content missing", ragy.ErrProtocol)
	}

	content, ok := contentValue.(string)
	if !ok {
		return ragy.Document{}, fmt.Errorf("%w: elasticsearch content must be string", ragy.ErrProtocol)
	}

	doc := ragy.Document{
		ID:         hit.ID,
		Content:    content,
		Attributes: nil,
		Relevance:  logistic(hit.Score),
	}

	attrs, err := s.projectAttributes(hit.Source)
	if err != nil {
		return ragy.Document{}, err
	}
	doc.Attributes = attrs

	return ragy.NormalizeDocument(doc)
}

func (s *Searcher) projectAttributes(source map[string]any) (ragy.Attributes, error) {
	if len(source) == 0 {
		var attrs ragy.Attributes
		return attrs, nil
	}

	projected := make(ragy.Attributes)
	for key, value := range source {
		if key == "content" {
			continue
		}
		if _, ok := s.schema.Lookup(key); !ok {
			continue
		}
		projected[key] = value
	}

	attrs, err := s.schema.NormalizeAttributes(projected)
	if err != nil {
		return nil, err
	}
	if len(attrs) == 0 {
		var normalized ragy.Attributes
		return normalized, nil
	}

	return attrs, nil
}

func validateSearchField(fieldName string) error {
	switch fieldName {
	case "":
		return fmt.Errorf("%w: elasticsearch search field", ragy.ErrInvalidArgument)
	case "content":
		return nil
	default:
		return filter.ValidateIdentifier(fieldName)
	}
}

type esFrame struct {
	op    string
	items []map[string]any
}

type esFilterWalker struct {
	stack  []esFrame
	result map[string]any
}

func (w *esFilterWalker) OnEmpty() error {
	return w.push(map[string]any{"match_all": map[string]any{}})
}

func (w *esFilterWalker) OnEq(field string, value filter.Value) error {
	return w.push(map[string]any{"term": map[string]any{field: value.Raw()}})
}

func (w *esFilterWalker) OnNeq(field string, value filter.Value) error {
	return w.push(map[string]any{
		"bool": map[string]any{
			"must_not": []any{map[string]any{"term": map[string]any{field: value.Raw()}}},
		},
	})
}

func (w *esFilterWalker) OnGt(field string, value filter.Value) error {
	return w.push(rangeQuery(field, "gt", value.Raw()))
}

func (w *esFilterWalker) OnGte(field string, value filter.Value) error {
	return w.push(rangeQuery(field, "gte", value.Raw()))
}

func (w *esFilterWalker) OnLt(field string, value filter.Value) error {
	return w.push(rangeQuery(field, "lt", value.Raw()))
}

func (w *esFilterWalker) OnLte(field string, value filter.Value) error {
	return w.push(rangeQuery(field, "lte", value.Raw()))
}

func (w *esFilterWalker) OnIn(field string, values []filter.Value) error {
	items := make([]any, 0, len(values))
	for _, value := range values {
		items = append(items, value.Raw())
	}
	return w.push(map[string]any{"terms": map[string]any{field: items}})
}

func (w *esFilterWalker) EnterAnd(_ int) error {
	w.stack = append(w.stack, esFrame{op: "and", items: nil})
	return nil
}

func (w *esFilterWalker) LeaveAnd() error {
	frame, err := w.pop("and")
	if err != nil {
		return err
	}

	items := make([]any, 0, len(frame.items))
	for _, item := range frame.items {
		items = append(items, item)
	}
	return w.push(map[string]any{"bool": map[string]any{"filter": items}})
}

func (w *esFilterWalker) EnterOr(_ int) error {
	w.stack = append(w.stack, esFrame{op: "or", items: nil})
	return nil
}

func (w *esFilterWalker) LeaveOr() error {
	frame, err := w.pop("or")
	if err != nil {
		return err
	}

	items := make([]any, 0, len(frame.items))
	for _, item := range frame.items {
		items = append(items, item)
	}
	return w.push(map[string]any{"bool": map[string]any{"should": items, "minimum_should_match": 1}})
}

func (w *esFilterWalker) EnterNot() error {
	w.stack = append(w.stack, esFrame{op: "not", items: nil})
	return nil
}

func (w *esFilterWalker) LeaveNot() error {
	frame, err := w.pop("not")
	if err != nil {
		return err
	}
	if len(frame.items) != 1 {
		return fmt.Errorf("%w: invalid NOT filter", ragy.ErrUnsupported)
	}
	return w.push(map[string]any{"bool": map[string]any{"must_not": []any{frame.items[0]}}})
}

func (w *esFilterWalker) push(query map[string]any) error {
	if len(w.stack) == 0 {
		w.result = query
		return nil
	}

	last := len(w.stack) - 1
	w.stack[last].items = append(w.stack[last].items, query)
	return nil
}

func (w *esFilterWalker) pop(op string) (esFrame, error) {
	if len(w.stack) == 0 {
		return esFrame{}, fmt.Errorf("%w: unmatched %s filter", ragy.ErrUnsupported, op)
	}

	last := len(w.stack) - 1
	frame := w.stack[last]
	w.stack = w.stack[:last]
	if frame.op != op {
		return esFrame{}, fmt.Errorf("%w: unexpected filter group %q", ragy.ErrUnsupported, frame.op)
	}
	return frame, nil
}

func rangeQuery(field, op string, value any) map[string]any {
	return map[string]any{"range": map[string]any{field: map[string]any{op: value}}}
}

func logistic(score float64) float64 {
	score = math.Max(-logisticClamp, math.Min(logisticClamp, score))
	return 1.0 / (1.0 + math.Exp(-score))
}

var _ lexical.Searcher = (*Searcher)(nil)
