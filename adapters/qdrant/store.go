package qdrant

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

const logisticClamp = 20.0

// Condition is a typed qdrant filter condition.
type Condition interface {
	isCondition()
}

// EqCondition is an equality filter.
type EqCondition struct {
	Field string
	Value any
}

func (EqCondition) isCondition() {}

// NeqCondition is a not-equal filter.
type NeqCondition struct {
	Field string
	Value any
}

func (NeqCondition) isCondition() {}

// RangeCondition is a range filter.
type RangeCondition struct {
	Field string
	Op    string
	Value any
}

func (RangeCondition) isCondition() {}

// InCondition is a membership filter.
type InCondition struct {
	Field  string
	Values []any
}

func (InCondition) isCondition() {}

// GroupCondition combines child conditions.
type GroupCondition struct {
	Op    string
	Items []Condition
}

func (GroupCondition) isCondition() {}

// NotCondition negates a condition.
type NotCondition struct {
	Item Condition
}

func (NotCondition) isCondition() {}

// MatchAllCondition represents the absence of a filter.
type MatchAllCondition struct{}

func (MatchAllCondition) isCondition() {}

// Point is a stored vector record.
type Point struct {
	ID         string
	Content    string
	Attributes filter.RawAttributes
	Vector     []float32
	Score      float64
}

// Client executes qdrant operations.
type Client interface {
	Upsert(ctx context.Context, collection string, points []Point) error
	Search(ctx context.Context, collection string, vector []float32, cond Condition, limit int) ([]Point, error)
	Get(ctx context.Context, collection string, ids []string) ([]Point, error)
	DeleteByIDs(ctx context.Context, collection string, ids []string) (int, error)
	DeleteByFilter(ctx context.Context, collection string, cond Condition) (int, error)
}

// Config configures the store.
type Config struct {
	Collection string
	Schema     filter.Schema
}

// Store is a dense qdrant-backed store.
type Store[TMeta any] struct {
	client     Client
	collection string
	schema     filter.Schema
}

// New constructs a store.
func New[TMeta any](client Client, cfg Config) (*Store[TMeta], error) {
	if client == nil {
		return nil, fmt.Errorf("%w: qdrant client", ragy.ErrInvalidArgument)
	}

	if err := filter.ValidateCollectionName(cfg.Collection); err != nil {
		return nil, err
	}
	if !cfg.Schema.IsFinalized() {
		return nil, fmt.Errorf("%w: qdrant schema", ragy.ErrInvalidArgument)
	}

	return &Store[TMeta]{client: client, collection: cfg.Collection, schema: cfg.Schema}, nil
}

// Retrieve implements retrieval.Backend.
func (s *Store[TMeta]) Retrieve(
	ctx context.Context,
	_ string,
	opts retrieval.RetrieveOptions,
) ([]retrieval.Document[TMeta], error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	if len(opts.Vector) == 0 {
		return nil, fmt.Errorf("%w: retrieve vector", ragy.ErrEmptyVector)
	}
	if err := s.Schema().ValidateSchemaIR(opts.Filters.IR()); err != nil {
		return nil, err
	}

	cond, err := renderFilter(opts.Filters.IR())
	if err != nil {
		return nil, err
	}

	points, err := s.client.Search(ctx, s.collection, opts.Vector, cond, opts.TopK)
	if err != nil {
		return nil, ragy.WrapBackendError(err, "qdrant search")
	}

	if len(points) == 0 {
		return nil, nil
	}

	docs := make([]retrieval.Document[TMeta], 0, len(points))
	for _, point := range points {
		doc, err := s.projectPoint(point, logistic(point.Score))
		if err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

// Upsert implements dense.Index.
func (s *Store[TMeta]) Upsert(ctx context.Context, records []dense.Record[TMeta]) error {
	if len(records) == 0 {
		return nil
	}

	points := make([]Point, 0, len(records))
	for _, record := range records {
		if err := record.Validate(); err != nil {
			return err
		}
		attrs, err := dense.NormalizeRecordMeta(s.schema, record.Meta)
		if err != nil {
			return err
		}

		points = append(points, Point{
			ID:         record.ID,
			Content:    record.Content,
			Attributes: filter.CloneRawAttributes(attrs),
			Vector:     append([]float32(nil), record.Vector...),
			Score:      0,
		})
	}

	return ragy.WrapBackendError(s.client.Upsert(ctx, s.collection, points), "qdrant upsert")
}

// FindByIDs implements documents.Store.
func (s *Store[TMeta]) FindByIDs(ctx context.Context, ids []string) ([]retrieval.Document[TMeta], error) {
	if len(ids) == 0 {
		return nil, nil
	}

	points, err := s.client.Get(ctx, s.collection, ids)
	if err != nil {
		return nil, ragy.WrapBackendError(err, "qdrant get")
	}

	if len(points) == 0 {
		return nil, nil
	}

	docs := make([]retrieval.Document[TMeta], 0, len(points))
	for _, point := range points {
		doc, err := s.projectPoint(point, 0)
		if err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

// DeleteByIDs implements documents.Store.
func (s *Store[TMeta]) DeleteByIDs(ctx context.Context, ids []string) (documents.DeleteResult, error) {
	if len(ids) == 0 {
		return documents.DeleteResult{}, nil
	}

	deleted, err := s.client.DeleteByIDs(ctx, s.collection, ids)
	if err != nil {
		return documents.DeleteResult{}, ragy.WrapBackendError(err, "qdrant delete by ids")
	}

	return documents.DeleteResult{Deleted: deleted}, nil
}

// DeleteByFilter implements documents.Store.
func (s *Store[TMeta]) DeleteByFilter(ctx context.Context, cond filter.Condition) (documents.DeleteResult, error) {
	ir := cond.IR()
	if filter.IsEmpty(ir) {
		return documents.DeleteResult{}, fmt.Errorf("%w: delete filter", ragy.ErrInvalidArgument)
	}
	if err := s.Schema().ValidateSchemaIR(ir); err != nil {
		return documents.DeleteResult{}, err
	}

	rendered, err := renderFilter(ir)
	if err != nil {
		return documents.DeleteResult{}, err
	}

	deleted, err := s.client.DeleteByFilter(ctx, s.collection, rendered)
	if err != nil {
		return documents.DeleteResult{}, ragy.WrapBackendError(err, "qdrant delete by filter")
	}

	return documents.DeleteResult{Deleted: deleted}, nil
}

// Schema returns the finalized filter schema used by the store.
func (s *Store[TMeta]) Schema() filter.Schema {
	return s.schema
}

func (s *Store[TMeta]) projectPoint(point Point, relevance float64) (retrieval.Document[TMeta], error) {
	normalized, err := s.schema.NormalizeAttributes(point.Attributes)
	if err != nil {
		return retrieval.Document[TMeta]{}, err
	}

	meta, err := decodeMeta[TMeta](normalized)
	if err != nil {
		return retrieval.Document[TMeta]{}, err
	}

	doc := retrieval.Document[TMeta]{
		ID:      point.ID,
		Content: point.Content,
		Score:   ragy.ClampScore(relevance),
		Meta:    meta,
	}
	if err := retrieval.ValidateDocument(doc); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	return doc, nil
}

func decodeMeta[TMeta any](attrs filter.RawAttributes) (TMeta, error) {
	var meta TMeta
	if len(attrs) == 0 {
		return meta, nil
	}
	data, err := json.Marshal(attrs)
	if err != nil {
		return meta, err
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		return meta, err
	}
	return meta, nil
}

func renderFilter(expr filter.IR) (Condition, error) {
	walker := &conditionWalker{stack: nil, result: nil}
	if err := filter.Walk(expr, walker); err != nil {
		return nil, err
	}
	return walker.result, nil
}

type conditionFrame struct {
	op    string
	items []Condition
}

type conditionWalker struct {
	stack  []conditionFrame
	result Condition
}

func (w *conditionWalker) OnEmpty() error {
	return w.push(MatchAllCondition{})
}

func (w *conditionWalker) OnEq(field string, value filter.Value) error {
	return w.push(EqCondition{Field: field, Value: value.Raw()})
}

func (w *conditionWalker) OnNeq(field string, value filter.Value) error {
	return w.push(NeqCondition{Field: field, Value: value.Raw()})
}

func (w *conditionWalker) OnGt(field string, value filter.Value) error {
	return w.push(RangeCondition{Field: field, Op: "gt", Value: value.Raw()})
}

func (w *conditionWalker) OnGte(field string, value filter.Value) error {
	return w.push(RangeCondition{Field: field, Op: "gte", Value: value.Raw()})
}

func (w *conditionWalker) OnLt(field string, value filter.Value) error {
	return w.push(RangeCondition{Field: field, Op: "lt", Value: value.Raw()})
}

func (w *conditionWalker) OnLte(field string, value filter.Value) error {
	return w.push(RangeCondition{Field: field, Op: "lte", Value: value.Raw()})
}

func (w *conditionWalker) OnIn(field string, values []filter.Value) error {
	items := make([]any, 0, len(values))
	for _, value := range values {
		items = append(items, value.Raw())
	}
	return w.push(InCondition{Field: field, Values: items})
}

func (w *conditionWalker) EnterAnd(_ int) error {
	w.stack = append(w.stack, conditionFrame{op: "and", items: nil})
	return nil
}

func (w *conditionWalker) LeaveAnd() error {
	return w.leaveGroup("and")
}

func (w *conditionWalker) EnterOr(_ int) error {
	w.stack = append(w.stack, conditionFrame{op: "or", items: nil})
	return nil
}

func (w *conditionWalker) LeaveOr() error {
	return w.leaveGroup("or")
}

func (w *conditionWalker) EnterNot() error {
	w.stack = append(w.stack, conditionFrame{op: "not", items: nil})
	return nil
}

func (w *conditionWalker) LeaveNot() error {
	frame, err := w.pop("not")
	if err != nil {
		return err
	}
	if len(frame.items) != 1 {
		return fmt.Errorf("%w: invalid NOT filter", ragy.ErrUnsupported)
	}
	return w.push(NotCondition{Item: frame.items[0]})
}

func (w *conditionWalker) leaveGroup(op string) error {
	frame, err := w.pop(op)
	if err != nil {
		return err
	}
	return w.push(GroupCondition{Op: op, Items: append([]Condition(nil), frame.items...)})
}

func (w *conditionWalker) push(condition Condition) error {
	if len(w.stack) == 0 {
		w.result = condition
		return nil
	}

	last := len(w.stack) - 1
	w.stack[last].items = append(w.stack[last].items, condition)
	return nil
}

func (w *conditionWalker) pop(op string) (conditionFrame, error) {
	if len(w.stack) == 0 {
		return conditionFrame{}, fmt.Errorf("%w: unmatched %s filter", ragy.ErrUnsupported, op)
	}

	last := len(w.stack) - 1
	frame := w.stack[last]
	w.stack = w.stack[:last]
	if frame.op != op {
		return conditionFrame{}, fmt.Errorf("%w: unexpected filter group %q", ragy.ErrUnsupported, frame.op)
	}

	return frame, nil
}

func logistic(score float64) float64 {
	score = math.Max(-logisticClamp, math.Min(logisticClamp, score))
	return 1.0 / (1.0 + math.Exp(-score))
}

var (
	_ retrieval.Backend[any] = (*Store[any])(nil)
	_ dense.Index[any]       = (*Store[any])(nil)
	_ documents.Store[any]   = (*Store[any])(nil)
)
