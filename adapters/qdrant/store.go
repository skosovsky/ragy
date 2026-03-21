// Package qdrant provides a ragy.VectorStore implementation using Qdrant (gRPC client).
//
// Search uses only req.DenseVector. Not yet supported (per Adapters.md 2026 roadmap): req.TensorVector
// (ColBERT/Late Interaction / multivector) and Sparse vectors (BM25). Full filter.Expr translation to
// Qdrant Filter (Must/Should/MustNot) is supported. Upsert returns an error if a document has no embedding.
package qdrant

import (
	"context"
	"fmt"
	"hash/fnv"
	"iter"
	"math"
	"regexp"
	"strconv"

	"github.com/qdrant/go-client/qdrant"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

const (
	embeddingKey     = "embedding"
	ragyIDPayloadKey = "_ragy_id" // Original document ID stored in payload for round-trip
	defaultBatchSize = 500
	// logisticClamp limits the input to logisticConfidence before Exp (numerical stability).
	logisticClamp = 20.0
)

var fieldSanitize = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)

// Store implements ragy.VectorStore using Qdrant go-client.
type Store struct {
	client         *qdrant.Client
	collectionName string
	batchSize      int
}

// Option configures the Store.
type Option func(*Store)

// WithCollectionName sets the collection name (required).
func WithCollectionName(name string) Option {
	return func(s *Store) { s.collectionName = name }
}

// WithBatchSize sets the micro-batch size for Upsert (default 500).
func WithBatchSize(n int) Option {
	return func(s *Store) { s.batchSize = n }
}

// New returns a new Qdrant Store. Client must be connected; collection must exist (dense or multivector).
func New(client *qdrant.Client, opts ...Option) *Store {
	s := &Store{
		client:         client,
		collectionName: "ragy",
		batchSize:      defaultBatchSize,
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

// Search implements ragy.VectorStore.
func (s *Store) Search(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	if len(req.DenseVector) == 0 {
		if len(req.SparseVector) > 0 {
			return nil, fmt.Errorf("qdrant: %w", ragy.ErrSparseVectorNotSupported)
		}
		return []ragy.Document{}, nil
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	offset := max(req.Offset, 0)
	qFilter := buildQdrantFilter(req.Filter)
	lim := uint64(limit + offset)
	queryPoints := &qdrant.QueryPoints{
		CollectionName: s.collectionName,
		Query:          qdrant.NewQuery(req.DenseVector...),
		Limit:          &lim,
		WithPayload:    qdrant.NewWithPayload(true),
		WithVectors:    qdrant.NewWithVectors(false),
	}
	if qFilter != nil {
		queryPoints.Filter = qFilter
	}
	result, err := s.client.Query(ctx, queryPoints)
	if err != nil {
		return nil, fmt.Errorf("qdrant query: %w", err)
	}
	if len(result) == 0 {
		return []ragy.Document{}, nil
	}
	// Apply offset in Go (Qdrant returns top limit+offset)
	start := min(offset, len(result))
	slice := result[start:]
	out := make([]ragy.Document, 0, len(slice))
	for _, sp := range slice {
		doc := scoredPointToDocument(sp)
		out = append(out, doc)
	}
	return out, nil
}

// Stream implements ragy.VectorStore.
func (s *Store) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := s.Search(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

// Upsert implements ragy.VectorStore. Micro-batches internally.
func (s *Store) Upsert(ctx context.Context, docs []ragy.Document) error {
	if len(docs) == 0 {
		return nil
	}
	for i := 0; i < len(docs); i += s.batchSize {
		end := min(i+s.batchSize, len(docs))
		batch := docs[i:end]
		if err := s.upsertBatch(ctx, batch); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) upsertBatch(ctx context.Context, docs []ragy.Document) error {
	points := make([]*qdrant.PointStruct, 0, len(docs))
	for _, d := range docs {
		emb, _ := d.Metadata[embeddingKey].([]float32)
		if len(emb) == 0 {
			return fmt.Errorf("qdrant: document %q missing embedding", d.ID)
		}
		payload := make(map[string]any)
		for k, v := range d.Metadata {
			if k == embeddingKey {
				continue
			}
			payload[k] = v
		}
		payload["content"] = d.Content
		payload[ragyIDPayloadKey] = d.ID
		points = append(points, &qdrant.PointStruct{
			Id:      qdrant.NewIDNum(idToUint64(d.ID)),
			Vectors: qdrant.NewVectors(emb...),
			Payload: qdrant.NewValueMap(payload),
		})
	}
	if len(points) == 0 {
		return nil
	}
	_, err := s.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: s.collectionName,
		Points:         points,
	})
	return err
}

// DeleteByFilter implements ragy.VectorStore.
func (s *Store) DeleteByFilter(ctx context.Context, f filter.Expr) error {
	if f == nil {
		return nil
	}
	qFilter := buildQdrantFilter(f)
	if qFilter == nil {
		return nil
	}
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collectionName,
		Points:         qdrant.NewPointsSelectorFilter(qFilter),
	})
	return err
}

func idToUint64(id string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(id))
	return h.Sum64()
}

//nolint:gocognit,nestif // Qdrant ScoredPoint payload map requires nested value-kind switches.
func scoredPointToDocument(sp *qdrant.ScoredPoint) ragy.Document {
	doc := ragy.Document{}
	if sp != nil && sp.GetId() != nil {
		if num, ok := sp.GetId().GetPointIdOptions().(*qdrant.PointId_Num); ok {
			doc.ID = strconv.FormatUint(num.Num, 10)
		}
	}
	if sp != nil && sp.Payload != nil {
		for k, v := range sp.GetPayload() {
			if k == "content" {
				if v != nil && v.Kind != nil {
					if s, ok := v.GetKind().(*qdrant.Value_StringValue); ok {
						doc.Content = s.StringValue
					}
				}
				continue
			}
			if k == ragyIDPayloadKey {
				if v != nil && v.Kind != nil {
					if s, ok := v.GetKind().(*qdrant.Value_StringValue); ok {
						doc.ID = s.StringValue
					}
				}
				continue
			}
			if doc.Metadata == nil {
				doc.Metadata = make(map[string]any)
			}
			doc.Metadata[k] = valueToAny(v)
		}
	}
	if sp != nil {
		doc.Score = sp.GetScore()
		doc.Confidence = logisticConfidence(float64(sp.GetScore()))
	}
	return doc
}

func logisticConfidence(s float64) float64 {
	s = math.Min(logisticClamp, math.Max(-logisticClamp, s))
	return 1.0 / (1.0 + math.Exp(-s))
}

func valueToAny(v *qdrant.Value) any {
	if v == nil || v.Kind == nil {
		return nil
	}
	switch k := v.GetKind().(type) {
	case *qdrant.Value_StringValue:
		return k.StringValue
	case *qdrant.Value_IntegerValue:
		return k.IntegerValue
	case *qdrant.Value_DoubleValue:
		return k.DoubleValue
	case *qdrant.Value_BoolValue:
		return k.BoolValue
	case *qdrant.Value_ListValue:
		arr := make([]any, len(k.ListValue.GetValues()))
		for i, e := range k.ListValue.GetValues() {
			arr[i] = valueToAny(e)
		}
		return arr
	case *qdrant.Value_StructValue:
		m := make(map[string]any)
		for key, val := range k.StructValue.GetFields() {
			m[key] = valueToAny(val)
		}
		return m
	default:
		return nil
	}
}

// filterValueToString converts filter value (any) to string for NewMatch.
func filterValueToString(v any) string {
	if v == nil {
		return ""
	}
	switch x := v.(type) {
	case string:
		return x
	case int:
		return strconv.FormatInt(int64(x), 10)
	case int64:
		return strconv.FormatInt(x, 10)
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(x), 'f', -1, 32)
	default:
		return fmt.Sprint(v)
	}
}

// anyToFloat64Ptr converts numeric filter value to *float64 for qdrant.Range (Gt/Gte/Lt/Lte).
func anyToFloat64Ptr(v any) *float64 {
	if v == nil {
		return nil
	}
	switch x := v.(type) {
	case int:
		f := float64(x)
		return &f
	case int32:
		f := float64(x)
		return &f
	case int64:
		f := float64(x)
		return &f
	case float32:
		f := float64(x)
		return &f
	case float64:
		return &x
	default:
		return nil
	}
}

// buildQdrantFilter converts filter.Expr to qdrant.Filter (Must/Should/MustNot).
func buildQdrantFilter(expr filter.Expr) *qdrant.Filter {
	if expr == nil {
		return nil
	}
	return buildQdrantFilterRec(expr)
}

//nolint:gocognit,cyclop,funlen // filter.Expr → Qdrant Filter is a large switch with nested And/Or handling.
func buildQdrantFilterRec(expr filter.Expr) *qdrant.Filter {
	switch e := expr.(type) {
	case filter.Eq:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return &qdrant.Filter{Must: []*qdrant.Condition{qdrant.NewMatch(e.Field, filterValueToString(e.Value))}}
	case filter.Neq:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return &qdrant.Filter{MustNot: []*qdrant.Condition{qdrant.NewMatch(e.Field, filterValueToString(e.Value))}}
	case filter.Gt:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewRange(e.Field, &qdrant.Range{Gte: nil, Gt: anyToFloat64Ptr(e.Value), Lte: nil, Lt: nil}),
			},
		}
	case filter.Gte:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewRange(e.Field, &qdrant.Range{Gte: anyToFloat64Ptr(e.Value), Gt: nil, Lte: nil, Lt: nil}),
			},
		}
	case filter.Lt:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewRange(e.Field, &qdrant.Range{Gte: nil, Gt: nil, Lte: nil, Lt: anyToFloat64Ptr(e.Value)}),
			},
		}
	case filter.Lte:
		if !fieldSanitize.MatchString(e.Field) {
			return nil
		}
		return &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewRange(e.Field, &qdrant.Range{Gte: nil, Gt: nil, Lte: anyToFloat64Ptr(e.Value), Lt: nil}),
			},
		}
	case filter.In:
		if !fieldSanitize.MatchString(e.Field) || len(e.Values) == 0 {
			return nil
		}
		conds := make([]*qdrant.Condition, 0, len(e.Values))
		for _, v := range e.Values {
			conds = append(conds, qdrant.NewMatch(e.Field, filterValueToString(v)))
		}
		return &qdrant.Filter{Should: conds}
	case filter.And:
		var must []*qdrant.Condition
		for _, sub := range e.Exprs {
			subF := buildQdrantFilterRec(sub)
			if subF != nil {
				if len(subF.GetMust()) > 0 {
					must = append(must, subF.GetMust()...)
				}
				if len(subF.GetMustNot()) > 0 {
					must = append(must, qdrant.NewNestedFilter("", subF))
				}
				if len(subF.GetShould()) > 0 {
					must = append(must, qdrant.NewNestedFilter("", subF))
				}
			}
		}
		if len(must) == 0 {
			return nil
		}
		return &qdrant.Filter{Must: must}
	case filter.Or:
		var should []*qdrant.Condition
		for _, sub := range e.Exprs {
			subF := buildQdrantFilterRec(sub)
			if subF != nil {
				should = append(should, qdrant.NewNestedFilter("", subF))
			}
		}
		if len(should) == 0 {
			return nil
		}
		return &qdrant.Filter{Should: should}
	case filter.Not:
		subF := buildQdrantFilterRec(e.Expr)
		if subF == nil {
			return nil
		}
		return &qdrant.Filter{MustNot: []*qdrant.Condition{qdrant.NewNestedFilter("", subF)}}
	default:
		return nil
	}
}

// FetchParents implements ragy.HierarchyRetriever. Reads payload[ragy.ParentDocumentIDKey] from child points, then loads parent points by business id (same id scheme as Upsert).
func (s *Store) FetchParents(ctx context.Context, childIDs []string) ([]ragy.Document, error) {
	if len(childIDs) == 0 {
		return nil, nil
	}
	ids := make([]*qdrant.PointId, 0, len(childIDs))
	for _, id := range childIDs {
		ids = append(ids, qdrant.NewIDNum(idToUint64(id)))
	}
	children, err := s.client.Get(ctx, &qdrant.GetPoints{
		CollectionName: s.collectionName,
		Ids:            ids,
		WithPayload:    qdrant.NewWithPayload(true),
		WithVectors:    qdrant.NewWithVectors(false),
	})
	if err != nil {
		return nil, err
	}
	parentSeen := make(map[string]struct{})
	var parentIDs []string
	for _, p := range children {
		doc := retrievedPointToDocument(p)
		raw, ok := doc.Metadata[ragy.ParentDocumentIDKey]
		if !ok || raw == nil {
			continue
		}
		pid, ok := raw.(string)
		if !ok || pid == "" {
			continue
		}
		if _, dup := parentSeen[pid]; dup {
			continue
		}
		parentSeen[pid] = struct{}{}
		parentIDs = append(parentIDs, pid)
	}
	if len(parentIDs) == 0 {
		return nil, nil
	}
	pids := make([]*qdrant.PointId, 0, len(parentIDs))
	for _, id := range parentIDs {
		pids = append(pids, qdrant.NewIDNum(idToUint64(id)))
	}
	parents, err := s.client.Get(ctx, &qdrant.GetPoints{
		CollectionName: s.collectionName,
		Ids:            pids,
		WithPayload:    qdrant.NewWithPayload(true),
		WithVectors:    qdrant.NewWithVectors(false),
	})
	if err != nil {
		return nil, err
	}
	out := make([]ragy.Document, 0, len(parents))
	for _, p := range parents {
		out = append(out, retrievedPointToDocument(p))
	}
	return out, nil
}

//nolint:gocognit,nestif // Qdrant RetrievedPoint payload map requires nested value-kind switches.
func retrievedPointToDocument(p *qdrant.RetrievedPoint) ragy.Document {
	doc := ragy.Document{}
	if p != nil && p.GetId() != nil {
		if num, ok := p.GetId().GetPointIdOptions().(*qdrant.PointId_Num); ok {
			doc.ID = strconv.FormatUint(num.Num, 10)
		}
	}
	if p != nil && p.Payload != nil {
		for k, v := range p.GetPayload() {
			if k == "content" {
				if v != nil && v.Kind != nil {
					if sv, ok := v.GetKind().(*qdrant.Value_StringValue); ok {
						doc.Content = sv.StringValue
					}
				}
				continue
			}
			if k == ragyIDPayloadKey {
				if v != nil && v.Kind != nil {
					if sv, ok := v.GetKind().(*qdrant.Value_StringValue); ok {
						doc.ID = sv.StringValue
					}
				}
				continue
			}
			if doc.Metadata == nil {
				doc.Metadata = make(map[string]any)
			}
			doc.Metadata[k] = valueToAny(v)
		}
	}
	return doc
}

var (
	_ ragy.VectorStore        = (*Store)(nil)
	_ ragy.HierarchyRetriever = (*Store)(nil)
)
