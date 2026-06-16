package lexical

import (
	"context"
	"fmt"
	"maps"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

const defaultBM25K1 = 1.2
const defaultBM25B = 0.75
const bm25IDFSmoothing = 0.5

// Config configures in-memory BM25 indexing and retrieval.
type Config[TMeta any] struct {
	SearchFields []string
	K1           float64
	B            float64
	Resolver     retrieval.IdentityResolver[TMeta]
	// Codec overrides metadata codec for filter matching. Defaults to JSONCodec when nil.
	Codec retrieval.MetadataCodec[TMeta]
}

// BM25Index is a thread-safe in-memory BM25 lexical index.
type BM25Index[TMeta any] struct {
	mu         sync.RWMutex
	schema     filter.Schema
	config     Config[TMeta]
	tokenizer  Tokenizer
	synonyms   SynonymMap
	resolver   retrieval.IdentityResolver[TMeta]
	codec      retrieval.MetadataCodec[TMeta]
	docs       map[string]retrieval.Document[TMeta]
	docLengths map[string]int
	avgLength  float64
	postings   map[string]map[string]int
	docCount   int
}

// NewBM25Index constructs an empty BM25 index.
func NewBM25Index[TMeta any](
	schema filter.Schema,
	config Config[TMeta],
	tokenizer Tokenizer,
	synonyms SynonymMap,
) (*BM25Index[TMeta], error) {
	if !schema.IsFinalized() {
		return nil, fmt.Errorf("%w: lexical schema", ragy.ErrInvalidArgument)
	}
	if len(config.SearchFields) == 0 {
		return nil, fmt.Errorf("%w: lexical search fields", ragy.ErrInvalidArgument)
	}
	if err := ValidateSearchFields(schema, config.SearchFields); err != nil {
		return nil, err
	}
	if tokenizer == nil {
		tokenizer = DefaultTokenizer{}
	}
	k1 := config.K1
	if k1 <= 0 {
		k1 = defaultBM25K1
	}
	b := config.B
	if b <= 0 {
		b = defaultBM25B
	}
	config.K1 = k1
	config.B = b
	codec := config.Codec
	if codec == nil {
		codec = retrieval.NewJSONCodec[TMeta](schema)
	}

	return &BM25Index[TMeta]{
		schema:     schema,
		config:     config,
		tokenizer:  tokenizer,
		synonyms:   synonyms,
		resolver:   retrieval.DefaultResolver(config.Resolver),
		codec:      codec,
		docs:       make(map[string]retrieval.Document[TMeta]),
		docLengths: make(map[string]int),
		postings:   make(map[string]map[string]int),
	}, nil
}

// Index replaces all documents in the index.
func (idx *BM25Index[TMeta]) Index(docs []retrieval.Document[TMeta]) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.docs = make(map[string]retrieval.Document[TMeta], len(docs))
	idx.docLengths = make(map[string]int, len(docs))
	idx.postings = make(map[string]map[string]int)
	idx.docCount = 0
	idx.avgLength = 0

	for _, doc := range docs {
		if err := retrieval.ValidateDocument(doc); err != nil {
			return ragy.WrapProjectionError(err, "bm25 index validate")
		}
		if err := idx.upsertLocked(doc); err != nil {
			return err
		}
	}
	return nil
}

// Upsert inserts or updates one document.
func (idx *BM25Index[TMeta]) Upsert(doc retrieval.Document[TMeta]) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.upsertLocked(doc)
}

func (idx *BM25Index[TMeta]) upsertLocked(doc retrieval.Document[TMeta]) error {
	if err := retrieval.ValidateDocument(doc); err != nil {
		return ragy.WrapProjectionError(err, "bm25 upsert validate")
	}
	tokens, length, err := idx.documentTokens(doc)
	if err != nil {
		return err
	}
	if length == 0 {
		return fmt.Errorf("%w: document has no indexable tokens", ragy.ErrEmptyText)
	}
	if existing, ok := idx.docs[doc.ID]; ok {
		if err := idx.removeLocked(existing); err != nil {
			return err
		}
	}
	idx.docs[doc.ID] = doc
	idx.docLengths[doc.ID] = length
	idx.docCount++
	idx.avgLength = idx.recomputeAvgLength()
	for _, token := range tokens {
		if idx.postings[token] == nil {
			idx.postings[token] = make(map[string]int)
		}
		idx.postings[token][doc.ID]++
	}
	return nil
}

func (idx *BM25Index[TMeta]) removeLocked(doc retrieval.Document[TMeta]) error {
	tokens, _, err := idx.documentTokens(doc)
	if err != nil {
		return err
	}
	for _, token := range tokens {
		posting := idx.postings[token]
		if posting == nil {
			continue
		}
		delete(posting, doc.ID)
		if len(posting) == 0 {
			delete(idx.postings, token)
		}
	}
	delete(idx.docs, doc.ID)
	delete(idx.docLengths, doc.ID)
	idx.docCount--
	idx.avgLength = idx.recomputeAvgLength()
	return nil
}

func (idx *BM25Index[TMeta]) recomputeAvgLength() float64 {
	if idx.docCount == 0 {
		return 0
	}
	total := 0
	for _, length := range idx.docLengths {
		total += length
	}
	return float64(total) / float64(idx.docCount)
}

func (idx *BM25Index[TMeta]) documentTokens(doc retrieval.Document[TMeta]) ([]string, int, error) {
	parts := make([]string, 0, len(idx.config.SearchFields))
	for _, field := range idx.config.SearchFields {
		var (
			value string
			err   error
		)
		if field == "content" {
			value = doc.Content
		} else {
			value, err = idx.fieldValue(doc, field)
			if err != nil {
				return nil, 0, err
			}
		}
		if value != "" {
			parts = append(parts, value)
		}
	}
	text := strings.Join(parts, " ")
	tokens := idx.tokenizer.Tokenize(text)
	return tokens, len(tokens), nil
}

func (idx *BM25Index[TMeta]) fieldValue(doc retrieval.Document[TMeta], field string) (string, error) {
	attrs, err := idx.codec.Encode(doc.Meta)
	if err != nil {
		return "", err
	}
	raw, ok := attrs[field]
	if !ok {
		return "", nil
	}
	kind, declared := idx.schema.Lookup(field)
	if !declared {
		return "", fmt.Errorf("%w: undeclared schema field %q", ragy.ErrInvalidArgument, field)
	}
	switch kind {
	case filter.KindString:
		value, ok := raw.(string)
		if !ok {
			return "", fmt.Errorf("%w: field %q must be string", ragy.ErrInvalidArgument, field)
		}
		return value, nil
	case filter.KindInt:
		return formatIntFieldValue(raw, field)
	case filter.KindBool:
		value, ok := raw.(bool)
		if !ok {
			return "", fmt.Errorf("%w: field %q must be bool", ragy.ErrInvalidArgument, field)
		}
		return strconv.FormatBool(value), nil
	case filter.KindFloat:
		switch value := raw.(type) {
		case float64:
			return strconv.FormatFloat(value, 'f', -1, 64), nil
		case float32:
			return strconv.FormatFloat(float64(value), 'f', -1, 32), nil
		default:
			return "", fmt.Errorf("%w: field %q must be float", ragy.ErrInvalidArgument, field)
		}
	default:
		return "", fmt.Errorf("%w: unsupported field kind %q", ragy.ErrInvalidArgument, kind)
	}
}

func formatIntFieldValue(raw any, field string) (string, error) {
	switch value := raw.(type) {
	case int:
		return strconv.FormatInt(int64(value), 10), nil
	case int64:
		return strconv.FormatInt(value, 10), nil
	case float64:
		if value != float64(int64(value)) {
			return "", fmt.Errorf("%w: field %q must be int", ragy.ErrInvalidArgument, field)
		}
		return strconv.FormatInt(int64(value), 10), nil
	default:
		return "", fmt.Errorf("%w: field %q must be int", ragy.ErrInvalidArgument, field)
	}
}

// Retrieve scores documents for a query and returns a ranked ResultSet.
func (idx *BM25Index[TMeta]) Retrieve(
	ctx context.Context,
	query string,
	opts retrieval.RetrieveOptions,
) (retrieval.ResultSet[TMeta], error) {
	if err := opts.Validate(); err != nil {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), err
	}
	if strings.TrimSpace(query) == "" {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver),
			fmt.Errorf("%w: retrieve query", ragy.ErrEmptyText)
	}
	if err := idx.schema.ValidateSchemaIR(opts.Filters.IR()); err != nil {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), err
	}

	idx.mu.RLock()
	snapshot := idx.snapshotLocked()
	idx.mu.RUnlock()

	if err := ctx.Err(); err != nil {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), err
	}

	queryTokens := idx.synonyms.Expand(idx.tokenizer.Tokenize(query))
	if len(queryTokens) == 0 {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), nil
	}

	scores := idx.scoreQuery(snapshot, queryTokens)
	if len(scores) == 0 {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), nil
	}

	filteredScores, err := idx.filterScoredDocs(snapshot, scores, opts.Filters, idx.codec)
	docs := idx.rankScoredDocs(snapshot, filteredScores, opts.BackendFetchLimit())
	rs := retrieval.NewResultSet(docs, idx.resolver)
	return retrieval.PreserveResultOnError(rs, err, idx.resolver)
}

func (idx *BM25Index[TMeta]) scoreQuery(snapshot bm25Snapshot[TMeta], queryTokens []string) map[string]float64 {
	avgLength := snapshot.avgLength
	if avgLength <= 0 {
		return nil
	}

	scores := make(map[string]float64)
	for _, token := range queryTokens {
		posting := snapshot.postings[token]
		if len(posting) == 0 {
			continue
		}
		df := len(posting)
		idf := math.Log(1 + (float64(snapshot.docCount)-float64(df)+bm25IDFSmoothing)/(float64(df)+bm25IDFSmoothing))
		for docID, tf := range posting {
			docLen := float64(snapshot.docLengths[docID])
			numerator := float64(tf) * (idx.config.K1 + 1)
			denominator := float64(tf) + idx.config.K1*(1-idx.config.B+idx.config.B*docLen/avgLength)
			scores[docID] += idf * numerator / denominator
		}
	}
	return scores
}

func (idx *BM25Index[TMeta]) filterScoredDocs(
	snapshot bm25Snapshot[TMeta],
	scores map[string]float64,
	cond filter.Condition,
	codec retrieval.MetadataCodec[TMeta],
) (map[string]float64, error) {
	if filter.IsEmpty(cond.IR()) {
		return scores, nil
	}

	filtered := make(map[string]float64, len(scores))
	docIDs := make([]string, 0, len(scores))
	for docID := range scores {
		docIDs = append(docIDs, docID)
	}
	sort.Strings(docIDs)
	for _, docID := range docIDs {
		score := scores[docID]
		doc := snapshot.docs[docID]
		matched, err := retrieval.MatchDocument(codec, doc, cond)
		if err != nil {
			return filtered, ragy.WrapProjectionError(err, "bm25 filter match")
		}
		if matched {
			filtered[docID] = score
		}
	}
	return filtered, nil
}

func (idx *BM25Index[TMeta]) rankScoredDocs(
	snapshot bm25Snapshot[TMeta],
	scores map[string]float64,
	limit int,
) []retrieval.Document[TMeta] {
	type scored struct {
		id    string
		score float64
	}
	docIDs := make([]string, 0, len(scores))
	for id := range scores {
		docIDs = append(docIDs, id)
	}
	sort.Strings(docIDs)

	ranked := make([]scored, 0, len(docIDs))
	maxScore := 0.0
	for _, id := range docIDs {
		score := scores[id]
		if score > maxScore {
			maxScore = score
		}
		ranked = append(ranked, scored{id: id, score: score})
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		return ranked[i].score > ranked[j].score
	})

	if limit <= 0 {
		limit = len(ranked)
	}
	if limit > len(ranked) {
		limit = len(ranked)
	}

	docs := make([]retrieval.Document[TMeta], 0, limit)
	for _, item := range ranked[:limit] {
		doc := snapshot.docs[item.id]
		if maxScore > 0 {
			doc.Score = ragy.ClampScore(item.score / maxScore)
		}
		docs = append(docs, doc)
	}
	return docs
}

type bm25Snapshot[TMeta any] struct {
	docs       map[string]retrieval.Document[TMeta]
	docLengths map[string]int
	postings   map[string]map[string]int
	docCount   int
	avgLength  float64
}

func (idx *BM25Index[TMeta]) snapshotLocked() bm25Snapshot[TMeta] {
	docs := make(map[string]retrieval.Document[TMeta], len(idx.docs))
	maps.Copy(docs, idx.docs)
	lengths := make(map[string]int, len(idx.docLengths))
	maps.Copy(lengths, idx.docLengths)
	postings := make(map[string]map[string]int, len(idx.postings))
	for term, posting := range idx.postings {
		copyPosting := make(map[string]int, len(posting))
		maps.Copy(copyPosting, posting)
		postings[term] = copyPosting
	}
	return bm25Snapshot[TMeta]{
		docs:       docs,
		docLengths: lengths,
		postings:   postings,
		docCount:   idx.docCount,
		avgLength:  idx.avgLength,
	}
}

// Schema returns the configured filter schema.
func (idx *BM25Index[TMeta]) Schema() filter.Schema {
	return idx.schema
}

// LexicalBackend marks BM25Index as a lexical backend.
func (idx *BM25Index[TMeta]) LexicalBackend() {}

var _ Backend[any] = (*BM25Index[any])(nil)
