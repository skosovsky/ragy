package lexical

import (
	"context"
	"fmt"
	"maps"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/skosovsky/ragy/access"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/readfailure"
	"github.com/skosovsky/ragy/retrieval"
)

const defaultBM25K1 = 1.2
const defaultBM25B = 0.75
const bm25IDFSmoothing = 0.5

// Config configures in-memory BM25 indexing and retrieval.
// Zero K1/B select defaults; explicit values require finite K1 > 0 and B in (0,1].
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
	mu                      sync.RWMutex
	snapshotReadFingerprint string
	schema                  filter.Schema
	config                  Config[TMeta]
	tokenizer               Tokenizer
	synonyms                SynonymMap
	resolver                retrieval.IdentityResolver[TMeta]
	codec                   retrieval.MetadataCodec[TMeta]
	docs                    map[string]retrieval.Document[TMeta]
	docLengths              map[string]int
	avgLength               float64
	totalLength             int
	postings                map[string]map[string]int
	docCount                int
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
	if math.IsNaN(config.K1) || math.IsInf(config.K1, 0) || config.K1 < 0 ||
		math.IsNaN(config.B) || math.IsInf(config.B, 0) || config.B < 0 || config.B > 1 {
		return nil, fmt.Errorf("%w: BM25 requires finite K1 >= 0 and B in [0,1]", ragy.ErrInvalidArgument)
	}
	k1 := config.K1
	if k1 == 0 {
		k1 = defaultBM25K1
	}
	b := config.B
	if b == 0 {
		b = defaultBM25B
	}
	config.SearchFields = slices.Clone(config.SearchFields)
	ownedSynonyms := make(SynonymMap, len(synonyms))
	for term, variants := range synonyms {
		ownedSynonyms[term] = slices.Clone(variants)
	}
	config.K1 = k1
	config.B = b
	codec := config.Codec
	if codec == nil {
		codec = retrieval.NewJSONCodec[TMeta](schema)
	}

	return &BM25Index[TMeta]{
		snapshotReadFingerprint: "",
		schema:                  schema,
		config:                  config,
		tokenizer:               tokenizer,
		synonyms:                ownedSynonyms,
		resolver:                retrieval.DefaultResolver(config.Resolver),
		codec:                   codec,
		docs:                    make(map[string]retrieval.Document[TMeta]),
		docLengths:              make(map[string]int),
		postings:                make(map[string]map[string]int),
	}, nil
}

// Index replaces all documents atomically. A failed rebuild preserves the old index.
// Rebuilds and Upsert are serialized; readers see a complete published snapshot.
func (idx *BM25Index[TMeta]) Index(docs []retrieval.Document[TMeta]) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	staged := &BM25Index[TMeta]{
		snapshotReadFingerprint: idx.snapshotReadFingerprint,
		schema:                  idx.schema,
		config:                  idx.config,
		tokenizer:               idx.tokenizer,
		synonyms:                idx.synonyms,
		resolver:                idx.resolver,
		codec:                   idx.codec,
		docs:                    make(map[string]retrieval.Document[TMeta], len(docs)),
		docLengths:              make(map[string]int, len(docs)),
		postings:                make(map[string]map[string]int),
	}

	for _, doc := range docs {
		if err := retrieval.ValidateDocument(doc); err != nil {
			return ragy.WrapProjectionError(err, "bm25 index validate")
		}
		if err := staged.upsertLocked(doc); err != nil {
			return err
		}
	}
	idx.docs = staged.docs
	idx.docLengths = staged.docLengths
	idx.postings = staged.postings
	idx.docCount = staged.docCount
	idx.avgLength = staged.avgLength
	idx.totalLength = staged.totalLength
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
	doc.SourceSupports = append(doc.SourceSupports[:0:0], doc.SourceSupports...)
	doc.ScoreHistory = append(doc.ScoreHistory[:0:0], doc.ScoreHistory...)
	idx.docs[doc.ID] = doc
	idx.docLengths[doc.ID] = length
	idx.docCount++
	idx.totalLength += length
	idx.updateAvgLength()
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
	idx.totalLength -= idx.docLengths[doc.ID]
	delete(idx.docLengths, doc.ID)
	idx.docCount--
	idx.updateAvgLength()
	return nil
}

func (idx *BM25Index[TMeta]) updateAvgLength() {
	idx.avgLength = 0
	if idx.docCount != 0 {
		idx.avgLength = float64(idx.totalLength) / float64(idx.docCount)
	}
}

func (idx *BM25Index[TMeta]) documentTokens(doc retrieval.Document[TMeta]) ([]string, int, error) {
	parts := make([]string, 0, len(idx.config.SearchFields))
	for _, field := range idx.config.SearchFields {
		var (
			value string
			err   error
		)
		if field == contentSearchField {
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
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[TMeta], error) {
	rs, err := idx.retrieve(ctx, req)
	err = readfailure.Check(ctx, req.Read, err)
	if access.IsProtectionFailure(err) {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), err
	}
	return rs, err
}

func (idx *BM25Index[TMeta]) retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[TMeta], error) {
	if idx.snapshotReadFingerprint != "" {
		if err := req.Read.Check(ctx); err != nil {
			return retrieval.NewResultSet[TMeta](nil, idx.resolver), err
		}
		fingerprint, err := req.Read.Fingerprint()
		if err != nil {
			return retrieval.NewResultSet[TMeta](nil, idx.resolver), err
		}
		if fingerprint != idx.snapshotReadFingerprint {
			return retrieval.NewResultSet[TMeta](nil, idx.resolver), access.Protect(ragy.ErrUnavailable)
		}
	}
	query := req.EffectiveText()
	prepared, readErr := retrieval.PrepareRead(ctx, req, idx)
	if readErr != nil {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), readErr
	}
	req = prepared
	opts := req.Options
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

	if err := ctx.Err(); err != nil {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), err
	}
	queryTokens := idx.synonyms.Expand(idx.tokenizer.Tokenize(query))
	if len(queryTokens) == 0 {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), nil
	}
	idx.mu.RLock()
	snapshot := idx.snapshotLocked(queryTokens)
	idx.mu.RUnlock()

	scores := idx.scoreQuery(snapshot, queryTokens)
	for _, score := range scores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return retrieval.NewResultSet[TMeta](nil, idx.resolver),
				fmt.Errorf("%w: non-finite BM25 score", ragy.ErrProtocol)
		}
	}
	if len(scores) == 0 {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), nil
	}

	filteredScores, err := idx.filterScoredDocs(ctx, req.Read, snapshot, scores, opts.Filters, idx.codec)
	if access.IsProtectionFailure(err) {
		return retrieval.NewResultSet[TMeta](nil, idx.resolver), access.Protect(err)
	}
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
	ctx context.Context,
	read access.Binding,
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
		if err := read.Check(ctx); err != nil {
			return nil, err
		}
		matched, err := retrieval.MatchDocument(codec, doc, cond)
		if gateErr := read.Check(ctx); gateErr != nil {
			return nil, readfailure.Join(gateErr, err)
		}
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
	for _, id := range docIDs {
		score := scores[id]
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
		doc.Score = item.score
		doc.ScoreState = retrieval.ScorePresent
		doc.ScoreSemantics = retrieval.ScoreSemantics(
			fmt.Sprintf("lexical.bm25:k1=%g,b=%g", idx.config.K1, idx.config.B),
		)
		doc.Rank = len(docs) + 1
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

// snapshotLocked captures only query postings and their candidate documents under
// one read lock. Global statistics remain corpus-wide; all copied maps are owned
// by this reader and writers cannot change its term frequencies or lengths.
func (idx *BM25Index[TMeta]) snapshotLocked(queryTokens []string) bm25Snapshot[TMeta] {
	docs := make(map[string]retrieval.Document[TMeta])
	lengths := make(map[string]int)
	postings := make(map[string]map[string]int, len(queryTokens))
	for _, term := range queryTokens {
		if _, exists := postings[term]; exists {
			continue
		}
		posting := idx.postings[term]
		if len(posting) == 0 {
			continue
		}
		copyPosting := make(map[string]int, len(posting))
		maps.Copy(copyPosting, posting)
		postings[term] = copyPosting
		for id := range posting {
			docs[id] = idx.docs[id]
			lengths[id] = idx.docLengths[id]
		}
	}
	return bm25Snapshot[TMeta]{docs: docs, docLengths: lengths, postings: postings,
		docCount: idx.docCount, avgLength: idx.avgLength}
}

// Schema returns the configured filter schema.
func (idx *BM25Index[TMeta]) Schema() filter.Schema {
	return idx.schema
}

// LexicalBackend marks BM25Index as a lexical backend.
func (idx *BM25Index[TMeta]) LexicalBackend() {}

var _ Backend[any] = (*BM25Index[any])(nil)

// ReadCapabilities declares pre-delivery scalar mandatory-filter enforcement.
// Managed pinned publication requires the lifecycle-aware adapter path.
func (idx *BM25Index[TMeta]) ReadCapabilities() access.Capabilities {
	return access.Capabilities{
		RequirePinnedPublication: idx.snapshotReadFingerprint != "",
		ScopeProfile:             true,
		PinnedPublication:        idx.snapshotReadFingerprint != "",
	}
}

// AdmitPublication permits partial pins only for an already captured readonly corpus.
// Retrieve independently requires the complete original binding fingerprint.
func (idx *BM25Index[TMeta]) AdmitPublication(_ access.Publication) error {
	if idx == nil || idx.snapshotReadFingerprint == "" {
		return access.UnsupportedCapability(ragy.ErrUnsupported)
	}
	return nil
}
