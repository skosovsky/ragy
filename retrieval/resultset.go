package retrieval

import (
	"fmt"
	"sort"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/nilvalue"
)

// ResultSet owns ranked documents and ragy-defined slices with merge semantics.
// Arbitrary BYOT metadata is host-owned; consumers must clone mutable metadata
// before sharing or concurrently mutating it.
type ResultSet[TMeta any] interface {
	Documents() []Document[TMeta]
	Merge(other ResultSet[TMeta]) (ResultSet[TMeta], error)
	Dedup() (ResultSet[TMeta], error)
	IsEmpty() bool
	Len() int
}

type sliceResultSet[TMeta any] struct {
	docs     []Document[TMeta]
	resolver IdentityResolver[TMeta]
}

// ResolverProvider is an optional capability for custom BYOT ResultSets.
type ResolverProvider[TMeta any] interface {
	IdentityResolver() IdentityResolver[TMeta]
}

// ResolverFor uses optional resolver capability or defaults to document identity.
func ResolverFor[TMeta any](rs ResultSet[TMeta]) IdentityResolver[TMeta] {
	if nilvalue.IsNil(rs) {
		return DocumentIDResolver[TMeta]{}
	}
	if provider, ok := rs.(ResolverProvider[TMeta]); ok {
		return DefaultResolver(provider.IdentityResolver())
	}
	return DocumentIDResolver[TMeta]{}
}

func (r sliceResultSet[TMeta]) IdentityResolver() IdentityResolver[TMeta] {
	return DefaultResolver(r.resolver)
}

// NewResultSet constructs a ResultSet; nil docs yields an empty non-nil set.
func NewResultSet[TMeta any](docs []Document[TMeta], resolver IdentityResolver[TMeta]) ResultSet[TMeta] {
	if nilvalue.IsNil(resolver) {
		resolver = DocumentIDResolver[TMeta]{}
	}
	return sliceResultSet[TMeta]{
		docs:     copyDocuments(docs),
		resolver: resolver,
	}
}

// RewrapResultSet re-binds rs documents to resolver without changing merge semantics.
func RewrapResultSet[TMeta any](rs ResultSet[TMeta], resolver IdentityResolver[TMeta]) ResultSet[TMeta] {
	if nilvalue.IsNil(resolver) {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if nilvalue.IsNil(rs) || rs.IsEmpty() {
		return NewResultSet[TMeta](nil, resolver)
	}
	return NewResultSet(rs.Documents(), resolver)
}

// Documents copies ranked documents and ragy-owned slices. BYOT metadata keeps
// its host-defined ownership; this method does not deep-clone domain values.
func (r sliceResultSet[TMeta]) Documents() []Document[TMeta] {
	return copyDocuments(r.docs)
}

// Len returns the number of documents.
func (r sliceResultSet[TMeta]) Len() int {
	return len(r.docs)
}

// IsEmpty reports whether the set has no documents.
func (r sliceResultSet[TMeta]) IsEmpty() bool {
	return len(r.docs) == 0
}

// Merge combines documents by MergeKey, keeping the highest score per key.
// When scores tie, the first seen document wins (not newest-by-timestamp).
// Returns ErrInvalidArgument when a custom resolver returns an empty MergeKey.
func (r sliceResultSet[TMeta]) Merge(other ResultSet[TMeta]) (ResultSet[TMeta], error) {
	return r.aggregate("merge validate", len(r.docs)+other.Len(), other.Documents)
}

// Dedup removes duplicate MergeKey entries, keeping the highest score.
// Output is sorted by Score descending (stable). Returns ErrInvalidArgument on empty MergeKey.
func (r sliceResultSet[TMeta]) Dedup() (ResultSet[TMeta], error) {
	return r.aggregate("dedup validate", len(r.docs), nil)
}

func (r sliceResultSet[TMeta]) aggregate(
	operation string,
	capacity int,
	additional func() []Document[TMeta],
) (ResultSet[TMeta], error) {
	byKey := make(map[string]Document[TMeta], capacity)
	var accumulated map[string]*winnerEvidence
	var reference *Document[TMeta]
	finish := func() ResultSet[TMeta] {
		for key, evidence := range accumulated {
			doc := byKey[key]
			doc.SourceSupports = evidence.locations()
			doc.ScoreHistory = evidence.history.snapshot()
			byKey[key] = doc
		}
		return resultSetFromByKey(byKey, r.resolver)
	}
	docs := r.docs
	for {
		for i := range docs {
			doc := docs[i]
			if err := ValidateDocument(doc); err != nil {
				return finish(), ragy.WrapProjectionError(err, operation)
			}
			key := r.resolver.Resolve(doc).MergeKey
			if err := validateMergeKey(key, doc.ID); err != nil {
				return finish(), err
			}
			// Each original input is validated once. The reference scale cannot change
			// during aggregation, so accumulated evidence needs no repeated validation.
			if reference != nil &&
				!sameScoreScale(*reference, doc) {
				return finish(), fmt.Errorf(
					"%w: incompatible score scales; use explicit rank fusion or normalization",
					ragy.ErrInvalidArgument,
				)
			}
			if reference == nil {
				reference = &docs[i]
			}
			accumulated = accumulateWinner(byKey, accumulated, key, doc)
		}
		if additional == nil {
			break
		}
		docs = additional()
		additional = nil
	}
	return finish(), nil
}

func resultSetFromByKey[TMeta any](
	byKey map[string]Document[TMeta],
	resolver IdentityResolver[TMeta],
) ResultSet[TMeta] {
	return NewResultSet(sortedDocumentsFromByKey(byKey), resolver)
}

func sortedDocumentsFromByKey[TMeta any](byKey map[string]Document[TMeta]) []Document[TMeta] {
	if len(byKey) == 0 {
		return nil
	}
	keys := make([]string, 0, len(byKey))
	for key := range byKey {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	out := make([]Document[TMeta], 0, len(keys))
	for _, key := range keys {
		out = append(out, byKey[key])
	}
	sort.SliceStable(out, func(i, j int) bool {
		return rankedDocumentLess(out[i], out[j])
	})
	return out
}

func validateMergeKey(key, docID string) error {
	if key == "" {
		return fmt.Errorf("%w: empty merge key for document %q", ragy.ErrInvalidArgument, docID)
	}
	return nil
}

func rankedDocumentLess[TMeta any](left, right Document[TMeta]) bool {
	leftScored := left.ScoreState.IsScored()
	rightScored := right.ScoreState.IsScored()
	switch {
	case leftScored && rightScored:
		return left.Score > right.Score
	case leftScored:
		return true
	case rightScored:
		return false
	case left.Rank > 0 && right.Rank > 0:
		return left.Rank < right.Rank
	case left.Rank > 0:
		return true
	case right.Rank > 0:
		return false
	default:
		return false
	}
}

func copyDocuments[TMeta any](docs []Document[TMeta]) []Document[TMeta] {
	if len(docs) == 0 {
		return nil
	}
	out := append([]Document[TMeta](nil), docs...)
	for i := range out {
		out[i].SourceSupports = append(out[i].SourceSupports[:0:0], out[i].SourceSupports...)
		out[i].ScoreHistory = append([]ScoreObservation(nil), out[i].ScoreHistory...)
	}
	return out
}

// accumulateWinner preserves the eager winner's evidence order using immutable segments.
func accumulateWinner[TMeta any](
	byKey map[string]Document[TMeta],
	accumulated map[string]*winnerEvidence,
	key string,
	doc Document[TMeta],
) map[string]*winnerEvidence {
	current, exists := byKey[key]
	if !exists {
		byKey[key] = doc
		return accumulated
	}
	if !samePayload(doc, current) {
		if rankedDocumentLess(doc, current) {
			byKey[key] = doc
			delete(accumulated, key)
		}
		return accumulated
	}
	if accumulated == nil {
		accumulated = make(map[string]*winnerEvidence)
	}
	evidence := accumulated[key]
	if evidence == nil {
		evidence = new(winnerEvidence)
		evidence.supports.add(current.SourceLocations(), false)
		evidence.history.add(current.ScoreHistory, false)
		accumulated[key] = evidence
	}
	if rankedDocumentLess(doc, current) {
		evidence.supports.add(doc.SourceLocations(), true)
		evidence.history.add(doc.ScoreHistory, true)
		observeWinner(evidence, current)
		byKey[key] = doc
	} else {
		evidence.supports.add(doc.SourceLocations(), false)
		evidence.history.add(doc.ScoreHistory, false)
		observeWinner(evidence, doc)
	}
	return accumulated
}

// sameScoreScale compares already-validated inputs without rescanning evidence.
func sameScoreScale[TMeta any](left, right Document[TMeta]) bool {
	return left.ScoreState == right.ScoreState && left.ScoreSemantics == right.ScoreSemantics
}
