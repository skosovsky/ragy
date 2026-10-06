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
	byKey := make(map[string]Document[TMeta], len(r.docs)+other.Len())
	for _, doc := range r.docs {
		if err := ValidateDocument(doc); err != nil {
			return resultSetFromByKey(byKey, r.resolver), ragy.WrapProjectionError(err, "merge validate")
		}
		key := r.resolver.Resolve(doc).MergeKey
		if err := validateMergeKey(key, doc.ID); err != nil {
			return resultSetFromByKey(byKey, r.resolver), err
		}
		for _, reference := range byKey {
			if err := validateComparable([]Document[TMeta]{reference, doc}); err != nil {
				return resultSetFromByKey(byKey, r.resolver), err
			}
			break
		}
		keepWinner(byKey, key, doc)
	}
	for _, doc := range other.Documents() {
		if err := ValidateDocument(doc); err != nil {
			return resultSetFromByKey(byKey, r.resolver), ragy.WrapProjectionError(err, "merge validate")
		}
		key := r.resolver.Resolve(doc).MergeKey
		if err := validateMergeKey(key, doc.ID); err != nil {
			return resultSetFromByKey(byKey, r.resolver), err
		}
		for _, reference := range byKey {
			if err := validateComparable([]Document[TMeta]{reference, doc}); err != nil {
				return resultSetFromByKey(byKey, r.resolver), err
			}
			break
		}
		keepWinner(byKey, key, doc)
	}

	return NewResultSet(sortedDocumentsFromByKey(byKey), r.resolver), nil
}

// Dedup removes duplicate MergeKey entries, keeping the highest score.
// Output is sorted by Score descending (stable). Returns ErrInvalidArgument on empty MergeKey.
func (r sliceResultSet[TMeta]) Dedup() (ResultSet[TMeta], error) {
	if r.IsEmpty() {
		return NewResultSet(nil, r.resolver), nil
	}

	byKey := make(map[string]Document[TMeta], len(r.docs))
	for _, doc := range r.docs {
		if err := ValidateDocument(doc); err != nil {
			return resultSetFromByKey(byKey, r.resolver), ragy.WrapProjectionError(err, "dedup validate")
		}
		key := r.resolver.Resolve(doc).MergeKey
		if err := validateMergeKey(key, doc.ID); err != nil {
			return resultSetFromByKey(byKey, r.resolver), err
		}
		for _, reference := range byKey {
			if err := validateComparable([]Document[TMeta]{reference, doc}); err != nil {
				return resultSetFromByKey(byKey, r.resolver), err
			}
			break
		}
		keepWinner(byKey, key, doc)
	}

	return NewResultSet(sortedDocumentsFromByKey(byKey), r.resolver), nil
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

func keepWinner[TMeta any](byKey map[string]Document[TMeta], key string, doc Document[TMeta]) {
	current, ok := byKey[key]
	if !ok {
		byKey[key] = doc
		return
	}
	if !samePayload(doc, current) {
		// A business merge key does not attest equality of source evidence.
		if rankedDocumentLess(doc, current) {
			byKey[key] = doc
		}
		return
	}
	if rankedDocumentLess(doc, current) {
		doc.SourceSupports = combineLocatorSupports(doc.SourceLocations(), current.SourceLocations())
		doc.ScoreHistory = append(append([]ScoreObservation(nil), doc.ScoreHistory...), current.ObservedScores()...)
		byKey[key] = doc
	} else {
		current.SourceSupports = combineLocatorSupports(current.SourceLocations(), doc.SourceLocations())
		current.ScoreHistory = append(append([]ScoreObservation(nil), current.ScoreHistory...), doc.ObservedScores()...)
		byKey[key] = current
	}
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
