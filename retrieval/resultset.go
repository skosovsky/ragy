package retrieval

import (
	"fmt"
	"sort"

	ragy "github.com/skosovsky/ragy"
)

// ResultSet is an immutable ranked document batch with merge semantics.
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

// ResolverFor returns the identity resolver bound to rs.
func ResolverFor[TMeta any](rs ResultSet[TMeta]) IdentityResolver[TMeta] {
	if rs == nil {
		return DocumentIDResolver[TMeta]{}
	}
	if typed, ok := rs.(sliceResultSet[TMeta]); ok {
		if typed.resolver == nil {
			return DocumentIDResolver[TMeta]{}
		}
		return typed.resolver
	}
	return DocumentIDResolver[TMeta]{}
}

// NewResultSet constructs a ResultSet; nil docs yields an empty non-nil set.
func NewResultSet[TMeta any](docs []Document[TMeta], resolver IdentityResolver[TMeta]) ResultSet[TMeta] {
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	return sliceResultSet[TMeta]{
		docs:     copyDocuments(docs),
		resolver: resolver,
	}
}

// RewrapResultSet re-binds rs documents to resolver without changing merge semantics.
func RewrapResultSet[TMeta any](rs ResultSet[TMeta], resolver IdentityResolver[TMeta]) ResultSet[TMeta] {
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if rs == nil || rs.IsEmpty() {
		return NewResultSet[TMeta](nil, resolver)
	}
	return NewResultSet(rs.Documents(), resolver)
}

// Documents returns a defensive copy of ranked documents.
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
	if other == nil || other.IsEmpty() {
		return NewResultSet(r.docs, r.resolver), nil
	}
	if r.IsEmpty() {
		return NewResultSet(other.Documents(), r.resolver), nil
	}

	byKey := make(map[string]Document[TMeta], len(r.docs)+other.Len())
	for _, doc := range r.docs {
		if err := ValidateDocument(doc); err != nil {
			return resultSetFromByKey(byKey, r.resolver), ragy.WrapProjectionError(err, "merge validate")
		}
		key := r.resolver.Resolve(doc).MergeKey
		if err := validateMergeKey(key, doc.ID); err != nil {
			return resultSetFromByKey(byKey, r.resolver), err
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
		return out[i].Score > out[j].Score
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
	if !ok || doc.Score > current.Score {
		byKey[key] = doc
	}
}

func copyDocuments[TMeta any](docs []Document[TMeta]) []Document[TMeta] {
	if len(docs) == 0 {
		return nil
	}
	return append([]Document[TMeta](nil), docs...)
}
