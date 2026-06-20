package retrieval

import (
	"fmt"
	"sort"
	"strings"

	ragy "github.com/skosovsky/ragy"
)

// MergeStrategy combines grouped documents into one result.
type MergeStrategy[TMeta any] func([]Document[TMeta]) (Document[TMeta], error)

// DefaultMergeStrategy concatenates content, keeps the max score, and uses meta/id from the top-scoring chunk.
func DefaultMergeStrategy[TMeta any]() MergeStrategy[TMeta] {
	return func(docs []Document[TMeta]) (Document[TMeta], error) {
		if len(docs) == 0 {
			return Document[TMeta]{}, fmt.Errorf("%w: merge requires at least one document", ragy.ErrInvalidArgument)
		}
		best := docs[0]
		for _, doc := range docs[1:] {
			if rankedDocumentLess(doc, best) {
				best = doc
			}
		}

		parts := make([]string, 0, len(docs))
		for _, doc := range docs {
			if doc.Content != "" {
				parts = append(parts, doc.Content)
			}
		}

		merged := best
		merged.Content = strings.Join(parts, "\n\n")
		return merged, ragy.WrapProjectionError(ValidateDocument(merged), "merge strategy validate")
	}
}

type invalidPostProcessor[TMeta any] struct {
	err      error
	resolver IdentityResolver[TMeta]
}

func invalidPostProcessorFor[TMeta any](err error) PostProcessor[TMeta] {
	return invalidPostProcessor[TMeta]{
		err:      err,
		resolver: DocumentIDResolver[TMeta]{},
	}
}

func (p invalidPostProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	return preserveResultOnError(rs, p.err, p.resolver)
}

type groupByProcessor[TMeta any] struct {
	keySelector   func(TMeta) string
	mergeStrategy MergeStrategy[TMeta]
	resolver      IdentityResolver[TMeta]
}

// GroupBy groups documents by a business field from Meta and merges each group.
// For identity-based deduplication use ResultSet.Merge or ResultSet.Dedup with IdentityResolver.
func GroupBy[TMeta any](
	keySelector func(TMeta) string,
	mergeStrategy MergeStrategy[TMeta],
) PostProcessor[TMeta] {
	if keySelector == nil {
		return invalidPostProcessorFor[TMeta](
			fmt.Errorf("%w: group by key selector", ragy.ErrInvalidArgument),
		)
	}
	if mergeStrategy == nil {
		mergeStrategy = DefaultMergeStrategy[TMeta]()
	}
	return groupByProcessor[TMeta]{
		keySelector:   keySelector,
		mergeStrategy: mergeStrategy,
		resolver:      DocumentIDResolver[TMeta]{},
	}
}

func (p groupByProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	if rs == nil || rs.IsEmpty() {
		return NewResultSet[TMeta](nil, p.resolver), nil
	}
	if err := validateResultSet(rs); err != nil {
		return preserveResultOnError(rs, err, p.resolver)
	}
	docs := rs.Documents()

	groups := make(map[string][]Document[TMeta])
	for _, doc := range docs {
		key := p.keySelector(doc.Meta)
		if key == "" {
			return preserveResultOnError(rs, fmt.Errorf("%w: empty group key", ragy.ErrInvalidArgument), p.resolver)
		}
		groups[key] = append(groups[key], doc)
	}

	groupKeys := make([]string, 0, len(groups))
	for key := range groups {
		groupKeys = append(groupKeys, key)
	}
	sort.Strings(groupKeys)

	out := make([]Document[TMeta], 0, len(groupKeys))
	for _, key := range groupKeys {
		merged, err := p.mergeStrategy(groups[key])
		if err != nil {
			return preserveResultOnError(NewResultSet(out, p.resolver), err, p.resolver)
		}
		out = append(out, merged)
	}

	sort.SliceStable(out, func(i, j int) bool {
		return rankedDocumentLess(out[i], out[j])
	})
	return NewResultSet(out, p.resolver), nil
}

type topPerGroupProcessor[TMeta any] struct {
	keySelector func(TMeta) string
	limit       int
	resolver    IdentityResolver[TMeta]
}

// TopPerGroup keeps at most limit highest-scoring documents per business group from Meta.
// For identity-based deduplication use ResultSet.Merge or ResultSet.Dedup with IdentityResolver.
func TopPerGroup[TMeta any](keySelector func(TMeta) string, limit int) PostProcessor[TMeta] {
	if keySelector == nil {
		return invalidPostProcessorFor[TMeta](
			fmt.Errorf("%w: top per group key selector", ragy.ErrInvalidArgument),
		)
	}
	if limit <= 0 {
		return invalidPostProcessorFor[TMeta](
			fmt.Errorf("%w: top per group limit must be > 0", ragy.ErrInvalidArgument),
		)
	}
	return topPerGroupProcessor[TMeta]{
		keySelector: keySelector,
		limit:       limit,
		resolver:    DocumentIDResolver[TMeta]{},
	}
}

func (p topPerGroupProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	if rs == nil || rs.IsEmpty() {
		return NewResultSet[TMeta](nil, p.resolver), nil
	}
	if err := validateResultSet(rs); err != nil {
		return preserveResultOnError(rs, err, p.resolver)
	}
	docs := rs.Documents()

	groups := make(map[string][]Document[TMeta])
	for _, doc := range docs {
		key := p.keySelector(doc.Meta)
		if key == "" {
			return preserveResultOnError(rs, fmt.Errorf("%w: empty group key", ragy.ErrInvalidArgument), p.resolver)
		}
		groups[key] = append(groups[key], doc)
	}

	groupKeys := make([]string, 0, len(groups))
	for key := range groups {
		groupKeys = append(groupKeys, key)
	}
	sort.Strings(groupKeys)

	out := make([]Document[TMeta], 0, len(docs))
	for _, key := range groupKeys {
		group := groups[key]
		sort.SliceStable(group, func(i, j int) bool {
			return rankedDocumentLess(group[i], group[j])
		})
		if len(group) > p.limit {
			group = group[:p.limit]
		}
		out = append(out, group...)
	}

	sort.SliceStable(out, func(i, j int) bool {
		return rankedDocumentLess(out[i], out[j])
	})
	return NewResultSet(out, p.resolver), nil
}

type rerankProcessor[TMeta any] struct {
	less     func(a, b Document[TMeta]) bool
	resolver IdentityResolver[TMeta]
}

// Rerank sorts documents with a caller-provided ordering function.
func Rerank[TMeta any](less func(a, b Document[TMeta]) bool) PostProcessor[TMeta] {
	if less == nil {
		return invalidPostProcessorFor[TMeta](
			fmt.Errorf("%w: rerank less function", ragy.ErrInvalidArgument),
		)
	}
	return rerankProcessor[TMeta]{less: less, resolver: DocumentIDResolver[TMeta]{}}
}

func (p rerankProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	if rs == nil || rs.IsEmpty() {
		return NewResultSet[TMeta](nil, p.resolver), nil
	}
	if err := validateResultSet(rs); err != nil {
		return preserveResultOnError(rs, err, p.resolver)
	}
	docs := rs.Documents()
	sort.SliceStable(docs, func(i, j int) bool {
		return p.less(docs[i], docs[j])
	})
	return NewResultSet(docs, p.resolver), nil
}
