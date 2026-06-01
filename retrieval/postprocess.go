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
			if doc.Score > best.Score {
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
		return merged, ValidateDocument(merged)
	}
}

type groupByProcessor[TMeta any] struct {
	keySelector   func(TMeta) string
	mergeStrategy MergeStrategy[TMeta]
}

// GroupBy groups documents by a meta-derived key and merges each group.
func GroupBy[TMeta any](
	keySelector func(TMeta) string,
	mergeStrategy MergeStrategy[TMeta],
) PostProcessor[TMeta] {
	if keySelector == nil {
		panic("retrieval.GroupBy: keySelector must not be nil")
	}
	if mergeStrategy == nil {
		mergeStrategy = DefaultMergeStrategy[TMeta]()
	}
	return groupByProcessor[TMeta]{
		keySelector:   keySelector,
		mergeStrategy: mergeStrategy,
	}
}

func (p groupByProcessor[TMeta]) Process(docs []Document[TMeta]) ([]Document[TMeta], error) {
	if len(docs) == 0 {
		return nil, nil
	}

	groups := make(map[string][]Document[TMeta])
	order := make([]string, 0)
	for _, doc := range docs {
		key := p.keySelector(doc.Meta)
		if _, ok := groups[key]; !ok {
			order = append(order, key)
		}
		groups[key] = append(groups[key], doc)
	}

	out := make([]Document[TMeta], 0, len(order))
	for _, key := range order {
		merged, err := p.mergeStrategy(groups[key])
		if err != nil {
			return nil, err
		}
		out = append(out, merged)
	}

	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Score > out[j].Score
	})
	return out, nil
}

type topPerGroupProcessor[TMeta any] struct {
	keySelector func(TMeta) string
	limit       int
}

// TopPerGroup keeps at most limit highest-scoring documents per group key.
func TopPerGroup[TMeta any](keySelector func(TMeta) string, limit int) PostProcessor[TMeta] {
	if keySelector == nil {
		panic("retrieval.TopPerGroup: keySelector must not be nil")
	}
	if limit <= 0 {
		panic("retrieval.TopPerGroup: limit must be > 0")
	}
	return topPerGroupProcessor[TMeta]{keySelector: keySelector, limit: limit}
}

func (p topPerGroupProcessor[TMeta]) Process(docs []Document[TMeta]) ([]Document[TMeta], error) {
	if len(docs) == 0 {
		return nil, nil
	}

	groups := make(map[string][]Document[TMeta])
	for _, doc := range docs {
		key := p.keySelector(doc.Meta)
		groups[key] = append(groups[key], doc)
	}

	out := make([]Document[TMeta], 0, len(docs))
	for _, group := range groups {
		sort.SliceStable(group, func(i, j int) bool {
			return group[i].Score > group[j].Score
		})
		if len(group) > p.limit {
			group = group[:p.limit]
		}
		out = append(out, group...)
	}

	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Score > out[j].Score
	})
	return out, nil
}

type rerankProcessor[TMeta any] struct {
	less func(a, b Document[TMeta]) bool
}

// Rerank sorts documents with a caller-provided ordering function.
func Rerank[TMeta any](less func(a, b Document[TMeta]) bool) PostProcessor[TMeta] {
	if less == nil {
		panic("retrieval.Rerank: less must not be nil")
	}
	return rerankProcessor[TMeta]{less: less}
}

func (p rerankProcessor[TMeta]) Process(docs []Document[TMeta]) ([]Document[TMeta], error) {
	if len(docs) == 0 {
		return nil, nil
	}
	sort.SliceStable(docs, func(i, j int) bool {
		return p.less(docs[i], docs[j])
	})
	return docs, nil
}
