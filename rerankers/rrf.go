package rerankers

import (
	"context"
	"sort"

	"github.com/skosovsky/ragy"
)

// RankedListsMerger merges multiple ranked document lists (e.g. RRF).
// Used by EnsembleRetriever to merge results from several retrievers.
type RankedListsMerger interface {
	ragy.Reranker
	MergeRankedLists(ctx context.Context, lists [][]ragy.Document, topK int) []ragy.Document
}

const defaultRRFK = 60

// RRFReranker merges multiple ranked lists using Reciprocal Rank Fusion.
// score(d) = sum over lists of 1/(K + rank_i(d)).
type RRFReranker struct {
	K int
}

// RRFOption configures RRFReranker.
type RRFOption func(*RRFReranker)

// WithRRFK sets the RRF constant K (default 60).
func WithRRFK(k int) RRFOption {
	return func(r *RRFReranker) {
		r.K = k
	}
}

// NewRRFReranker returns a new RRFReranker.
func NewRRFReranker(opts ...RRFOption) *RRFReranker {
	r := &RRFReranker{K: defaultRRFK}
	for _, o := range opts {
		o(r)
	}
	return r
}

// Rerank implements ragy.Reranker. For a single list, returns docs up to topK (order preserved).
// For eval pipelines that pass pre-merged lists, use MergeRankedLists instead.
func (r *RRFReranker) Rerank(_ context.Context, _ string, docs []ragy.Document, topK int) ([]ragy.Document, error) {
	if len(docs) <= topK || topK <= 0 {
		return docs, nil
	}
	return docs[:topK], nil
}

// MergeRankedLists merges multiple ranked lists with RRF and returns topK documents.
// Used by EnsembleRetriever.
func (r *RRFReranker) MergeRankedLists(_ context.Context, lists [][]ragy.Document, topK int) []ragy.Document {
	k := r.K
	if k <= 0 {
		k = defaultRRFK
	}
	// Compute RRF score per document ID.
	scores := make(map[string]float32)
	docByID := make(map[string]ragy.Document)
	for _, list := range lists {
		for rank, d := range list {
			id := d.ID
			if id == "" {
				id = d.Content
			}
			scores[id] += 1.0 / float32(k+rank+1)
			docByID[id] = d
		}
	}
	type scored struct {
		id    string
		score float32
	}
	var sorted []scored
	for id, sc := range scores {
		sorted = append(sorted, scored{id, sc})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].score > sorted[j].score
	})
	if topK <= 0 {
		topK = len(sorted)
	}
	out := make([]ragy.Document, 0, topK)
	for i := 0; i < topK && i < len(sorted); i++ {
		d := docByID[sorted[i].id]
		d.Score = sorted[i].score
		out = append(out, d)
	}
	return out
}

var _ ragy.Reranker = (*RRFReranker)(nil)
