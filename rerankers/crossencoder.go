// Package rerankers provides RRF and cross-encoder reranker implementations for ragy.
package rerankers

import (
	"context"
	"sort"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/skosovsky/ragy"
)

// ScoreFunc scores a (query, document) pair (e.g. cross-encoder model).
type ScoreFunc func(ctx context.Context, query string, doc ragy.Document) (float32, error)

// CrossEncoderReranker reranks by scoring each (query, doc) pair and taking topK.
type CrossEncoderReranker struct {
	Score ScoreFunc
}

// NewCrossEncoderReranker returns a new CrossEncoderReranker.
func NewCrossEncoderReranker(score ScoreFunc) *CrossEncoderReranker {
	return &CrossEncoderReranker{Score: score}
}

// Rerank implements ragy.Reranker.
func (r *CrossEncoderReranker) Rerank(ctx context.Context, query string, docs []ragy.Document, topK int) ([]ragy.Document, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	type scored struct {
		doc   ragy.Document
		score float32
	}
	scoredList := make([]scored, len(docs))
	g, gctx := errgroup.WithContext(ctx)
	var mu sync.Mutex
	for i := range docs {
		g.Go(func() error {
			select {
			case <-gctx.Done():
				return gctx.Err()
			default:
			}
			sc, err := r.Score(gctx, query, docs[i])
			if err != nil {
				return err
			}
			mu.Lock()
			scoredList[i] = scored{doc: docs[i], score: sc}
			mu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	sort.Slice(scoredList, func(i, j int) bool {
		return scoredList[i].score > scoredList[j].score
	})
	if topK <= 0 || topK > len(scoredList) {
		topK = len(scoredList)
	}
	out := make([]ragy.Document, topK)
	for i := 0; i < topK; i++ {
		out[i] = scoredList[i].doc
		out[i].Score = scoredList[i].score
	}
	return out, nil
}

var _ ragy.Reranker = (*CrossEncoderReranker)(nil)
