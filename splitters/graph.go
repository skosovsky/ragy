// Package splitters provides Splitter implementations (recursive, markdown, semantic, GraphExtractor, ContextualSplitter).
package splitters

import (
	"context"
	"iter"
	"sync"

	"github.com/skosovsky/ragy"
)

// GraphExtractor is a middleware Splitter that runs an inner Splitter, then for each chunk
// optionally calls ChunkGraphProvider with the full chunk (e.g. metadata prepared upstream) and upserts into GraphStore.
// Core does not extract entities from raw text; LLM-centric extraction belongs in application code.
// Uses a worker pool and a slice of channels (Future/Promise) to preserve chunk order when yielding.
type GraphExtractor struct {
	Inner       Splitter
	Graph       ragy.GraphStore
	Provider    ragy.ChunkGraphProvider // optional; if nil, only yields inner chunks without graph writes
	Concurrency int
}

// GraphExtractorOption configures GraphExtractor.
type GraphExtractorOption func(*GraphExtractor)

// WithConcurrency sets the number of worker goroutines for chunk graph preparation.
func WithConcurrency(n int) GraphExtractorOption {
	return func(g *GraphExtractor) {
		g.Concurrency = n
	}
}

// NewGraphExtractor returns a GraphExtractor that wraps inner and writes to graph when Provider is set.
func NewGraphExtractor(inner Splitter, graph ragy.GraphStore, provider ragy.ChunkGraphProvider, opts ...GraphExtractorOption) *GraphExtractor {
	ge := &GraphExtractor{
		Inner:       inner,
		Graph:       graph,
		Provider:    provider,
		Concurrency: 5,
	}
	for _, o := range opts {
		o(ge)
	}
	return ge
}

// Split implements Splitter. Order of yielded chunks is preserved via per-index result channels.
func (g *GraphExtractor) Split(ctx context.Context, doc ragy.Document) iter.Seq2[ragy.Document, error] {
	return func(yield func(ragy.Document, error) bool) {
		// Collect all chunks from inner splitter so we have a fixed order and count.
		var chunks []ragy.Document
		for chunk, err := range g.Inner.Split(ctx, doc) {
			if err != nil {
				_ = yield(ragy.Document{}, err)
				return
			}
			chunks = append(chunks, chunk)
		}
		if len(chunks) == 0 {
			return
		}

		if g.Provider == nil {
			for _, chunk := range chunks {
				if !yield(chunk, nil) {
					return
				}
			}
			return
		}

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		n := len(chunks)
		if g.Concurrency <= 0 {
			g.Concurrency = 1
		}
		// One channel per chunk index (Future/Promise pattern).
		results := make([]chan struct {
			doc ragy.Document
			err error
		}, n)
		for i := range results {
			results[i] = make(chan struct {
				doc ragy.Document
				err error
			}, 1)
		}

		jobs := make(chan int, n)
		var wg sync.WaitGroup
		for w := 0; w < g.Concurrency; w++ {
			wg.Go(func() {
				for {
					select {
					case <-ctx.Done():
						return
					case i, ok := <-jobs:
						if !ok {
							return
						}
						chunk := chunks[i]
						nodes, edges, err := g.Provider(ctx, chunk)
						if err != nil {
							results[i] <- struct {
								doc ragy.Document
								err error
							}{ragy.Document{}, err}
							continue
						}
						if len(nodes) > 0 || len(edges) > 0 {
							_ = g.Graph.UpsertGraph(ctx, nodes, edges)
						}
						results[i] <- struct {
							doc ragy.Document
							err error
						}{chunk, nil}
					}
				}
			})
		}

		go func() {
			for i := range n {
				select {
				case <-ctx.Done():
					return
				case jobs <- i:
				}
			}
			close(jobs)
		}()

		// Read results strictly in order.
		for i := range n {
			select {
			case <-ctx.Done():
				wg.Wait()
				_ = yield(ragy.Document{}, ctx.Err())
				return
			case res := <-results[i]:
				if res.err != nil {
					cancel()
					wg.Wait()
					_ = yield(ragy.Document{}, res.err)
					return
				}
				if !yield(res.doc, nil) {
					cancel()
					wg.Wait()
					return
				}
			}
		}
		wg.Wait()
	}
}
