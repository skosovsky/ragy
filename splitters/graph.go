// Package splitters provides Splitter implementations (recursive, markdown, semantic, GraphExtractor, ContextualSplitter).
package splitters

import (
	"context"
	"iter"
	"reflect"
	"sync"

	"github.com/skosovsky/ragy"
)

const defaultGraphConcurrency = 5

// GraphExtractor is a middleware Splitter that runs an inner Splitter, then for each chunk
// optionally calls ChunkGraphProvider with the full chunk (e.g. metadata prepared upstream) and upserts into GraphStore.
// Core does not extract entities from raw text; LLM-centric extraction belongs in application code.
// Uses a worker pool and a slice of channels (Future/Promise) to preserve chunk order when yielding.
type GraphExtractor struct {
	Inner       Splitter
	Graph       ragy.GraphStore         // required when Provider is non-nil; otherwise Split returns ragy.ErrMissingGraphStore
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
func NewGraphExtractor(
	inner Splitter,
	graph ragy.GraphStore,
	provider ragy.ChunkGraphProvider,
	opts ...GraphExtractorOption,
) *GraphExtractor {
	ge := &GraphExtractor{
		Inner:       inner,
		Graph:       graph,
		Provider:    provider,
		Concurrency: defaultGraphConcurrency,
	}
	for _, o := range opts {
		o(ge)
	}
	return ge
}

// Split implements Splitter. Order of yielded chunks is preserved via per-index result channels.
//
//nolint:gocognit,funlen // Worker pool + per-index channels; long but linear coordination.
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

		if isNilGraphStore(g.Graph) {
			_ = yield(ragy.Document{}, ragy.ErrMissingGraphStore)
			return
		}

		runCtx, cancel := context.WithCancel(ctx)
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
		for range g.Concurrency {
			wg.Go(func() {
				for {
					select {
					case <-runCtx.Done():
						return
					case i, ok := <-jobs:
						if !ok {
							return
						}
						chunk := chunks[i]
						nodes, edges, err := g.Provider(runCtx, chunk)
						if err != nil {
							results[i] <- struct {
								doc ragy.Document
								err error
							}{ragy.Document{}, err}
							continue
						}
						if len(nodes) > 0 || len(edges) > 0 {
							if err := g.Graph.UpsertGraph(runCtx, nodes, edges); err != nil {
								results[i] <- struct {
									doc ragy.Document
									err error
								}{ragy.Document{}, err}
								continue
							}
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
				case <-runCtx.Done():
					return
				case jobs <- i:
				}
			}
			close(jobs)
		}()

		// Read results strictly in order.
		for i := range n {
			select {
			case <-runCtx.Done():
				wg.Wait()
				_ = yield(ragy.Document{}, runCtx.Err())
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

// isNilGraphStore reports whether g is nil or a typed nil (e.g. (*Store)(nil) assigned to GraphStore).
func isNilGraphStore(g ragy.GraphStore) bool {
	if g == nil {
		return true
	}
	v := reflect.ValueOf(g)
	//nolint:exhaustive // only Pointer/Interface/Map/Func/Slice/Chan can be nil; other kinds are never nil here.
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Func, reflect.Slice, reflect.Chan:
		return v.IsNil()
	default:
		return false
	}
}
