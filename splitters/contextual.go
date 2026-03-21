package splitters

import (
	"context"
	"iter"
	"strings"
	"sync"

	"github.com/skosovsky/ragy"
)

const defaultContextualConcurrency = 5

// ContextualSplitter is a middleware Splitter that runs an inner Splitter, then for each chunk
// calls Contextualizer to generate enriching context and prepends it to the chunk content.
// Uses a worker pool and per-index result channels to preserve chunk order when yielding.
// If yield returns false or context is cancelled, cancel() is called before wg.Wait() to stop in-flight LLM calls.
type ContextualSplitter struct {
	Inner          Splitter
	Contextualizer ragy.Contextualizer
	Concurrency    int
}

// ContextualSplitterOption configures ContextualSplitter.
type ContextualSplitterOption func(*ContextualSplitter)

// WithContextualConcurrency sets the number of worker goroutines for context generation.
func WithContextualConcurrency(n int) ContextualSplitterOption {
	return func(c *ContextualSplitter) {
		c.Concurrency = n
	}
}

// NewContextualSplitter returns a ContextualSplitter that wraps inner and enriches chunks via contextualizer.
func NewContextualSplitter(
	inner Splitter,
	contextualizer ragy.Contextualizer,
	opts ...ContextualSplitterOption,
) *ContextualSplitter {
	cs := &ContextualSplitter{
		Inner:          inner,
		Contextualizer: contextualizer,
		Concurrency:    defaultContextualConcurrency,
	}
	for _, o := range opts {
		o(cs)
	}
	return cs
}

type contextualResult struct {
	doc ragy.Document
	err error
}

// Split implements Splitter. Order of yielded chunks is preserved via per-index result channels.
//
//nolint:gocognit // Worker pool + ordered channels; splitting would fragment the coordination logic.
func (c *ContextualSplitter) Split(ctx context.Context, doc ragy.Document) iter.Seq2[ragy.Document, error] {
	return func(yield func(ragy.Document, error) bool) {
		var chunks []ragy.Document
		for chunk, err := range c.Inner.Split(ctx, doc) {
			if err != nil {
				_ = yield(ragy.Document{}, err)
				return
			}
			chunks = append(chunks, chunk)
		}
		if len(chunks) == 0 {
			return
		}

		runCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		n := len(chunks)
		concurrency := c.Concurrency
		if concurrency <= 0 {
			concurrency = 1
		}
		results := make([]chan contextualResult, n)
		for i := range results {
			results[i] = make(chan contextualResult, 1)
		}

		jobs := make(chan int, n)
		var wg sync.WaitGroup
		for range concurrency {
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
						contextText, err := c.Contextualizer.GenerateContext(runCtx, doc.Content, chunk.Content)
						if err != nil {
							results[i] <- contextualResult{doc: ragy.Document{}, err: err}
							continue
						}
						contextText = strings.TrimSpace(contextText)
						if contextText != "" {
							chunk.Content = contextText + "\n\n" + chunk.Content
						}
						results[i] <- contextualResult{doc: chunk, err: nil}
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

		for i := range n {
			select {
			case <-runCtx.Done():
				cancel()
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
