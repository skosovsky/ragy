package parallel

import (
	"context"
	"errors"
	"fmt"
	"sync"

	ragy "github.com/skosovsky/ragy"
)

type task[T any] struct {
	index int
	item  T
}
type result[R any] struct {
	index int
	value R
	err   error
}

// MapOrdered preserves order with bounded structured concurrency. Callback errors
// cancel new dispatch and siblings; all started cooperative callbacks are joined.
// Callbacks must honor context and own mutable state. Arbitrary uncooperative host
// functions cannot be terminated. Ordinary failures return no partial result slice.
func MapOrdered[T any, R any](
	ctx context.Context,
	concurrency int,
	items []T,
	fn func(context.Context, T) (R, error),
) ([]R, error) {
	if concurrency <= 0 || fn == nil {
		return nil, fmt.Errorf("%w: parallel map concurrency/callback", ragy.ErrInvalidArgument)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, nil
	}
	child, cancel := context.WithCancel(ctx)
	defer cancel()
	tasks := make(chan task[T])
	results := make(chan result[R], len(items))
	var workers sync.WaitGroup
	startMapWorkers(child, cancel, &workers, min(concurrency, len(items)), tasks, results, fn)
	workers.Go(func() { dispatchMapTasks(child, tasks, items) })
	go func() { workers.Wait(); close(results) }()
	return collectMapResults(ctx, results, len(items))
}

func startMapWorkers[T any, R any](
	ctx context.Context,
	cancel context.CancelFunc,
	wg *sync.WaitGroup,
	concurrency int,
	tasks <-chan task[T],
	results chan<- result[R],
	fn func(context.Context, T) (R, error),
) {
	for range concurrency {
		wg.Go(func() {
			for task := range tasks {
				if ctx.Err() != nil {
					break
				}
				value, err := fn(ctx, task.item)
				if err == nil {
					err = ctx.Err()
				}
				if err != nil {
					cancel()
				}
				results <- result[R]{index: task.index, value: value, err: err}
			}
		})
	}
}
func dispatchMapTasks[T any](ctx context.Context, tasks chan<- task[T], items []T) {
	defer close(tasks)
	for index, item := range items {
		if ctx.Err() != nil {
			return
		}
		select {
		case <-ctx.Done():
			return
		case tasks <- task[T]{index: index, item: item}:
		}
	}
}
func collectMapResults[R any](ctx context.Context, results <-chan result[R], size int) ([]R, error) {
	out := make([]R, size)
	var firstErr, errorFatal error
	for item := range results {
		out[item.index] = item.value
		if item.err != nil {
			if firstErr == nil {
				firstErr = item.err
			}
			if errorFatal == nil && !errors.Is(item.err, context.Canceled) &&
				!errors.Is(item.err, context.DeadlineExceeded) {
				errorFatal = item.err
			}
		}
	}
	if errorFatal != nil {
		return nil, errorFatal
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if firstErr != nil {
		return out, firstErr
	}
	return out, nil
}
