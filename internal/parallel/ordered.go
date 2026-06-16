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

// MapOrdered applies fn with bounded parallelism and preserves item order.
func MapOrdered[T any, R any](
	ctx context.Context,
	concurrency int,
	items []T,
	fn func(context.Context, T) (R, error),
) ([]R, error) {
	if concurrency <= 0 {
		return nil, fmt.Errorf("%w: concurrency must be > 0", ragy.ErrInvalidArgument)
	}

	if len(items) == 0 {
		return nil, nil
	}
	taskCh := make(chan task[T])
	resultCh := make(chan result[R], len(items))

	var wg sync.WaitGroup
	startWorkers(ctx, &wg, concurrency, taskCh, resultCh, fn)
	go dispatchTasks(ctx, taskCh, items)
	go closeResultsOnWait(&wg, resultCh)

	out := make([]R, len(items))
	got := make([]bool, len(items))
	firstErr, fatalErr := collectOrderedResults(resultCh, out, got)
	if fatalErr != nil {
		return nil, fatalErr
	}
	return finalizeMapOrdered(ctx, out, got, firstErr)
}

func collectOrderedResults[R any](resultCh <-chan result[R], out []R, got []bool) (error, error) {
	var firstErr error
	for result := range resultCh {
		if result.err != nil {
			if !errors.Is(result.err, context.Canceled) && !errors.Is(result.err, context.DeadlineExceeded) {
				return firstErr, result.err
			}
			if firstErr == nil {
				firstErr = result.err
			}
		}
		out[result.index] = result.value
		got[result.index] = true
	}
	return firstErr, nil
}

func finalizeMapOrdered[R any](ctx context.Context, out []R, got []bool, firstErr error) ([]R, error) {
	for i := range got {
		if got[i] {
			continue
		}
		if err := ctx.Err(); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			return out, firstErr
		}
		return nil, fmt.Errorf("%w: parallel map missing result at index %d", ragy.ErrProtocol, i)
	}
	if firstErr != nil {
		return out, firstErr
	}
	return out, nil
}

func startWorkers[T any, R any](
	ctx context.Context,
	wg *sync.WaitGroup,
	concurrency int,
	taskCh <-chan task[T],
	resultCh chan<- result[R],
	fn func(context.Context, T) (R, error),
) {
	for range concurrency {
		wg.Go(func() {
			for task := range taskCh {
				value, err := fn(ctx, task.item)
				if ctxErr := ctx.Err(); ctxErr != nil {
					if err != nil {
						err = fmt.Errorf("%w: %w", ctxErr, err)
					} else {
						err = ctxErr
					}
				}
				resultCh <- result[R]{index: task.index, value: value, err: err}
			}
		})
	}
}

func dispatchTasks[T any](ctx context.Context, taskCh chan<- task[T], items []T) {
	defer close(taskCh)

	// Early return on cancel may leave unprocessed items; MapOrdered detects missing slots.
	for index, item := range items {
		select {
		case <-ctx.Done():
			return
		case taskCh <- task[T]{index: index, item: item}:
		}
	}
}

func closeResultsOnWait[R any](wg *sync.WaitGroup, resultCh chan result[R]) {
	wg.Wait()
	close(resultCh)
}
