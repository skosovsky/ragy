package parallel

import (
	"context"
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
	for result := range resultCh {
		if result.err != nil {
			var zero []R
			return zero, result.err
		}

		out[result.index] = result.value
	}

	if err := ctx.Err(); err != nil {
		return nil, err
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
				select {
				case <-ctx.Done():
					return
				case resultCh <- result[R]{index: task.index, value: value, err: err}:
				}
			}
		})
	}
}

func dispatchTasks[T any](ctx context.Context, taskCh chan<- task[T], items []T) {
	defer close(taskCh)

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
