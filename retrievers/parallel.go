package retrievers

import (
	"context"
	"sync"
)

// parallel runs n tasks concurrently. The first error cancels the shared context
// so other goroutines observe cancellation (similar to errgroup).
func parallel(ctx context.Context, n int, fn func(context.Context, int) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	var once sync.Once
	var retErr error

	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if err := fn(ctx, i); err != nil {
				once.Do(func() {
					retErr = err
					cancel()
				})
			}
		}(i)
	}
	wg.Wait()
	if retErr != nil {
		return retErr
	}
	return ctx.Err()
}
