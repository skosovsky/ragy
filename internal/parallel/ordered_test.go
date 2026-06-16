package parallel

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestMapOrderedPreservesOrder(t *testing.T) {
	t.Parallel()

	items := []int{0, 1, 2, 3}
	out, err := MapOrdered(context.Background(), 2, items, func(_ context.Context, item int) (int, error) {
		return item * 10, nil
	})
	if err != nil {
		t.Fatalf("MapOrdered() error = %v", err)
	}
	want := []int{0, 10, 20, 30}
	for i, v := range want {
		if out[i] != v {
			t.Fatalf("out[%d] = %d, want %d", i, out[i], v)
		}
	}
}

func TestMapOrderedReturnsFnError(t *testing.T) {
	t.Parallel()

	items := []int{1, 2, 3}
	out, err := MapOrdered(context.Background(), 2, items, func(_ context.Context, item int) (int, error) {
		if item == 2 {
			return 0, errors.New("boom")
		}
		return item, nil
	})
	if err == nil {
		t.Fatal("MapOrdered() error = nil, want error")
	}
	if out != nil {
		t.Fatalf("MapOrdered() out = %#v, want nil slice", out)
	}
}

func TestMapOrderedDeliversAllResultsOnContextCancel(t *testing.T) {
	t.Parallel()

	blocked := make(chan struct{})
	var notifyOnce sync.Once
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-blocked
		cancel()
	}()

	items := []int{0, 1, 2, 3}
	out, err := MapOrdered(ctx, 4, items, func(ctx context.Context, item int) (int, error) {
		if item == 2 {
			notifyOnce.Do(func() { close(blocked) })
			<-ctx.Done()
			return 0, ctx.Err()
		}
		return item, nil
	})
	if err == nil {
		t.Fatal("MapOrdered() error = nil, want cancel or fn error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("MapOrdered() error = %v, want context canceled", err)
	}
	if len(out) != 4 {
		t.Fatalf("len(out) = %d, want 4 (no silent zero slots)", len(out))
	}
}

func TestMapOrderedPropagatesContextErrorAfterPartialResults(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())

	items := []int{0, 1}
	done := make(chan struct{})
	go func() {
		_, _ = MapOrdered(ctx, 2, items, func(c context.Context, item int) (int, error) {
			if item == 1 {
				<-c.Done()
				return 0, c.Err()
			}
			return item, nil
		})
		close(done)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done
}
