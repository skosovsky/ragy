package managed

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestInventoryObservationRetainsActualMutationFence(t *testing.T) {
	// Arrange: empty retained inventory has no host callbacks or record projections.
	adapter := &Adapter[struct{}]{
		config:   Config[struct{}]{Namespace: "n", Target: "lexical"},
		versions: make(map[revisionKey]staged[struct{}]),
	}
	observer, err := adapter.InventoryObserver(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	input := lifecycle.Inventory{
		Namespace: "n",
		Kind:      lifecycle.CompleteInventory,
		Watermark: "empty",
		Coverage:  lifecycle.FullInventory,
		Targets:   []string{"lexical"},
	}
	// Act: the actual Stage/Cleanup/host-basis mutex remains fenced in the callback.
	calls := 0
	err = observer.ObserveInventory(t.Context(), input, func() error {
		calls++
		if adapter.mu.TryLock() {
			adapter.mu.Unlock()
			t.Fatal("mutation fence escaped")
		}
		return nil
	})
	// Assert: callback once, released fence afterwards.
	if err != nil || calls != 1 {
		t.Fatal("fenced observation", err, calls)
	}
	if !adapter.mu.TryLock() {
		t.Fatal("fence leaked")
	}
	// An existing writer produces conflict without a wait or callback.
	err = observer.ObserveInventory(
		t.Context(),
		input,
		func() error { t.Fatal("callback despite conflict"); return nil },
	)
	adapter.mu.Unlock()
	if !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("writer conflict", err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	err = observer.ObserveInventory(canceled, input, func() error { cancel(); return nil })
	if !errors.Is(err, context.Canceled) {
		t.Fatal("cancellation in callback acknowledged", err)
	}
	if _, err = adapter.InventoryObserver(0, 1); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("unbounded observer", err)
	}
}
