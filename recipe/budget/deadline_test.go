package budget_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/skosovsky/ragy/recipe/budget"
)

func TestLedgerContextUsesRemainingHostTimeAndParent(t *testing.T) {
	for _, parentShort := range []bool{false, true} {
		t.Run(map[bool]string{false: "ledger", true: "parent"}[parentShort], func(t *testing.T) {
			// Arrange: configured host epoch intentionally differs from wall time.
			hostNow := time.Date(2001, 1, 1, 0, 0, 0, 0, time.UTC)
			configured := hostNow.Add(40 * time.Millisecond)
			ledger, err := budget.New(budget.Config{Deadline: configured, Now: func() time.Time { return hostNow }})
			if err != nil {
				t.Fatal(err)
			}
			duration := time.Second
			if parentShort {
				duration = 5 * time.Millisecond
			}
			parent, cancel := context.WithTimeout(t.Context(), duration)
			defer cancel()
			// Act.
			child, stop := ledger.Context(parent)
			defer stop()
			deadline, ok := child.Deadline()
			// Assert.
			if !ok || ledger.Deadline() != configured || time.Until(deadline) > 50*time.Millisecond {
				t.Fatal(deadline, ledger.Deadline())
			}
			<-child.Done()
			if !errors.Is(child.Err(), context.DeadlineExceeded) {
				t.Fatal(child.Err())
			}
		})
	}
}
