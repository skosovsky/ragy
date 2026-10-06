package budget_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/recipe/budget"
)

func TestCheckUsesOwnClockWithoutSpendingOrRefundingCapacity(t *testing.T) {
	// Arrange: capacity is fully occupied, but the attempt deadline is still valid.
	now := time.Unix(100, 0)
	l := ledger(t, &now, true)
	lease, err := l.Reserve(context.Background(), budget.Reservation{
		Kind: budget.Model, CostKnown: true, Usage: budget.Usage{InputTokens: 2048, OutputTokens: 512, Cost: 100},
	})
	if err != nil {
		t.Fatal(err)
	}
	before := l.Snapshot()
	ctx, cancel := context.WithCancel(context.Background())
	// Act: Check is a clock/context gate, not another capacity admission.
	active := l.Check(ctx)
	now = now.Add(5 * time.Second)
	expired := l.Check(ctx)
	cancel()
	both := l.Check(ctx)
	after := l.Snapshot()
	settled := lease.Settle(budget.Usage{InputTokens: 20, OutputTokens: 10, Cost: 30}, true)
	// Assert: equality expires, cancellation and deadline coexist, late settlement remains permitted exactly once.
	if active != nil || !errors.Is(expired, context.DeadlineExceeded) || !errors.Is(both, context.Canceled) ||
		!errors.Is(
			both,
			context.DeadlineExceeded,
		) || after != before || settled != nil || l.Snapshot().Outstanding != 0 ||
		!errors.Is(lease.Settle(budget.Usage{}, true), budget.ErrSettled) {
		t.Fatal(active, expired, both, settled, after)
	}
}

func TestNilDeadlineCheckRejects(t *testing.T) {
	// Arrange.
	var ledger *budget.Ledger
	// Act.
	err := ledger.Check(context.Background())
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal(err)
	}
}
