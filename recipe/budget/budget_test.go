package budget_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/recipe/budget"
)

func ledger(t *testing.T, now *time.Time, required bool) *budget.Ledger {
	t.Helper()
	l, err := budget.New(budget.Config{
		Limits: budget.Limits{
			RetrievalCalls: 3,
			ModelCalls:     4,
			Usage:          budget.Usage{InputTokens: 2048, OutputTokens: 512, Cost: 100},
		},
		Deadline:         now.Add(5 * time.Second),
		Now:              func() time.Time { return *now },
		RequireKnownCost: required,
	})
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func TestParallelReservationsCannotOverrunCost(t *testing.T) {
	// Arrange: four concurrent model attempts, three fit the common cost cap.
	now := time.Unix(100, 0)
	l := ledger(t, &now, true)
	outcomes := make(chan error, 4)
	var group sync.WaitGroup
	// Act.
	for range 4 {
		group.Go(func() {
			lease, err := l.Reserve(context.Background(), budget.Reservation{
				Kind: budget.Model, CostKnown: true, Usage: budget.Usage{InputTokens: 100, OutputTokens: 100, Cost: 30},
			})
			if err == nil {
				err = lease.Settle(budget.Usage{InputTokens: 100, OutputTokens: 100, Cost: 30}, true)
			}
			outcomes <- err
		})
	}
	group.Wait()
	close(outcomes)
	// Assert.
	success := 0
	for err := range outcomes {
		if err == nil {
			success++
		} else if !errors.Is(err, budget.ErrExhausted) {
			t.Fatal(err)
		}
	}
	s := l.Snapshot()
	if success != 3 || s.Occupied.ModelCalls != 3 || s.Occupied.Usage.Cost != 90 || s.Outstanding != 0 {
		t.Fatal("parallel reservation exceeded common cap", s)
	}
}

func TestUnknownUsageNeverRefundsAndSettlementCannotReplay(t *testing.T) {
	// Arrange.
	now := time.Unix(100, 0)
	l := ledger(t, &now, true)
	lease, err := l.Reserve(context.Background(), budget.Reservation{
		Kind: budget.Model, CostKnown: true, Usage: budget.Usage{InputTokens: 2048, OutputTokens: 512, Cost: 30},
	})
	if err != nil {
		t.Fatal(err)
	}
	copyOfLease := lease
	// Act.
	if err = lease.Settle(budget.Usage{}, false); err != nil {
		t.Fatal(err)
	}
	_, exhausted := l.Reserve(context.Background(), budget.Reservation{
		Kind: budget.Model, CostKnown: true, Usage: budget.Usage{InputTokens: 1},
	})
	// Assert.
	if !errors.Is(exhausted, budget.ErrExhausted) ||
		!errors.Is(copyOfLease.Settle(budget.Usage{}, true), budget.ErrSettled) {
		t.Fatal("unknown usage or copied lease bypassed accounting")
	}
	s := l.Snapshot()
	if s.UnknownUsage != 1 || s.Occupied.Usage.InputTokens != 2048 || s.Occupied.ModelCalls != 1 {
		t.Fatal(s)
	}
}

func TestKnownSettlementRefundsOnlyUsageAndRejectsOverrun(t *testing.T) {
	// Arrange.
	now := time.Unix(100, 0)
	l := ledger(t, &now, true)
	request := budget.Reservation{
		Kind:      budget.Model,
		CostKnown: true,
		Usage:     budget.Usage{InputTokens: 100, OutputTokens: 100, Cost: 30},
	}
	first, err := l.Reserve(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	if err = first.Settle(budget.Usage{InputTokens: 20, OutputTokens: 10, Cost: 10}, true); err != nil {
		t.Fatal(err)
	}
	second, err := l.Reserve(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	overrun := second.Settle(budget.Usage{InputTokens: 101}, true)
	// Assert.
	s := l.Snapshot()
	if !errors.Is(overrun, budget.ErrUsageExceeded) || s.Occupied.ModelCalls != 2 ||
		s.Occupied.Usage.InputTokens != 120 ||
		s.Occupied.Usage.Cost != 40 ||
		s.Actual.InputTokens != 20 ||
		s.UnknownUsage != 1 {
		t.Fatal(s, overrun)
	}
}

func TestPriceDeadlineCancellationAndOverflowBeforeAdmission(t *testing.T) {
	// Arrange.
	now := time.Unix(100, 0)
	l := ledger(t, &now, true)
	request := budget.Reservation{Kind: budget.Model, CostKnown: false, Usage: budget.Usage{}}
	// Act and Assert.
	if _, err := l.Reserve(context.Background(), request); !errors.Is(err, budget.ErrUnknownPrice) {
		t.Fatal(err)
	}
	advisory := ledger(t, &now, false)
	lease, err := advisory.Reserve(context.Background(), request)
	if err != nil || !advisory.Snapshot().UnknownCost {
		t.Fatal(err)
	}
	if err = lease.Settle(budget.Usage{}, false); err != nil {
		t.Fatal(err)
	}
	request.CostKnown = true
	request.Usage.InputTokens = ^uint64(0)
	if _, err = l.Reserve(context.Background(), request); !errors.Is(err, budget.ErrExhausted) {
		t.Fatal(err)
	}
	request.Usage.InputTokens = 0
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = l.Reserve(canceled, request); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	now = now.Add(5 * time.Second)
	if _, err = l.Reserve(context.Background(), request); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	if l.Snapshot() != (budget.Snapshot{}) {
		t.Fatal("failed admission spent budget")
	}
	var invalid budget.Lease
	if err = invalid.Settle(budget.Usage{}, true); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal(err)
	}
}

func TestCallAndOutputLimitsAndConcurrentLeaseCopies(t *testing.T) {
	// Arrange.
	now := time.Unix(100, 0)
	l := ledger(t, &now, true)
	lease, err := l.Reserve(context.Background(), budget.Reservation{
		Kind: budget.Model, CostKnown: true, Usage: budget.Usage{OutputTokens: 512},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, overflow := l.Reserve(context.Background(), budget.Reservation{
		Kind: budget.Model, CostKnown: true, Usage: budget.Usage{OutputTokens: 1},
	})
	var group sync.WaitGroup
	outcomes := make(chan error, 2)
	for range 2 {
		group.Go(func() { outcomes <- lease.Settle(budget.Usage{OutputTokens: 20}, true) })
	}
	group.Wait()
	close(outcomes)
	// Assert.
	if !errors.Is(overflow, budget.ErrExhausted) {
		t.Fatal(overflow)
	}
	succeeded, replayed := 0, 0
	for outcome := range outcomes {
		switch {
		case outcome == nil:
			succeeded++
		case errors.Is(outcome, budget.ErrSettled):
			replayed++
		default:
			t.Fatal(outcome)
		}
	}
	if succeeded != 1 || replayed != 1 || l.Snapshot().Occupied.Usage.OutputTokens != 20 {
		t.Fatal("concurrent lease refund replay")
	}
	for range 3 {
		call, callErr := l.Reserve(context.Background(), budget.Reservation{Kind: budget.Retrieval, CostKnown: true})
		if callErr != nil {
			t.Fatal(callErr)
		}
		if callErr = call.Settle(budget.Usage{}, true); callErr != nil {
			t.Fatal(callErr)
		}
	}
	if _, err = l.Reserve(
		context.Background(),
		budget.Reservation{Kind: budget.Retrieval, CostKnown: true},
	); !errors.Is(
		err,
		budget.ErrExhausted,
	) {
		t.Fatal("settlement refunded dispatched retrieval calls", err)
	}
	if _, err = budget.New(budget.Config{Now: func() time.Time { return now }, Deadline: now.Add(time.Second),
		Limits: budget.Limits{ModelCalls: ^uint64(0), RetrievalCalls: 1},
	}); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("combined call counter overflow accepted", err)
	}
}
