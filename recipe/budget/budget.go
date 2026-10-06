// Package budget provides an attempt-local atomic reservation ledger. It does not
// perform pricing, billing, scheduling, dispatch or retries.
package budget

import (
	"context"
	"errors"
	"math"
	"sync"
	"time"

	ragy "github.com/skosovsky/ragy"
)

var (
	ErrExhausted     = errors.New("recipe budget exhausted")
	ErrUnknownPrice  = errors.New("recipe price unavailable")
	ErrUsageExceeded = errors.New("recipe actual usage exceeds reservation")
	ErrSettled       = errors.New("recipe reservation already settled")
)

type Kind string

const (
	Retrieval Kind = "retrieval"
	Model     Kind = "model"
)

// Usage counts tokens and host-defined integer cost units, never provider prices.
type Usage struct {
	InputTokens  uint64
	OutputTokens uint64
	Cost         uint64
}

type Limits struct {
	RetrievalCalls uint64
	ModelCalls     uint64
	Usage          Usage
}

type Config struct {
	Limits           Limits
	Deadline         time.Time
	Now              func() time.Time
	RequireKnownCost bool
}

type Reservation struct {
	Kind      Kind
	Usage     Usage
	CostKnown bool
}

// Snapshot includes settled actual usage plus outstanding/conservatively retained
// reservations. Calls are never refunded, including failed or canceled calls.
type Snapshot struct {
	Occupied     Limits
	Actual       Usage
	Outstanding  uint64
	UnknownUsage uint64
	UnknownCost  bool
}

type Ledger struct {
	mu     sync.Mutex
	config Config
	state  Snapshot
}

// Lease is single-settlement. Copies share the same private lease state.
type Lease struct{ state *leaseState }

type leaseState struct {
	ledger      *Ledger
	reservation Reservation
	settled     bool
}

func New(config Config) (*Ledger, error) {
	if config.Now == nil || config.Deadline.IsZero() ||
		config.Limits.ModelCalls > math.MaxUint64-config.Limits.RetrievalCalls {
		return nil, ragy.ErrInvalidArgument
	}
	var initial Snapshot
	return &Ledger{mu: sync.Mutex{}, config: config, state: initial}, nil
}

// Deadline is the immutable host deadline for this attempt. A nil ledger has no deadline.
func (l *Ledger) Deadline() time.Time {
	if l == nil {
		return time.Time{}
	}
	return l.config.Deadline
}

// Check checks context and the attempt deadline using the ledger's own clock.
// It does not reserve, settle or inspect remaining call/token/cost capacity.
// Call it at completion/delivery boundaries as well as dispatch admission;
// injected clock leaps cannot be observed by the cooperative context timer.
func (l *Ledger) Check(ctx context.Context) error {
	if l == nil {
		return ragy.ErrInvalidArgument
	}
	contextErr := ctx.Err()
	if l.config.Now().Before(l.config.Deadline) {
		return contextErr
	}
	if contextErr == nil || contextErr == context.DeadlineExceeded {
		return context.DeadlineExceeded
	}
	return errors.Join(contextErr, context.DeadlineExceeded)
}

// Context bounds cooperative work by the remaining attempt time and parent deadline.
// Remaining time uses the host clock; the context timer uses elapsed wall time.
// The host clock must be concurrency-safe. Reserve also rechecks the host deadline.
func (l *Ledger) Context(ctx context.Context) (context.Context, context.CancelFunc) {
	if l == nil {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, l.config.Deadline.Sub(l.config.Now()))
}

// Reserve atomically checks every dimension before admitting one dispatch. A
// successful reservation spends a call even if dispatch subsequently fails. Use
// the same ledger for all concurrent branches of one attempt.
func (l *Ledger) Reserve(ctx context.Context, request Reservation) (Lease, error) {
	if l == nil || (request.Kind != Retrieval && request.Kind != Model) ||
		(!request.CostKnown && request.Usage.Cost != 0) {
		return Lease{}, ragy.ErrInvalidArgument
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.Check(ctx); err != nil {
		return Lease{}, err
	}
	if l.config.RequireKnownCost && !request.CostKnown {
		return Lease{}, ErrUnknownPrice
	}
	next := l.state.Occupied
	if !fits(next.Usage, request.Usage, l.config.Limits.Usage) || !l.admitCall(&next, request.Kind) {
		return Lease{}, ErrExhausted
	}
	next.Usage = add(next.Usage, request.Usage)
	l.state.Occupied = next
	l.state.Outstanding++
	l.state.UnknownCost = l.state.UnknownCost || !request.CostKnown
	return Lease{state: &leaseState{ledger: l, reservation: request, settled: false}}, nil
}

func (l *Ledger) admitCall(next *Limits, kind Kind) bool {
	if kind == Retrieval {
		if next.RetrievalCalls >= l.config.Limits.RetrievalCalls {
			return false
		}
		next.RetrievalCalls++
		return true
	}
	if next.ModelCalls >= l.config.Limits.ModelCalls {
		return false
	}
	next.ModelCalls++
	return true
}

// Settle accounts known usage after a call, even when its context has expired.
// Unknown usage keeps the full reservation. Excess usage is a protocol failure:
// the reservation remains charged, no refund or successful completion is reported.
// Advisory unknown pricing cannot enforce a cost cap; it remains explicitly unknown.
func (lease Lease) Settle(actual Usage, known bool) error {
	if lease.state == nil {
		return ragy.ErrInvalidArgument
	}
	l := lease.state.ledger
	l.mu.Lock()
	defer l.mu.Unlock()
	if lease.state.settled {
		return ErrSettled
	}
	lease.state.settled = true
	l.state.Outstanding--
	reserved := lease.state.reservation.Usage
	if !known {
		l.state.UnknownUsage++
		return nil
	}
	var zero Usage
	if !fits(zero, actual, reserved) {
		l.state.UnknownUsage++
		return ErrUsageExceeded
	}
	l.state.Occupied.Usage = add(subtract(l.state.Occupied.Usage, reserved), actual)
	l.state.Actual = add(l.state.Actual, actual)
	return nil
}

func (l *Ledger) Snapshot() Snapshot {
	if l == nil {
		return Snapshot{}
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.state
}

// Subtracting from the limit avoids overflow when counters approach uint64 max.
func fits(used, requested, limit Usage) bool {
	return used.InputTokens <= limit.InputTokens && requested.InputTokens <= limit.InputTokens-used.InputTokens &&
		used.OutputTokens <= limit.OutputTokens && requested.OutputTokens <= limit.OutputTokens-used.OutputTokens &&
		used.Cost <= limit.Cost && requested.Cost <= limit.Cost-used.Cost
}
func add(a, b Usage) Usage {
	return Usage{
		InputTokens:  a.InputTokens + b.InputTokens,
		OutputTokens: a.OutputTokens + b.OutputTokens,
		Cost:         a.Cost + b.Cost,
	}
}
func subtract(a, b Usage) Usage {
	return Usage{
		InputTokens:  a.InputTokens - b.InputTokens,
		OutputTokens: a.OutputTokens - b.OutputTokens,
		Cost:         a.Cost - b.Cost,
	}
}
