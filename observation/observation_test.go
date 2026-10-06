package observation

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestDisabledHasNoWork(t *testing.T) {
	// Arrange.
	ctx := context.Background()
	// Act.
	next, span := Begin(ctx, StageRetrieval)
	span.End(Completion{})
	// Assert.
	if next != ctx || span != nil || Enabled(ctx) {
		t.Fatal("disabled observation allocated a span")
	}
	allocations := testing.AllocsPerRun(
		100,
		func() { _, disabled := Begin(ctx, StageRetrieval); disabled.End(Completion{}) },
	)
	if allocations != 0 {
		t.Fatalf("disabled allocations = %v", allocations)
	}
}
func TestCapacityReservesPairsAndEndIsOnce(t *testing.T) {
	// Arrange.
	var events []Event
	session, err := New(
		Config{
			MaxEvents: 2,
			Observer:  ObserverFunc(func(_ context.Context, e Event) error { events = append(events, e); return nil }),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx := WithSession(context.Background(), session)
	// Act.
	parent, span := Begin(ctx, StagePipeline)
	_, dropped := Begin(parent, StageRetrieval)
	span.End(Finish(nil, Count{Known: true}))
	span.End(Completion{Outcome: OutcomeFailed})
	// Assert.
	if dropped != nil || len(events) != 2 || events[1].Completion.Outcome != OutcomeEmpty ||
		session.Stats() != (Stats{Events: 2, Dropped: 1}) {
		t.Fatalf("events=%+v stats=%+v", events, session.Stats())
	}
}
func TestConcurrentSerializedCallbackAndSafeStats(t *testing.T) {
	// Arrange.
	const workers = 40
	var session *Session
	var active, maximum atomic.Int64
	var events []Event
	observer := ObserverFunc(func(ctx context.Context, e Event) error {
		now := active.Add(1)
		defer active.Add(-1)
		if now > maximum.Load() {
			maximum.Store(now)
		}
		if Enabled(ctx) {
			t.Error("callback context observation still enabled")
		}
		_ = session.Stats()
		events = append(events, e)
		return nil
	})
	var err error
	session, err = New(Config{MaxEvents: 2 * (workers + 1), Observer: observer})
	if err != nil {
		t.Fatal(err)
	}
	rootCtx, root := Begin(WithSession(context.Background(), session), StagePipeline)
	var wait sync.WaitGroup
	// Act.
	for index := range workers {
		wait.Go(func() {
			ctx := WithQuery(rootCtx, uint64(index))
			ctx = WithBranch(ctx, 0)
			_, span := Begin(ctx, StageRetrieval)
			span.End(Finish(nil, Count{Known: true, Value: 1}))
			span.End(Completion{})
		})
	}
	wait.Wait()
	root.End(Finish(nil, Count{}))
	// Assert.
	if maximum.Load() != 1 || len(events) != 2*(workers+1) {
		t.Fatalf("max=%v events=%v", maximum.Load(), len(events))
	}
	starts := map[uint64]Event{}
	for _, event := range events {
		if event.Kind == KindStart {
			starts[event.Operation] = event
			continue
		}
		start, ok := starts[event.Operation]
		if !ok || start.Stage != event.Stage {
			t.Fatalf("unmatched completion %+v", event)
		}
		delete(starts, event.Operation)
		if event.Stage == StageRetrieval &&
			(event.Parent != 1 || !event.Query.Known || !event.Branch.Known || event.Branch.Value != 0) {
			t.Fatalf("correlation %+v", event)
		}
	}
	if len(starts) != 0 {
		t.Fatal("unmatched starts")
	}
}
func TestObserverFailuresDoNotChangeOperation(t *testing.T) {
	for _, panics := range []bool{false, true} {
		t.Run(map[bool]string{false: "error", true: "panic"}[panics], func(t *testing.T) {
			// Arrange.
			callback := ObserverFunc(func(context.Context, Event) error {
				if panics {
					panic("sensitive diagnostic failure")
				}
				return errors.New("sensitive diagnostic failure")
			})
			session, err := New(Config{MaxEvents: 2, Observer: callback})
			if err != nil {
				t.Fatal(err)
			}
			dispatches := 0
			// Act.
			_, span := Begin(WithSession(context.Background(), session), StageRetrieval)
			dispatches++
			span.End(Finish(nil, Count{Known: true, Value: 1}))
			// Assert.
			if dispatches != 1 || session.Stats() != (Stats{Events: 2, Failures: 2}) {
				t.Fatalf("dispatch=%v stats=%+v", dispatches, session.Stats())
			}
		})
	}
}
func TestUnknownUsageNormalizedAndOutcomeClassification(t *testing.T) {
	// Arrange.
	cases := []struct {
		err     error
		count   Count
		outcome Outcome
		class   ErrorClass
	}{{nil, Count{Known: true}, OutcomeEmpty, ErrorNone}, {nil, Count{}, OutcomeSuccess, ErrorNone}, {errors.New("secret"), Count{Known: true, Value: 1}, OutcomePartial, ErrorUnknown}, {ragy.ErrUnsupported, Count{}, OutcomeUnsupported, ErrorUnsupported}, {context.Canceled, Count{}, OutcomeCanceled, ErrorCanceled}, {context.DeadlineExceeded, Count{}, OutcomeCanceled, ErrorDeadline}, {errors.New("secret"), Count{}, OutcomeFailed, ErrorUnknown}}
	// Act and assert.
	for _, test := range cases {
		got := Finish(test.err, test.count)
		if got.Outcome != test.outcome || got.Error != test.class {
			t.Fatalf("got %+v", got)
		}
	}
	var terminal Event
	session, err := New(
		Config{
			MaxEvents: 2,
			Observer:  ObserverFunc(func(_ context.Context, event Event) error { terminal = event; return nil }),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	_, span := Begin(WithSession(context.Background(), session), StageEncoding)
	span.End(
		Completion{Outcome: OutcomeSuccess, Count: Count{Value: 999}, Usage: Usage{InputTokens: Count{Value: 999}}},
	)
	if terminal.Completion.Count.Value != 0 || terminal.Completion.Usage.InputTokens.Value != 0 ||
		terminal.Completion.Usage.InputTokens.Known {
		t.Fatal("unobserved usage exported")
	}
}
func TestRejectInvalidSession(t *testing.T) {
	// Arrange.
	var typedNil ObserverFunc
	cases := []Config{
		{MaxEvents: 0},
		{MaxEvents: 1, Observer: ObserverFunc(func(context.Context, Event) error { return nil })},
		{MaxEvents: maxEvents + 1, Observer: ObserverFunc(func(context.Context, Event) error { return nil })},
		{MaxEvents: 2, Observer: typedNil},
	}
	// Act and assert.
	for _, config := range cases {
		if _, err := New(config); !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("invalid config accepted %+v", config)
		}
	}
}

type messageMustNotBeReadError struct{}

func (messageMustNotBeReadError) Error() string { panic("error message must not enter observation") }

func TestClassificationDoesNotReadErrorMessage(t *testing.T) {
	// Arrange.
	err := messageMustNotBeReadError{}
	// Act.
	completion := Finish(err, Count{})
	// Assert.
	if completion.Error != ErrorUnknown || completion.Outcome != OutcomeFailed {
		t.Fatalf("unexpected completion %+v", completion)
	}
}

func TestDisabledCorrelationNoAllocation(t *testing.T) {
	// Arrange.
	ctx := context.Background()
	// Act.
	query := WithQuery(ctx, 0)
	branch := WithBranch(ctx, 1)
	allocations := testing.AllocsPerRun(100, func() { _ = WithQuery(ctx, 0); _ = WithBranch(ctx, 1) })
	// Assert.
	if query != ctx || branch != ctx || allocations != 0 {
		t.Fatalf("disabled correlation allocated %v", allocations)
	}
}

func TestConcurrentCapacityNeverLeavesAcceptedStartUnmatched(t *testing.T) {
	// Arrange.
	const capacity = 10
	var events []Event
	session, err := New(
		Config{
			MaxEvents: capacity,
			Observer: ObserverFunc(
				func(_ context.Context, event Event) error { events = append(events, event); return nil },
			),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx := WithSession(context.Background(), session)
	var wait sync.WaitGroup
	// Act.
	for range 40 {
		wait.Go(func() { _, span := Begin(ctx, StageRetrieval); span.End(Finish(nil, Count{})) })
	}
	wait.Wait()
	// Assert.
	stats := session.Stats()
	if stats.Events != capacity || stats.Dropped != 35 {
		t.Fatalf("stats=%+v", stats)
	}
	pending := map[uint64]bool{}
	for _, event := range events {
		if event.Kind == KindStart {
			pending[event.Operation] = true
		} else {
			if !pending[event.Operation] {
				t.Fatal("completion without start")
			}
			delete(pending, event.Operation)
		}
	}
	if len(pending) != 0 {
		t.Fatalf("unmatched starts=%v", len(pending))
	}
}
