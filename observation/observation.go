// Package observation provides bounded payload-free execution diagnostics.
// Callbacks are serialized, synchronous and cooperative. They must return promptly
// and must not recursively use the same session. Diagnostic failures never change
// the observed operation. Local cancellation does not attest remote cancellation.
package observation

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"time"

	"github.com/skosovsky/ragy/access"

	ragy "github.com/skosovsky/ragy"
)

// Stage is a fixed operation kind; host labels and identities are not accepted.
type Stage uint8

const (
	StageUnknown Stage = iota
	StagePipeline
	StageRetrieval
	StageFallback
	StageRescue
	StageRoute
	StageConditional
	StageAggregate
	StageCache
	StageCacheHit
	StageCacheMiss
	StagePlan
	StageAssess
	StageEncoding
	StageModel
	StageFusion
	StageDelivery
	StageSummaryMap
	StageSummaryReduce
	StageLifecycle
	StageLifecycleStage
	StageLifecyclePublish
	StageLifecycleCleanup
	StageLifecyclePrepare
	StageLifecycleReconcile
	StageLifecycleInspect
	StageLifecycleCleanupBegin
	StageLifecycleCleanupInspect
	StageLifecycleCleanupReconcile
	StageLifecycleReuse
	StageLifecycleBootstrap
)

// Outcome describes an observed local operation, never remote billing state.
type Outcome uint8

const (
	OutcomeUnknown Outcome = iota
	OutcomeSuccess
	OutcomeEmpty
	OutcomePartial
	OutcomeFailed
	OutcomeCanceled
	OutcomeUnsupported
	OutcomeExhausted
	OutcomeSkipped
)

// ErrorClass excludes error messages, provider bodies and identities.
type ErrorClass uint8

const (
	ErrorNone ErrorClass = iota
	ErrorUnknown
	ErrorCanceled
	ErrorDeadline
	ErrorUnsupported
	ErrorInvalid
	ErrorProtocol
	ErrorUnavailable
	ErrorProtection
	ErrorResource
)

// Kind distinguishes a real start from a terminal observation.
type Kind uint8

const (
	KindStart Kind = iota + 1
	KindEnd
)

// Count makes unavailable counts distinct from observed zero.
type Count struct {
	Known bool
	Value uint64
}

// Usage contains only actually observed counters. Unknown stays unknown.
type Usage struct {
	InputTokens  Count
	OutputTokens Count
	BilledUnits  Count
}

// Completion is payload-free and copied by value.
type Completion struct {
	Outcome Outcome
	Error   ErrorClass
	Count   Count
	Usage   Usage
}

// Event is immutable by value and contains no host data or arbitrary strings.
type Event struct {
	Kind       Kind
	Stage      Stage
	Operation  uint64
	Parent     uint64
	Query      Count
	Branch     Count
	Elapsed    time.Duration
	Completion Completion
}

// Observer receives diagnostics. Its error or panic is counted, never propagated.
type Observer interface {
	Observe(context.Context, Event) error
}

// ObserverFunc adapts a callback.
type ObserverFunc func(context.Context, Event) error

// Observe implements Observer.
func (f ObserverFunc) Observe(ctx context.Context, event Event) error { return f(ctx, event) }

// Config explicitly limits callbacks, including paired start/end events.
type Config struct {
	MaxEvents uint64
	Observer  Observer
}

// Stats reports local exporter health without storing failed payloads/errors.
type Stats struct {
	Events   uint64
	Dropped  uint64
	Failures uint64
}

// Session is safe for concurrent operations; callback order follows serialization.
// Parallel operation ordinals reflect actual scheduling, not deterministic replay.
type Session struct {
	mu         sync.Mutex
	callbackMu sync.Mutex
	config     Config
	stats      Stats
	reserved   uint64
	next       uint64
}

const maxEvents = 1 << 20

// New requires a finite capacity and nonnil observer.
func New(config Config) (*Session, error) {
	if config.MaxEvents < 2 || config.MaxEvents > maxEvents || config.Observer == nil || nilObserver(config.Observer) {
		return nil, ragy.ErrInvalidArgument
	}
	return &Session{
		mu:         sync.Mutex{},
		callbackMu: sync.Mutex{},
		config:     config,
		stats:      Stats{Events: 0, Dropped: 0, Failures: 0},
		reserved:   0,
		next:       0,
	}, nil
}

type contextKey struct{}
type state struct {
	session       *Session
	operation     uint64
	query, branch Count
}

// WithSession enables diagnostics for this context; nil disables them.
func WithSession(ctx context.Context, session *Session) context.Context {
	return context.WithValue(
		ctx,
		contextKey{},
		state{
			session:   session,
			operation: 0,
			query:     Count{Known: false, Value: 0},
			branch:    Count{Known: false, Value: 0},
		},
	)
}

// WithQuery propagates an attempt-local numeric query ordinal; no query text.
func WithQuery(ctx context.Context, ordinal uint64) context.Context {
	value, _ := ctx.Value(contextKey{}).(state)
	if value.session == nil {
		return ctx
	}
	value.query = Count{Known: true, Value: ordinal}
	return context.WithValue(ctx, contextKey{}, value)
}

// WithBranch propagates an attempt-local numeric branch ordinal; no host identity.
func WithBranch(ctx context.Context, ordinal uint64) context.Context {
	value, _ := ctx.Value(contextKey{}).(state)
	if value.session == nil {
		return ctx
	}
	value.branch = Count{Known: true, Value: ordinal}
	return context.WithValue(ctx, contextKey{}, value)
}

// Enabled reports whether callbacks are configured, without serializing anything.
func Enabled(ctx context.Context) bool {
	value, _ := ctx.Value(contextKey{}).(state)
	return value.session != nil
}

// Span ends at most once. A nil span is an allocation-free disabled operation.
type Span struct {
	session *Session
	event   Event
	started time.Time
	once    sync.Once
	ctx     context.Context
}

// Begin reserves both callbacks before starting. Capacity exhaustion drops the
// pair, leaving no fabricated completion and no unmatched accepted start.
func Begin(ctx context.Context, stage Stage) (context.Context, *Span) {
	value, _ := ctx.Value(contextKey{}).(state)
	if value.session == nil {
		return ctx, nil
	}
	s := value.session
	s.mu.Lock()
	if stage == StageUnknown || stage > StageLifecycleBootstrap || s.stats.Events+s.reserved+2 > s.config.MaxEvents {
		s.stats.Dropped++
		s.mu.Unlock()
		return ctx, nil
	}
	s.next++
	event := Event{
		Kind:      KindStart,
		Stage:     stage,
		Operation: s.next,
		Parent:    value.operation,
		Query:     value.query,
		Branch:    value.branch,
		Elapsed:   0, Completion: unknownCompletion(),
	}
	s.reserved += 2
	span := &Span{session: s, event: event, started: time.Now(), ctx: WithSession(ctx, nil), once: sync.Once{}}
	s.mu.Unlock()
	s.emit(span.ctx, event)
	value.operation = event.Operation
	return context.WithValue(ctx, contextKey{}, value), span
}

// End records supplied observed facts. Unknown counters are normalized to zero;
// callers cannot smuggle unobserved numeric usage into an event.
func (span *Span) End(completion Completion) {
	if span == nil {
		return
	}
	span.once.Do(func() {
		s := span.session

		event := span.event
		event.Kind = KindEnd
		event.Elapsed = time.Since(span.started)
		if completion.Outcome > OutcomeSkipped {
			completion.Outcome = OutcomeUnknown
		}
		if completion.Error > ErrorResource {
			completion.Error = ErrorUnknown
		}
		completion.Count = normalize(completion.Count)
		completion.Usage.InputTokens = normalize(completion.Usage.InputTokens)
		completion.Usage.OutputTokens = normalize(completion.Usage.OutputTokens)
		completion.Usage.BilledUnits = normalize(completion.Usage.BilledUnits)
		event.Completion = completion
		s.emit(span.ctx, event)
	})
}
func normalize(count Count) Count {
	if !count.Known {
		count.Value = 0
	}
	return count
}
func (s *Session) emit(ctx context.Context, event Event) {
	s.callbackMu.Lock()
	defer s.callbackMu.Unlock()
	s.mu.Lock()
	s.reserved--
	s.stats.Events++
	s.mu.Unlock()
	failed := true
	defer func() {
		if recover() != nil || failed {
			s.mu.Lock()
			s.stats.Failures++
			s.mu.Unlock()
		}
	}()
	failed = s.config.Observer.Observe(ctx, event) != nil
}
func nilObserver(observer Observer) bool {
	value := reflect.ValueOf(observer)
	//nolint:exhaustive // All other kinds are non-nil concrete values.
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

// Stats returns an independent snapshot of local diagnostic accounting.
func (s *Session) Stats() Stats {
	if s == nil {
		return Stats{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stats
}

// Classify never calls Error() and accepts only stable error sentinels.
func Classify(err error) ErrorClass {
	switch {
	case err == nil:
		return ErrorNone
	case access.IsProtectionFailure(err):
		return ErrorProtection
	case errors.Is(err, context.Canceled):
		return ErrorCanceled
	case errors.Is(err, context.DeadlineExceeded):
		return ErrorDeadline
	case errors.Is(err, ragy.ErrUnsupported):
		return ErrorUnsupported
	case errors.Is(err, ragy.ErrInvalidArgument):
		return ErrorInvalid
	case errors.Is(err, ragy.ErrProtocol):
		return ErrorProtocol
	case errors.Is(err, ragy.ErrUnavailable):
		return ErrorUnavailable
	default:
		return ErrorUnknown
	}
}

// Finish classifies local errors; count is known only when actually observed.
func Finish(err error, count Count) Completion {
	result := Completion{Outcome: OutcomeSuccess, Error: Classify(err), Count: count, Usage: unknownUsage()}
	switch {
	case errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded):
		result.Outcome = OutcomeCanceled
	case errors.Is(err, ragy.ErrUnsupported):
		result.Outcome = OutcomeUnsupported
	case result.Error == ErrorProtection:
		result.Outcome = OutcomeFailed
	case err != nil && count.Known && count.Value > 0:
		result.Outcome = OutcomePartial
	case err != nil:
		result.Outcome = OutcomeFailed
	case count.Known && count.Value == 0:
		result.Outcome = OutcomeEmpty
	}
	return result
}

func unknownCompletion() Completion { var completion Completion; return completion }
func unknownUsage() Usage           { var usage Usage; return usage }
