package extraction_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/recipe/budget"
)

func sharedLedger(t *testing.T, now func() time.Time, duration time.Duration) *budget.Ledger {
	t.Helper()
	ledger, err := budget.New(budget.Config{
		Limits:   budget.Limits{ModelCalls: 2, Usage: budget.Usage{InputTokens: 4096, OutputTokens: 512, Cost: 100}},
		Deadline: now().Add(duration), Now: now, RequireKnownCost: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	return ledger
}

func TestModelContextComposesIndependentClockDomains(t *testing.T) {
	for _, earliest := range []string{"parent", "shared", "local"} {
		t.Run(earliest, func(t *testing.T) {
			// Arrange: fake clock epochs differ from each other and real parent time.
			f := newFixture(t)
			localNow, ledgerNow := time.Unix(1000000, 0), time.Unix(100, 0)
			parentDuration, sharedDuration, localDuration := 10*time.Second, 5*time.Second, 5*time.Second
			switch earliest {
			case "parent":
				parentDuration = time.Second
			case "shared":
				sharedDuration = time.Second
			case "local":
				localDuration = time.Second
			}
			f.config.Now = func() time.Time { return localNow }
			f.config.Duration = localDuration
			f.ledger = sharedLedger(t, func() time.Time { return ledgerNow }, sharedDuration)
			ctx, cancel := context.WithTimeout(context.Background(), parentDuration)
			defer cancel()
			var observed time.Time
			var remaining time.Duration
			original := f.config.Model
			f.config.Model = func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
				deadline, exists := ctx.Deadline()
				if !exists {
					t.Fatal("missing composed deadline")
				}
				observed = deadline
				remaining = time.Until(deadline)
				return original(ctx, input)
			}
			adapter, err := extraction.New(f.config)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			started := time.Now()
			result, err := adapter.Extract(ctx, f.read, f.ledger, f.input)
			assertComposedDeadline(ctx, t, earliest, started, observed)
			// Assert: supplied model context uses the minimum remaining interval, not fake absolute timestamps.
			if err != nil || len(result.Extraction.Entities) != 2 || remaining <= 0 || remaining > time.Second ||
				f.calls != 1 || f.ledger.Snapshot().Outstanding != 0 {
				t.Fatal("clock composition", remaining, err)
			}
		})
	}
}

func assertComposedDeadline(ctx context.Context, t *testing.T, earliest string, started, observed time.Time) {
	t.Helper()
	parentDeadline, _ := ctx.Deadline()
	if earliest == "parent" {
		if !observed.Equal(parentDeadline) {
			t.Fatal("parent deadline changed", observed, parentDeadline)
		}
	} else if observed.Before(started.Add(time.Second)) {
		t.Fatal("remaining interval narrowed unexpectedly", observed, started)
	}
}

func TestIndependentClockExpiryAtEveryCallbackBoundary(t *testing.T) {
	phases := []string{
		"clone-access",
		"attributes",
		"admission",
		"quote",
		"tokens",
		"model",
		"clone-first",
		"entity",
		"clone-second",
		"relation",
		"clone-final",
	}
	for _, clock := range []string{"shared", "local"} {
		for _, phase := range phases {
			t.Run(clock+"/"+phase, func(t *testing.T) {
				// Arrange: ledger and adapter have unrelated immutable deadline epochs.
				f := newFixture(t)
				localNow, sharedNow := time.Unix(1000000, 0), time.Unix(100, 0)
				f.config.Now = func() time.Time { return localNow }
				f.config.Duration = time.Minute
				f.ledger = sharedLedger(t, func() time.Time { return sharedNow }, 5*time.Second)
				leap := func() {
					if clock == "shared" {
						sharedNow = sharedNow.Add(5 * time.Second)
					} else {
						localNow = localNow.Add(time.Minute)
					}
				}
				instrumentBoundary(f, phase, leap)
				adapter, err := extraction.New(f.config)
				if err != nil {
					t.Fatal(err)
				}
				// Act.
				result, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
				// Assert: equality expires; no output escapes; model reservation settles exactly once when dispatched.
				modelCalls := uint64(1)
				if slices.Contains([]string{"clone-access", "attributes", "admission", "quote", "tokens"}, phase) {
					modelCalls = 0
				}
				assertExpiredBoundary(t, f, result, err, modelCalls)
			})
		}
	}
}

func assertExpiredBoundary(
	t *testing.T, f *fixture, result extraction.Result[string, string, attributes], err error, modelCalls uint64,
) {
	t.Helper()
	snapshot := f.ledger.Snapshot()
	if !errors.Is(err, context.DeadlineExceeded) || len(result.Extraction.Entities) != 0 ||
		len(result.Extraction.Relations) != 0 ||
		result.Usage.Known ||
		result.Usage.Value != (budget.Usage{}) ||
		uint64(f.calls) != modelCalls ||
		snapshot.Occupied.ModelCalls != modelCalls ||
		snapshot.Outstanding != 0 ||
		snapshot.Actual.InputTokens != 20*modelCalls {
		t.Fatal("expired boundary escaped/accounting", err, snapshot, result)
	}
}

func instrumentBoundary(f *fixture, phase string, leap func()) {
	if instrumentAdmissionBoundary(f, phase, leap) {
		return
	}
	switch phase {
	case "quote":
		original := f.config.Quote
		f.config.Quote = func(ctx context.Context) (budget.Reservation, error) { v, err := original(ctx); leap(); return v, err }
	case "tokens":
		original := f.config.CountInputTokens
		f.config.CountInputTokens = func(input extraction.ModelInput) (uint64, error) { v, err := original(input); leap(); return v, err }
	case "model":
		original := f.config.Model
		f.config.Model = func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
			v, u, err := original(ctx, input)
			leap()
			return v, u, err
		}
	default:
		instrumentProjectionBoundary(f, phase, leap)
	}
}

func instrumentAdmissionBoundary(f *fixture, phase string, leap func()) bool {
	switch phase {
	case "clone-access":
		original := f.config.CloneAccess
		f.config.CloneAccess = func(value acl) (acl, error) { v, err := original(value); leap(); return v, err }
	case "attributes":
		original := f.config.Attributes
		f.config.Attributes = func(value acl) (filter.RawAttributes, error) { v, err := original(value); leap(); return v, err }
	case "admission":
		original := f.config.AdmitSnippet
		f.config.AdmitSnippet = func(ctx context.Context, read access.Binding, value extraction.Snippet[acl]) error {
			err := original(ctx, read, value)
			leap()
			return err
		}
	default:
		return false
	}
	return true
}

func instrumentProjectionBoundary(f *fixture, phase string, leap func()) {
	switch phase {
	case "entity":
		original := f.config.ValidateEntity
		f.config.ValidateEntity = func(kind string, value attributes) error { err := original(kind, value); leap(); return err }
	case "relation":
		original := f.config.ValidateRelation
		f.config.ValidateRelation = func(kind, from, to string, value attributes) error {
			err := original(kind, from, to, value)
			leap()
			return err
		}
	default:
		at := 1
		switch phase {
		case "clone-second":
			at = 2
		case "clone-final":
			at = 6
		}
		clones := 0
		original := f.config.CloneAttributes
		f.config.CloneAttributes = func(value attributes) (attributes, error) {
			v, err := original(value)
			clones++
			if clones == at {
				leap()
			}
			return v, err
		}
	}
}

func TestSharedExpiryWhileModelBlockedSuppressesValidLateOutput(t *testing.T) {
	// Arrange: a concurrency-safe ledger clock, cooperative callback barrier and unknown actual usage.
	f := newFixture(t)
	var clock atomic.Int64
	clock.Store(time.Unix(100, 0).UnixNano())
	f.ledger = sharedLedger(t, func() time.Time { return time.Unix(0, clock.Load()) }, 5*time.Second)
	f.config.Duration = time.Minute
	entered, release := make(chan struct{}), make(chan struct{})
	f.config.Model = func(context.Context, extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
		f.calls++
		close(entered)
		<-release
		return f.output, extraction.Usage{}, nil
	}
	adapter, err := extraction.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	// Act: ledger expires without waiting for the real context timer.
	go func() {
		output, callErr := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
		if len(output.Extraction.Entities) != 0 {
			done <- errors.New("late payload")
			return
		}
		done <- callErr
	}()
	<-entered
	clock.Add(int64(5 * time.Second))
	close(release)
	err = <-done
	// Assert: zero delivery and conservative unknown accounting, with no retry.
	snapshot := f.ledger.Snapshot()
	if !errors.Is(err, context.DeadlineExceeded) || f.calls != 1 || snapshot.Outstanding != 0 ||
		snapshot.UnknownUsage != 1 ||
		snapshot.Occupied.Usage.InputTokens != 100 {
		t.Fatal(err, snapshot)
	}
}

func TestJustBeforeSharedDeadlineSucceeds(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	now := time.Unix(100, 0)
	f.ledger = sharedLedger(t, func() time.Time { return now }, 5*time.Second)
	f.config.Duration = time.Minute
	original := f.config.Model
	f.config.Model = func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
		now = now.Add(5*time.Second - time.Nanosecond)
		return original(ctx, input)
	}
	adapter, err := extraction.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	output, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
	// Assert.
	if err != nil || len(output.Extraction.Entities) != 2 || len(output.Extraction.Relations) != 1 || f.calls != 1 ||
		f.ledger.Snapshot().Outstanding != 0 {
		t.Fatal(err)
	}
}

func TestSourceBytesAndActualEnvelopeTokensHaveSeparateBounds(t *testing.T) {
	for _, refuse := range []bool{false, true} {
		// Arrange: identity/framing is larger than the exact source-text byte cap.
		f := newFixture(t)
		f.config.MaxInputBytes = 0
		for _, snippet := range f.input {
			f.config.MaxInputBytes += len(snippet.Mapping.Text())
		}
		f.config.OntologyIdentity = strings.Repeat("ontology", 50)
		inputLimit := uint64(2048)
		if refuse {
			inputLimit = 100
		}
		f.config.Quote = func(context.Context) (budget.Reservation, error) {
			return budget.Reservation{
				Kind:      budget.Model,
				CostKnown: true,
				Usage:     budget.Usage{InputTokens: inputLimit, OutputTokens: 50, Cost: 30},
			}, nil
		}
		var counted string
		f.config.CountInputTokens = func(input extraction.ModelInput) (uint64, error) {
			counted = modelEnvelope(t, input)
			return uint64(len(counted)), nil
		}
		f.config.Model = func(_ context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
			f.calls++
			if modelEnvelope(t, input) != counted {
				t.Fatal("counter and provider envelope differ")
			}
			return f.output, extraction.Usage{
				Known: true,
				Value: budget.Usage{InputTokens: uint64(len(counted)), OutputTokens: 10, Cost: 30},
			}, nil
		}
		adapter, err := extraction.New(f.config)
		if err != nil {
			t.Fatal(err)
		}
		// Act.
		output, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
		// Assert: source cap does not silently become request-envelope byte cap; complete input count rejects before reserve.
		if len(counted) <= f.config.MaxInputBytes {
			t.Fatal("missing envelope overhead")
		}
		if refuse {
			if !errors.Is(err, budget.ErrExhausted) || f.calls != 0 || f.ledger.Snapshot().Occupied.ModelCalls != 0 ||
				len(output.Extraction.Entities) != 0 {
				t.Fatal(err)
			}
		} else if err != nil || f.calls != 1 || len(output.Extraction.Entities) != 2 {
			t.Fatal(err)
		}
	}
}

func modelEnvelope(t *testing.T, input extraction.ModelInput) string {
	t.Helper()
	encoded, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	return `{"provider_instruction":"extract source-bound graph","input":` + string(encoded) + `}`
}

func TestParentCancellationWhileModelBlockedSettlesAndSuppressesProjection(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered := make(chan struct{})
	f.config.Model = func(ctx context.Context, _ extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
		f.calls++
		close(entered)
		<-ctx.Done()
		return f.output, extraction.Usage{
			Known: true,
			Value: budget.Usage{InputTokens: 20, OutputTokens: 10, Cost: 30},
		}, nil
	}
	projected := false
	f.config.CloneAttributes = func(value attributes) (attributes, error) { projected = true; return value, nil }
	adapter, err := extraction.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	// Act.
	go func() {
		output, callErr := adapter.Extract(ctx, f.read, f.ledger, f.input)
		if len(output.Extraction.Entities) != 0 {
			done <- errors.New("parent-cancel payload")
			return
		}
		done <- callErr
	}()
	<-entered
	cancel()
	err = <-done
	// Assert.
	snapshot := f.ledger.Snapshot()
	if !errors.Is(err, context.Canceled) || projected || f.calls != 1 || snapshot.Occupied.ModelCalls != 1 ||
		snapshot.Outstanding != 0 ||
		snapshot.Actual.InputTokens != 20 {
		t.Fatal(err, snapshot)
	}
}

func TestCloneFailureAndSharedExpiryRemainDiscoverable(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	now := time.Unix(100, 0)
	f.ledger = sharedLedger(t, func() time.Time { return now }, 5*time.Second)
	f.config.Duration = time.Minute
	// Use a stable cause so the assertion checks retained callback identity.
	cause := errors.New("clone failure")
	f.config.CloneAttributes = func(attributes) (attributes, error) { now = now.Add(5 * time.Second); return attributes{}, cause }
	adapter, err := extraction.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	output, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
	// Assert.
	if !errors.Is(err, cause) || !errors.Is(err, context.DeadlineExceeded) || len(output.Extraction.Entities) != 0 ||
		f.calls != 1 ||
		f.ledger.Snapshot().Outstanding != 0 {
		t.Fatal(err)
	}
}

func TestExpiredEntryRejectsBeforeHostCallbacksOrReservation(t *testing.T) {
	for _, scope := range []string{"parent", "shared", "local"} {
		// Arrange.
		f := newFixture(t)
		ctx, cancel := context.WithCancel(context.Background())
		sharedNow := time.Unix(100, 0)
		f.ledger = sharedLedger(t, func() time.Time { return sharedNow }, 5*time.Second)
		switch scope {
		case "parent":
			cancel()
		case "shared":
			sharedNow = sharedNow.Add(5 * time.Second)
		case "local":
			reads := 0
			f.config.Now = func() time.Time {
				reads++
				if reads == 1 {
					return time.Unix(1000000, 0)
				}
				return time.Unix(1000000, 0).Add(f.config.Duration)
			}
		}
		clones, quotes := 0, 0
		f.config.CloneAccess = func(value acl) (acl, error) { clones++; return value, nil }
		original := f.config.Quote
		f.config.Quote = func(ctx context.Context) (budget.Reservation, error) { quotes++; return original(ctx) }
		adapter, err := extraction.New(f.config)
		if err != nil {
			cancel()
			t.Fatal(err)
		}
		// Act.
		output, err := adapter.Extract(ctx, f.read, f.ledger, f.input)
		cancel()
		// Assert.
		cause := context.DeadlineExceeded
		if scope == "parent" {
			cause = context.Canceled
		}
		if !errors.Is(err, cause) || len(output.Extraction.Entities) != 0 || f.calls != 0 || f.admitted != 0 ||
			clones != 0 ||
			quotes != 0 ||
			f.ledger.Snapshot().Occupied.ModelCalls != 0 {
			t.Fatal(scope, err)
		}
	}
}

func TestFinalAuthorityCallbackCannotHideClockExpiry(t *testing.T) {
	for _, clock := range []string{"local", "shared"} {
		// Arrange: calibrate the final authority boundary on a successful identical run.
		finalCheck := finalAuthorityRun(t, clock, 0)
		// Act/Assert: injected expiry inside that final delivery callback must suppress output.
		finalAuthorityRun(t, clock, finalCheck)
	}
}

func finalAuthorityRun(t *testing.T, clock string, finalCheck int) int {
	t.Helper()
	f := newFixture(t)
	localNow, sharedNow := time.Unix(1000000, 0), time.Unix(100, 0)
	f.config.Now = func() time.Time { return localNow }
	f.config.Duration = time.Minute
	f.ledger = sharedLedger(t, func() time.Time { return sharedNow }, time.Minute)
	checks := 0
	bindTestAuthority(t, f, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
		checks++
		if finalCheck > 0 && checks == finalCheck {
			if clock == "local" {
				localNow = localNow.Add(time.Minute)
			} else {
				sharedNow = sharedNow.Add(time.Minute)
			}
		}
		return nil
	}))
	adapter, err := extraction.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	result, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
	if finalCheck == 0 {
		if err != nil || len(result.Extraction.Entities) != 2 {
			t.Fatal(err)
		}
	} else {
		assertExpiredBoundary(t, f, result, err, 1)
		if checks != finalCheck {
			t.Fatal("final callback not reached", checks, finalCheck)
		}
	}
	return checks
}

func bindTestAuthority(t *testing.T, f *fixture, authority access.AuthorityFunc) {
	t.Helper()
	mandatory, err := f.read.Prepare(context.Background(), f.config.Schema, filter.Condition{},
		access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: false})
	if err != nil {
		t.Fatal(err)
	}
	f.read, err = access.Scoped(access.ScopedConfig{
		Snapshot: f.read.Snapshot(), Mandatory: mandatory, Schema: f.config.Schema,
		Publication: f.read.Publication(), Now: func() time.Time { return f.now }, Authority: authority,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestPrepareFailureRetainsClockExpiryWithoutAnotherAuthorityDispatch(t *testing.T) {
	for _, clock := range []string{"local", "shared"} {
		// Arrange: the second authority check runs inside read.Prepare after the initial gate.
		f := newFixture(t)
		localNow, sharedNow := time.Unix(1000000, 0), time.Unix(100, 0)
		f.config.Now = func() time.Time { return localNow }
		f.config.Duration = time.Minute
		f.ledger = sharedLedger(t, func() time.Time { return sharedNow }, time.Minute)
		checks, clones := 0, 0
		cause := errors.New("authority failed")
		bindTestAuthority(t, f, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			checks++
			if checks == 2 {
				if clock == "local" {
					localNow = localNow.Add(time.Minute)
				} else {
					sharedNow = sharedNow.Add(time.Minute)
				}
				return cause
			}
			return nil
		}))
		f.config.CloneAccess = func(value acl) (acl, error) { clones++; return value, nil }
		adapter, err := extraction.New(f.config)
		if err != nil {
			t.Fatal(err)
		}
		// Act.
		result, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
		// Assert: both independent causes, zero payload/dispatch/reservation and no third authority callback.
		assertExpiredBoundary(t, f, result, err, 0)
		if !errors.Is(err, cause) || !access.IsProtectionFailure(err) || checks != 2 || clones != 0 || f.admitted != 0 {
			t.Fatal(err, checks, clones)
		}
	}
}
