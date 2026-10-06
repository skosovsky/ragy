//go:build darwin || linux

package lifecycle_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/observation"
)

func lifecycleObserver(t *testing.T, events *[]observation.Event, fail bool) (context.Context, *observation.Session) {
	t.Helper()
	session, err := observation.New(observation.Config{
		MaxEvents: 128,
		Observer: observation.ObserverFunc(func(_ context.Context, event observation.Event) error {
			*events = append(*events, event)
			if fail {
				panic("sensitive exporter failure")
			}
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	return observation.WithSession(context.Background(), session), session
}

func lifecycleEnds(events []observation.Event, stage observation.Stage) []observation.Event {
	var ends []observation.Event
	for _, event := range events {
		if event.Kind == observation.KindEnd && event.Stage == stage {
			ends = append(ends, event)
		}
	}
	return ends
}

func TestLifecycleObservesActualDispatchWithoutIdempotentReplayOrPayload(t *testing.T) {
	// Arrange: a real durable store and a failing diagnostic exporter.
	executor, _, dense, _ := executorFixture(t)
	var events []observation.Event
	ctx, session := lifecycleObserver(t, &events, true)
	plan := plannedManifest()
	plan.ID, plan.Key, plan.Payload = "private-document-id", "private-idempotency-key", "private-payload"

	// Act: prepare and repeat the same ready target stage.
	if _, err := executor.Prepare(ctx, plan); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if _, err := executor.Stage(ctx, "n", plan.ID, "dense", plan.Payload); err != nil {
			t.Fatal(err)
		}
	}

	// Assert: diagnostics cannot dispatch twice or serialize host payloads.
	ends := lifecycleEnds(events, observation.StageLifecycleStage)
	if dense.calls != 1 || len(ends) != 1 || ends[0].Parent == 0 ||
		ends[0].Completion.Outcome != observation.OutcomeSuccess {
		t.Fatal("actual dispatch correlation mismatch", dense.calls, ends)
	}
	if ends[0].Completion.Usage.InputTokens.Known || ends[0].Completion.Count.Known {
		t.Fatal("fabricated remote accounting")
	}
	if session.Stats().Failures != uint64(len(events)) {
		t.Fatal("exporter panic not isolated")
	}
	encoded := fmt.Sprint(events)
	for _, sensitive := range []string{plan.ID, plan.Key, plan.Payload, "sensitive exporter failure"} {
		if strings.Contains(encoded, sensitive) {
			t.Fatal("diagnostic payload leak")
		}
	}
}

func TestLifecycleCancellationPreservesUnknownOutcomeWithoutRetry(t *testing.T) {
	// Arrange: a target loses the stage response; exporter succeeds.
	executor, _, dense, _ := executorFixture(t)
	var events []observation.Event
	ctx, _ := lifecycleObserver(t, &events, false)
	dense.fail = true
	if _, err := executor.Prepare(ctx, plannedManifest()); err != nil {
		t.Fatal(err)
	}

	// Act: dispatch once, then attempt the unknown operation again.
	for range 2 {
		if _, err := executor.Stage(
			ctx,
			"n",
			"pub1",
			"dense",
			"payload",
		); !errors.Is(
			err,
			lifecycle.ErrOutcomeUnknown,
		) {
			t.Fatal(err)
		}
	}

	// Assert: actual canceled dispatch remains distinct from blocked unknown outcome.
	dispatches := lifecycleEnds(events, observation.StageLifecycleStage)
	operations := lifecycleEnds(events, observation.StageLifecycle)
	if dense.calls != 1 || len(dispatches) != 1 || len(operations) != 2 {
		t.Fatal("replayed unknown dispatch")
	}
	if dispatches[0].Completion.Outcome != observation.OutcomeCanceled ||
		dispatches[0].Completion.Error != observation.ErrorDeadline {
		t.Fatal("missing local cancellation", dispatches)
	}
	if operations[1].Completion.Outcome != observation.OutcomePartial ||
		operations[1].Completion.Error != observation.ErrorUnknown {
		t.Fatal("unknown durable outcome mislabeled", operations)
	}
}

func TestLifecycleCleanupWaitingIsPartialAndNotDueDoesNotDispatch(t *testing.T) {
	// Arrange: tombstone publication and a cooperative cleanup port.
	now := time.Unix(100, 0).UTC()
	port := &cleanupHost{waiting: true}
	cleaner, _ := cleanupFixture(t, &now, port)
	var events []observation.Event
	ctx, _ := lifecycleObserver(t, &events, false)

	// Act: one actual due attempt and a premature second attempt.
	if _, err := cleaner.Attempt(ctx, "n", "deleted", "pub1", "dense", false); err != nil {
		t.Fatal(err)
	}
	if _, err := cleaner.Attempt(
		ctx,
		"n",
		"deleted",
		"pub1",
		"dense",
		false,
	); !errors.Is(
		err,
		lifecycle.ErrCleanupNotDue,
	) {
		t.Fatal(err)
	}

	// Assert: observed waiting is partial and no diagnostic creates another cleanup.
	dispatches := lifecycleEnds(events, observation.StageLifecycleCleanup)
	if port.calls != 1 || len(dispatches) != 1 || dispatches[0].Completion.Outcome != observation.OutcomePartial {
		t.Fatal("cleanup outcome mismatch", port.calls, dispatches)
	}
}
