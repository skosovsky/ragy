package graphexpand_test

import (
	"context"
	"errors"
	"testing"

	"github.com/skosovsky/ragy/observation"
)

func TestActualGraphExpansionObservesDispatchOnceWithFailingExporter(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	var events []observation.Event
	session, err := observation.New(
		observation.Config{
			MaxEvents: 32,
			Observer: observation.ObserverFunc(func(_ context.Context, event observation.Event) error {
				events = append(events, event)
				return errors.New("private-exporter-body")
			}),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, ledger, err := run(observation.WithSession(context.Background(), session), t, f)
	// Assert.
	if err != nil || result.GraphCalls != 1 || ledger.Snapshot().Occupied.RetrievalCalls != 1 {
		t.Fatal(result, err)
	}
	calls := 0
	pipelines := 0
	for _, event := range events {
		if event.Kind != observation.KindEnd {
			continue
		}
		if event.Stage == observation.StageRetrieval {
			calls++
			if event.Completion.Outcome != observation.OutcomeSuccess || !event.Completion.Count.Known ||
				event.Completion.Usage.InputTokens.Known {
				t.Fatal(event)
			}
		}
		if event.Stage == observation.StagePipeline {
			pipelines++
		}
	}
	if calls != 1 || pipelines != 1 {
		t.Fatal(events)
	}
}
