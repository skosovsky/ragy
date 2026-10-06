package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/observation"
)

//nolint:gocognit // Table asserts the full ordered branch correlation contract.
func TestLegacyObservedBranchOutcomes(t *testing.T) {
	tests := []struct {
		name    string
		root    resultNode[stubIntent, NoRequestMeta, string]
		stage   observation.Stage
		wantErr error
	}{
		{
			"fallback",
			resultFallbackNode[stubIntent, NoRequestMeta, string]{
				Primary: resultRetrieverNode[stubIntent, NoRequestMeta, string]{
					Backend: orchestratorStubBackend[stubIntent, string]{},
				},
				Secondary: resultRetrieverNode[stubIntent, NoRequestMeta, string]{
					Backend: orchestratorStubBackend[stubIntent, string]{
						docs: []Document[string]{{ID: "private-document", Content: "private-content"}},
					},
				},
			},
			observation.StageFallback,
			nil,
		},
		{
			"rescue",
			resultRescueNode[stubIntent, NoRequestMeta, string]{
				Primary: resultRetrieverNode[stubIntent, NoRequestMeta, string]{
					Backend: orchestratorFailingBackend[stubIntent, string]{},
				},
				Secondary: resultRetrieverNode[stubIntent, NoRequestMeta, string]{
					Backend: orchestratorStubBackend[stubIntent, string]{
						docs: []Document[string]{{ID: "private-document"}},
					},
				},
			},
			observation.StageRescue,
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange.
			var events []observation.Event
			session, err := observation.New(
				observation.Config{
					MaxEvents: 32,
					Observer: observation.ObserverFunc(func(_ context.Context, e observation.Event) error {
						events = append(events, e)
						return errors.New("export failed private payload")
					}),
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			ctx := observation.WithQuery(observation.WithSession(context.Background(), session), 9)
			// Act.
			result, err := tt.root.Retrieve(ctx, pipelineTestQuery("private-query"))
			// Assert: diagnostic errors do not alter dispatch/result; child stages correlate to actual parent/branch.
			if !errors.Is(err, tt.wantErr) || result.Len() != 1 {
				t.Fatalf("result %v error %v", result, err)
			}
			if len(events) != 6 || events[0].Stage != tt.stage ||
				events[5].Completion.Outcome != observation.OutcomeSuccess {
				t.Fatalf("events %+v", events)
			}
			for _, idx := range []int{1, 3} {
				if events[idx].Parent != events[0].Operation || !events[idx].Branch.Known || !events[idx].Query.Known ||
					events[idx].Query.Value != 9 {
					t.Fatalf("correlation %+v", events[idx])
				}
			}
			if events[1].Branch.Value != 1 || events[3].Branch.Value != 2 {
				t.Fatalf("branches %+v", events)
			}
			if session.Stats().Failures != 6 {
				t.Fatal(session.Stats())
			}
		})
	}
}

func TestLegacyObservedCanceledAndPartial(t *testing.T) {
	// Arrange.
	var ends []observation.Event
	session, _ := observation.New(
		observation.Config{
			MaxEvents: 16,
			Observer: observation.ObserverFunc(func(_ context.Context, e observation.Event) error {
				if e.Kind == observation.KindEnd {
					ends = append(ends, e)
				}
				return nil
			}),
		},
	)
	ctx, cancel := context.WithCancel(observation.WithSession(context.Background(), session))
	cancel()
	node := resultRetrieverNode[stubIntent, NoRequestMeta, string]{
		Backend: orchestratorStubBackend[stubIntent, string]{},
	}
	// Act.
	_, err := node.Retrieve(ctx, pipelineTestQuery("private"))
	partial := observationCompletion(ragy.ErrUnavailable, NewResultSet([]Document[string]{{ID: "private"}}, nil))
	// Assert.
	if !access.IsProtectionFailure(err) || len(ends) != 1 || ends[0].Completion.Error != observation.ErrorProtection ||
		partial.Outcome != observation.OutcomePartial {
		t.Fatalf("error %v events %+v partial %+v", err, ends, partial)
	}
}
