package graphingest_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/graphingest/materialization"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/recipe/budget"
)

type stages struct {
	failed string
	calls  []string
	cancel func()
}

func (s *stages) step(name string) error {
	s.calls = append(s.calls, name)
	if s.cancel != nil {
		s.cancel()
	}
	if s.failed == name {
		return ragy.ErrUnavailable
	}
	return nil
}

func (s *stages) Extract(
	context.Context,
	access.Binding,
	*budget.Ledger,
	[]extraction.Snippet[int],
) (extraction.Result[string, string, int], error) {
	return extraction.Result[string, string, int]{}, s.step("extraction")
}

func (s *stages) Resolve(
	context.Context,
	access.Binding,
	resolution.Extraction[string, string, int],
) (resolution.Result[string, string, int], error) {
	return resolution.Result[string, string, int]{}, s.step("resolution")
}

func (s *stages) Build(
	context.Context,
	access.Binding,
	materialization.Request,
	resolution.Result[string, string, int],
) (materialization.Result[int], error) {
	return materialization.Result[int]{}, s.step("materialization")
}

func TestPipelineRejectsMissingPorts(t *testing.T) {
	// Arrange/Act.
	_, err := graphingest.New(graphingest.Config[int, string, string, int, int]{})
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal(err)
	}
}
func TestFailedStageCannotExposePublishablePlan(t *testing.T) {
	for i, failed := range []string{"extraction", "resolution", "materialization"} {
		t.Run(failed, func(t *testing.T) {
			// Arrange.
			ports := &stages{failed: failed}
			pipeline, err := graphingest.New(
				graphingest.Config[int, string, string, int, int]{
					Extraction:      ports,
					Resolution:      ports,
					Materialization: ports,
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			result, err := pipeline.Build(
				t.Context(),
				access.Unrestricted(),
				&budget.Ledger{},
				nil,
				materialization.Request{},
			)
			// Assert: failed work never becomes a plan or dispatches later stages.
			if !errors.Is(err, ragy.ErrUnavailable) || result.Plan.Manifest.ID != "" || len(ports.calls) != i+1 {
				t.Fatalf("failed stage: %v %#v %v", err, result.Plan, ports.calls)
			}
		})
	}
}
func TestCancellationBetweenStagesStopsComposition(t *testing.T) {
	// Arrange.
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	ports := &stages{cancel: cancel}
	pipeline, err := graphingest.New(
		graphingest.Config[int, string, string, int, int]{Extraction: ports, Resolution: ports, Materialization: ports},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := pipeline.Build(ctx, access.Unrestricted(), &budget.Ledger{}, nil, materialization.Request{})
	// Assert.
	if !errors.Is(err, context.Canceled) || len(ports.calls) != 1 || result.Plan.Manifest.ID != "" {
		t.Fatalf("cancelled: %v %v", err, ports.calls)
	}
}
