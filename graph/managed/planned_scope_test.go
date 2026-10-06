package managed_test

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/retrieval"
)

func TestGraphBackendUnsupportedPlanBeforePayloadProjection(t *testing.T) {
	// Arrange: actual published graph and an unmapped host planner predicate.
	f := newFixture(t)
	input := payload("policy", "r1")
	f.ingest(t, plan("policy", "", input, "p1"), input)
	backend, err := managed.NewBackend(managed.BackendConfig[metadata]{Adapter: f.adapter, MaxNodes: 50, MaxEdges: 100})
	if err != nil {
		t.Fatal(err)
	}
	fields := filter.NewSchema()
	field, err := fields.String("foreign_plan_field")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := filter.Eq(builder, field, "value").Build()
	if err != nil {
		t.Fatal(err)
	}
	request := backendQuery(f.pin(t)).WithPlan(retrieval.PlannedQuery[struct{}]{Filters: condition})
	f.calls = nil
	// Act.
	coverage, admitErr := backend.AdmitRead(t.Context(), request)
	result, readErr := backend.Retrieve(t.Context(), request)
	// Assert: negotiation is unsupported and neither path materializes metadata.
	if !errors.Is(admitErr, ragy.ErrUnsupported) || !access.IsUnsupportedCapability(admitErr) ||
		coverage.State() == retrieval.CoverageComplete {
		t.Fatal("unsupported plan admitted", admitErr)
	}
	if !errors.Is(readErr, ragy.ErrUnsupported) || !access.IsProtectionFailure(readErr) || result.Len() != 0 ||
		len(f.calls) != 0 {
		t.Fatal("unsupported graph plan projected payload", readErr, f.calls)
	}
}
