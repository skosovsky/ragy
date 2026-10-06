package retrieval_test

import (
	"context"
	"errors"
	"maps"
	"sync/atomic"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type admissionIntent struct{ Accept bool }
type admissionBackend struct {
	schema filter.Schema
	calls  atomic.Int64
}

func (b *admissionBackend) Schema() filter.Schema { return b.schema }
func (*admissionBackend) ReadCapabilities() access.Capabilities {
	return access.Capabilities{ScopeProfile: true}
}

func (b *admissionBackend) AdmitRead(
	_ context.Context,
	r retrieval.Query[admissionIntent],
) (retrieval.ReadCoverage, error) {
	if !r.Intent.Accept {
		return retrieval.UnobservedReadCoverage(), access.UnsupportedCapability(ragy.ErrUnsupported)
	}
	return retrieval.PartialReadCoverage("optional-source")
}

func (b *admissionBackend) Retrieve(
	_ context.Context,
	_ retrieval.Query[admissionIntent],
) (retrieval.ResultSet[accessMeta], error) {
	b.calls.Add(1)
	return retrieval.NewResultSet(
		[]retrieval.Document[accessMeta]{
			{ID: "a", Content: "policy", Meta: accessMeta{Tenant: "a", Visibility: "public"}},
		},
		nil,
	), nil
}

func TestProjectedRequestAdmissionUsesPureProjectionAndPreservesCoverage(t *testing.T) {
	// Arrange: target needs a different host intent during preflight.
	fixture := newScopeFixture(t, allowAuthority())
	next := &admissionBackend{schema: fixture.schema}
	var payloadProjections atomic.Int64
	var admissionProjections atomic.Int64
	project := func(r retrieval.Query[struct{}]) retrieval.Query[admissionIntent] {
		return retrieval.Query[admissionIntent]{Read: r.Read, Intent: admissionIntent{Accept: true}, Options: r.Options}
	}
	backend := retrieval.ProjectedBackend[struct{}, retrieval.NoRequestMeta, admissionIntent, retrieval.NoRequestMeta, accessMeta]{
		Next: next,
		Project: func(r retrieval.Query[struct{}]) retrieval.Query[admissionIntent] {
			payloadProjections.Add(1)
			return project(r)
		},
		AdmissionProject: func(r retrieval.Query[struct{}]) retrieval.Query[admissionIntent] {
			admissionProjections.Add(1)
			return project(r)
		},
	}
	node := retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: backend}
	query := retrieval.Query[struct{}]{Read: fixture.binding, Options: retrieval.RetrieveOptions{TopK: 1}}
	// Act.
	coverage, err := retrieval.InspectRead(t.Context(), query, node)
	// Assert: payload projection is not used; custom partial coverage survives.
	if err != nil || !coverage.IsPartial() || payloadProjections.Load() != 0 || admissionProjections.Load() != 1 ||
		next.calls.Load() != 0 {
		t.Fatalf(
			"preflight: %v %v payload=%d admission=%d calls=%d",
			err,
			coverage.State(),
			payloadProjections.Load(),
			admissionProjections.Load(),
			next.calls.Load(),
		)
	}
	backend.AdmissionProject = nil
	_, err = retrieval.InspectRead(
		t.Context(),
		query,
		retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: backend},
	)
	if !access.IsUnsupportedCapability(err) || payloadProjections.Load() != 0 {
		t.Fatal("payload projector invoked to fake admission", err)
	}
	backend.AdmissionProject = func(r retrieval.Query[struct{}]) retrieval.Query[admissionIntent] {
		out := project(r)
		out.Intent.Accept = false
		return out
	}
	_, err = retrieval.InspectRead(
		t.Context(),
		query,
		retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: backend},
	)
	if !errors.Is(err, ragy.ErrUnsupported) || next.calls.Load() != 0 {
		t.Fatal("denied target dispatch", err)
	}
}

type branchState struct{ Values map[string]int }

type ownedBranch struct{ index int }

func (b ownedBranch) Execute(
	_ context.Context,
	_ retrieval.Query[struct{}],
	shared branchState,
) (retrieval.RetrievalResult[accessMeta, branchState], error) {
	owned := branchState{Values: make(map[string]int, len(shared.Values)+1)}
	maps.Copy(owned.Values, shared.Values)
	owned.Values["branch"] = b.index
	return retrieval.RetrievalResult[accessMeta, branchState]{
		ResultSet: retrieval.NewResultSet[accessMeta](nil, nil),
		Executed:  owned,
	}, nil
}

func TestParallelBranchesConstructOwnedBYOTState(t *testing.T) {
	// Arrange: reference-containing host execution input is immutable.
	input := branchState{Values: map[string]int{"seed": 7}}
	var nodes []retrieval.ExecutionNode[struct{}, accessMeta, branchState]
	for i := range 8 {
		nodes = append(nodes, ownedBranch{index: i})
	}
	aggregate := retrieval.AggregateNode[struct{}, accessMeta, branchState]{Nodes: nodes, Concurrency: 8}
	// Act.
	_, err := aggregate.Execute(
		t.Context(),
		retrieval.Query[struct{}]{Read: retrieval.UnrestrictedRead(), Options: retrieval.RetrieveOptions{TopK: 1}},
		input,
	)
	// Assert: branches have owned state and the caller input is unchanged under -race.
	if err != nil || len(input.Values) != 1 || input.Values["seed"] != 7 {
		t.Fatalf("shared state mutated: %v %#v", err, input)
	}
}
