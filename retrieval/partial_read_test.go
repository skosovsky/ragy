package retrieval_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

func partialFixtureNode(
	backend retrieval.Backend[struct{}, accessMeta],
) retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta] {
	return retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: backend}
}
func partialRequest(read access.Binding) retrieval.Query[struct{}] {
	return retrieval.Query[struct{}]{Read: read, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 10}}
}

func optionalScopeNode(
	child retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta],
) retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta] {
	return retrieval.PartialReadNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
		Child:    child,
		Name:     "optional-graph",
		Resolver: nil,
	}
}

func TestPartialScopeSkipsUnsupportedBeforeIOAndPreservesNestedCoverage(t *testing.T) {
	for _, kind := range []string{"aggregate", "nested optional", "unused fallback"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange: only the explicitly optional branch lacks mandatory scope support.
			fixture := newScopeFixture(t, allowAuthority())
			allowed := &admittedScopeBackend{fixture: fixture}
			denied := &incompatibleReadBackend{}
			optional := optionalScopeNode(partialFixtureNode(denied))
			var root retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta]
			aggregate := retrieval.AggregateNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
				Nodes: []retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
					partialFixtureNode(allowed),
					optional,
				},
				Concurrency: 2,
			}
			switch kind {
			case "aggregate":
				root = aggregate
			case "nested optional":
				root = optionalScopeNode(aggregate)
			case "unused fallback":
				root = retrieval.FallbackNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
					Primary:   partialFixtureNode(allowed),
					Secondary: optional,
				}
			}
			pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, accessMeta, retrieval.NoExecutionMeta]().WithRoot(root).
				Build()
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			result, err := pipeline.Execute(context.Background(), partialRequest(fixture.binding))
			// Assert: useful scoped payload survives; unsupported I/O does not occur.
			if err != nil || result.Len() != 1 || result.Documents()[0].ID != "a-public" ||
				!result.Coverage.IsPartial() ||
				allowed.calls.Load() != 1 ||
				denied.calls != 0 {
				t.Fatalf(
					"partial scope failed: %v, %v, coverage=%v, calls=%d/%d",
					result.Documents(),
					err,
					result.Coverage.State(),
					allowed.calls.Load(),
					denied.calls,
				)
			}
			assertPartialCoverage(t, result.Coverage)
		})
	}
}

func TestPartialCannotSkipAuthorityDenialOrCancellation(t *testing.T) {
	for _, failure := range []string{"authority unsupported", "canceled"} {
		t.Run(failure, func(t *testing.T) {
			// Arrange: even a host error classified unsupported is an authority failure.
			fixture := newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
				return access.UnsupportedCapability(ragy.ErrUnsupported)
			}))
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if failure == "canceled" {
				fixture = newScopeFixture(t, allowAuthority())
				cancel()
			}
			backend := &incompatibleReadBackend{}
			root := optionalScopeNode(partialFixtureNode(backend))
			// Act.
			result, err := root.Execute(ctx, partialRequest(fixture.binding), retrieval.NoExecutionMeta{})
			// Assert: denial is not transformed into an empty successful partial result.
			if err == nil || !access.IsProtectionFailure(err) || access.IsUnsupportedCapability(err) ||
				!result.IsEmpty() ||
				result.Coverage.IsPartial() ||
				backend.calls != 0 {
				t.Fatalf(
					"authority/cancellation skipped: %v, state=%v, calls=%d",
					err,
					result.Coverage.State(),
					backend.calls,
				)
			}
		})
	}
}

type partialRuntimeBackend struct {
	fixture scopeFixture
	calls   int
}

func (b *partialRuntimeBackend) Schema() filter.Schema { return b.fixture.schema }
func (*partialRuntimeBackend) ReadCapabilities() access.Capabilities {
	return access.Capabilities{RequirePinnedPublication: false, ScopeProfile: true}
}

func (b *partialRuntimeBackend) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[accessMeta], error) {
	b.calls++
	rs, err := b.fixture.index.Retrieve(ctx, req)
	if err != nil {
		return rs, err
	}
	return rs, access.UnsupportedCapability(ragy.ErrUnsupported)
}
func TestPartialNeverConvertsRuntimeFailureToCapabilitySkip(t *testing.T) {
	// Arrange: target admission succeeds, then an I/O path returns unsupported.
	fixture := newScopeFixture(t, allowAuthority())
	backend := &partialRuntimeBackend{fixture: fixture}
	root := optionalScopeNode(partialFixtureNode(backend))
	// Act.
	result, err := root.Execute(context.Background(), partialRequest(fixture.binding), retrieval.NoExecutionMeta{})
	// Assert: no success/partial-skip conversion after target I/O.
	if err == nil || !access.IsProtectionFailure(err) || !result.IsEmpty() || result.Coverage.IsPartial() ||
		backend.calls != 1 {
		t.Fatalf("runtime unsupported skipped: %v, state=%v, calls=%d", err, result.Coverage.State(), backend.calls)
	}
}

type joinedAdmissionNode struct{ calls int }

func (*joinedAdmissionNode) AdmitRead(context.Context, retrieval.Query[struct{}]) (retrieval.ReadCoverage, error) {
	return retrieval.UnobservedReadCoverage(), errors.Join(
		access.UnsupportedCapability(ragy.ErrUnsupported),
		context.Canceled,
	)
}

func (n *joinedAdmissionNode) Execute(
	context.Context,
	retrieval.Query[struct{}],
	retrieval.NoExecutionMeta,
) (retrieval.RetrievalResult[accessMeta, retrieval.NoExecutionMeta], error) {
	n.calls++
	return retrieval.RetrievalResult[accessMeta, retrieval.NoExecutionMeta]{}, nil
}
func TestPartialDoesNotSkipJoinedAdmissionFailures(t *testing.T) {
	// Arrange: unsupported is joined with a fatal failure; it is not a pure capability miss.
	fixture := newScopeFixture(t, allowAuthority())
	child := &joinedAdmissionNode{}
	root := optionalScopeNode(child)
	// Act.
	result, err := root.Execute(context.Background(), partialRequest(fixture.binding), retrieval.NoExecutionMeta{})
	// Assert.
	if err == nil || access.IsUnsupportedCapability(err) || result.Coverage.IsPartial() || child.calls != 0 {
		t.Fatalf("joined admission failure skipped: %v, state=%v, calls=%d", err, result.Coverage.State(), child.calls)
	}
}

func TestCoverageWireRoundtripRejectsUnknownSchemaAndMalformedReports(t *testing.T) {
	// Arrange.
	data, err := os.ReadFile("../docs/task12/fixtures/read_coverage.json")
	if err != nil {
		t.Fatal(err)
	}
	var decoded retrieval.ReadCoverage
	// Act.
	if decodeErr := json.Unmarshal(data, &decoded); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	// Assert.
	if !decoded.IsPartial() || decoded.SkippedBranches()[0] != "optional-graph" {
		t.Fatal("coverage roundtrip lost state")
	}
	roundtrip, err := json.Marshal(decoded)
	if err != nil {
		t.Fatal(err)
	}
	var before, after any
	if err := json.Unmarshal(data, &before); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(roundtrip, &after); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("coverage wire changed: %s", roundtrip)
	}
	for _, invalid := range []string{
		`{"schema":"unknown","state":"complete","skipped_branches":[]}`,
		`{"schema":"ragy.read-coverage/admission","state":"partial","skipped_branches":[]}`,
		`{"schema":"ragy.read-coverage/admission","state":"complete","skipped_branches":["optional-graph"]}`,
		`{"schema":"ragy.read-coverage/admission","state":"partial","skipped_branches":["a","a"]}`,
		`{"schema":"ragy.read-coverage/admission","state":"","skipped_branches":[]}`,
		`{"schema":"ragy.read-coverage/admission","state":"complete"}`,
		`{"schema":"ragy.read-coverage/admission","state":"complete","skipped_branches":[],"private_ids":["secret"]}`,
	} {
		if err := json.Unmarshal([]byte(invalid), &decoded); err == nil {
			t.Fatalf("invalid coverage accepted: %s", invalid)
		}
		if !decoded.IsPartial() || decoded.SkippedBranches()[0] != "optional-graph" {
			t.Fatal("invalid decode mutated previous coverage")
		}
	}
}

func assertPartialCoverage(t *testing.T, coverage retrieval.ReadCoverage) {
	t.Helper()
	branches := coverage.SkippedBranches()
	if len(branches) != 1 || branches[0] != "optional-graph" {
		t.Fatalf("coverage lost/doubled: %v", branches)
	}
	branches[0] = "mutated"
	if coverage.SkippedBranches()[0] != "optional-graph" {
		t.Fatal("coverage aliases caller slices")
	}
	serialized, err := json.Marshal(coverage)
	if err != nil {
		t.Fatal(err)
	}
	for _, hidden := range []string{"a-private", "b-public", "tenant", "visibility", "document_count"} {
		if strings.Contains(string(serialized), hidden) {
			t.Fatalf("hidden payload in coverage: %s", serialized)
		}
	}
}

func TestPartialRejectsTypedNilTargetsBeforeDispatch(t *testing.T) {
	// Arrange: interfaces containing nil pointers are not supported optional targets.
	fixture := newScopeFixture(t, allowAuthority())
	var backend *partialRuntimeBackend
	var child *joinedAdmissionNode
	for _, node := range []retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{partialFixtureNode(backend), child} {
		root := optionalScopeNode(node)
		// Act.
		result, err := root.Execute(context.Background(), partialRequest(fixture.binding), retrieval.NoExecutionMeta{})
		// Assert: no nil receiver call and no false successful partial skip.
		if !errors.Is(err, ragy.ErrInvalidArgument) || access.IsUnsupportedCapability(err) || !result.IsEmpty() ||
			result.Coverage.IsPartial() {
			t.Fatalf("typed nil target admitted: %v, state=%v", err, result.Coverage.State())
		}
	}
}
