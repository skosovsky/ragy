package query_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

type candidateStub struct {
	schema filter.Schema
	caps   access.Capabilities
	docs   []retrieval.Document[int]
	calls  int
	mutate bool
}

func (c *candidateStub) Schema() filter.Schema                 { return c.schema }
func (c *candidateStub) ReadCapabilities() access.Capabilities { return c.caps }

func (c *candidateStub) Retrieve(
	_ context.Context,
	req retrieval.Query[tensorquery.Intent],
) (retrieval.ResultSet[int], error) {
	c.calls++
	if c.mutate {
		req.Intent.Embedding.Tokens[0][0] = -1
	}
	return retrieval.NewResultSet(c.docs, nil), nil
}

type targetStub struct {
	schema   filter.Schema
	caps     access.Capabilities
	calls    int
	observed float32
}

func (t *targetStub) Schema() filter.Schema                 { return t.schema }
func (t *targetStub) ReadCapabilities() access.Capabilities { return t.caps }
func (*targetStub) QueryCapabilities() tensor.QueryCapabilities {
	return tensor.QueryCapabilities{
		Space:                 fixtureSpace(),
		ScoreSemantics:        tensor.MaxSimSemantics,
		CandidateLimit:        2,
		ExactWithinCandidates: true,
		Exhaustive:            false,
	}
}

func (t *targetStub) Query(
	_ context.Context,
	req retrieval.Query[tensorquery.Intent],
) (tensorquery.Result[int], error) {
	t.calls++
	t.observed = req.Intent.Embedding.Tokens[0][0]
	return tensorquery.Result[int]{
		Documents: retrieval.NewResultSet[int](nil, nil),
		Evidence:  tensor.RerankResult{Ranking: nil, CandidateIDs: nil, CandidateBudget: req.Intent.CandidateBudget},
	}, nil
}
func fixtureSpace() tensor.Space {
	return tensor.Space{Metric: "normalized-dot",
		Model:         "fixture",
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
}
func fixture(t *testing.T) (*candidateStub, *targetStub, retrieval.Query[tensorquery.Intent]) {
	t.Helper()
	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatal(err)
	}
	publication, err := access.PinPublication("pin", nil)
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	caps := access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true}
	candidate := &candidateStub{
		schema: schema,
		caps:   caps,
		docs:   []retrieval.Document[int]{{ID: "candidate", Meta: 1}},
		calls:  0,
		mutate: false,
	}
	target := &targetStub{schema: schema, caps: caps, calls: 0, observed: 0}
	request := retrieval.Query[tensorquery.Intent]{
		Read: read,
		Intent: tensorquery.Intent{
			Embedding:       tensor.Embedding{Space: fixtureSpace(), Tokens: tensor.Tensor{{1, 0}}},
			Candidates:      nil,
			CandidateBudget: 2,
		},
		Options: retrieval.RetrieveOptions{TopK: 1},
	}
	return candidate, target, request
}
func reference(retrieval.Document[int]) (source.Reference, error) {
	return source.Reference{
		Namespace:         "n",
		Source:            "s",
		Revision:          "r1",
		Transformation:    "embedding",
		AccessFingerprint: "acl",
		Artifact:          "one",
		Representation:    "token-matrix",
	}, nil
}

func search(
	t *testing.T,
	candidate *candidateStub,
	target *targetStub,
	clone func(int) (int, error),
) *tensorquery.Search[int, int] {
	t.Helper()
	result, err := tensorquery.New(
		tensorquery.Config[int, int]{
			Candidates:         candidate,
			Target:             target,
			CloneCandidateMeta: clone,
			Reference:          reference,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return result
}
func clone(value int) (int, error) { return value, nil }

func TestSearchAdmitsBothLeavesBeforeCandidateDispatch(t *testing.T) {
	// Arrange: candidate can enforce the pinned read, target cannot.
	candidate, target, request := fixture(t)
	target.caps.PinnedPublication = false
	pipeline := search(t, candidate, target, clone)
	// Act.
	result, err := pipeline.Query(context.Background(), request)
	// Assert.
	if !errors.Is(err, ragy.ErrUnsupported) || candidate.calls != 0 || target.calls != 0 ||
		result.Documents.Len() != 0 {
		t.Fatal("unsupported target dispatched candidate I/O", err)
	}
}
func TestSearchRejectsOverflowWithoutSilentTruncation(t *testing.T) {
	// Arrange: faulty candidate backend exceeds the declared budget.
	candidate, target, request := fixture(t)
	candidate.docs = append(candidate.docs, retrieval.Document[int]{ID: "two"}, retrieval.Document[int]{ID: "three"})
	pipeline := search(t, candidate, target, clone)
	// Act.
	_, err := pipeline.Query(context.Background(), request)
	// Assert.
	if !errors.Is(err, ragy.ErrProtocol) || candidate.calls != 1 || target.calls != 0 {
		t.Fatal("overflow dispatched tensor scoring", err)
	}
}
func TestSearchOwnsQueryAcrossCandidateCallback(t *testing.T) {
	// Arrange: adversarial backend mutates its private request matrix.
	candidate, target, request := fixture(t)
	candidate.mutate = true
	pipeline := search(t, candidate, target, clone)
	// Act.
	_, err := pipeline.Query(context.Background(), request)
	// Assert: original query and tensor request retain the captured embedding.
	if err != nil || target.calls != 1 || target.observed != 1 || request.Intent.Embedding.Tokens[0][0] != 1 {
		t.Fatal("candidate callback changed tensor query", err)
	}
}
func TestSearchCancellationDuringProjectionStopsTensorDispatch(t *testing.T) {
	// Arrange.
	candidate, target, request := fixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pipeline := search(t, candidate, target, func(value int) (int, error) { cancel(); return value, nil })
	// Act.
	result, err := pipeline.Query(ctx, request)
	// Assert.
	if !errors.Is(err, context.Canceled) || candidate.calls != 1 || target.calls != 0 || result.Documents.Len() != 0 {
		t.Fatal("canceled projection dispatched tensor query", err)
	}
}
func TestSearchRejectsInvalidEmbeddingBeforeCandidateIO(t *testing.T) {
	// Arrange.
	candidate, target, request := fixture(t)
	request.Intent.Embedding.Tokens = tensor.Tensor{{2, 0}}
	pipeline := search(t, candidate, target, clone)
	// Act.
	_, err := pipeline.Query(context.Background(), request)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) || candidate.calls != 0 || target.calls != 0 {
		t.Fatal("invalid embedding dispatched candidate query", err)
	}
}

func TestSearchInvalidConstructionAndZeroValueFailClosed(t *testing.T) {
	// Arrange: typed-nil interface and a zero Search are both invalid configurations.
	candidate, _, request := fixture(t)
	var target *targetStub
	_, err := tensorquery.New(
		tensorquery.Config[int, int]{
			Candidates:         candidate,
			Target:             target,
			CloneCandidateMeta: clone,
			Reference:          reference,
		},
	)
	// Act/Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("typed-nil target accepted", err)
	}
	var zero tensorquery.Search[int, int]
	result, err := zero.Query(context.Background(), request)
	if !errors.Is(err, ragy.ErrInvalidArgument) || result.Documents.Len() != 0 || candidate.calls != 0 {
		t.Fatal("zero Search did not fail closed", err)
	}
	if _, err = zero.AdmitRead(context.Background(), request); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("zero admission accepted", err)
	}
}
