package query_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/retrieval"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

type admittedCandidates struct {
	*candidateStub

	admitted int
	err      error
}

func (c *admittedCandidates) AdmitPublication(access.Publication) error { c.admitted++; return c.err }

type admittedTarget struct {
	*targetStub

	admitted int
	err      error
}

func (t *admittedTarget) AdmitPublication(access.Publication) error { t.admitted++; return t.err }

func TestPartialPublicationNegotiatesBothSearchTargets(t *testing.T) {
	for _, denied := range []string{"", "candidates", "tensor"} {
		t.Run(denied, func(t *testing.T) {
			checkPartialPublicationTargets(t, denied)
		})
	}
}

type requestAdmittedCandidates struct {
	*candidateStub

	admissionErr error
	coverage     retrieval.ReadCoverage
}

func (c *requestAdmittedCandidates) AdmitRead(
	context.Context,
	retrieval.Query[tensorquery.Intent],
) (retrieval.ReadCoverage, error) {
	return c.coverage, c.admissionErr
}

type requestAdmittedTarget struct {
	*targetStub

	admissionErr error
	coverage     retrieval.ReadCoverage
}

func (t *requestAdmittedTarget) AdmitRead(
	context.Context,
	retrieval.Query[tensorquery.Intent],
) (retrieval.ReadCoverage, error) {
	return t.coverage, t.admissionErr
}

func TestSearchPreservesChildRequestAdmission(t *testing.T) {
	for _, denied := range []string{"", "candidates", "tensor"} {
		t.Run(denied, func(t *testing.T) {
			checkChildRequestAdmission(t, denied)
		})
	}
}

func checkPartialPublicationTargets(t *testing.T, denied string) {
	t.Helper()
	// Arrange: both leaves explicitly negotiate the same partial publication.
	candidate, target, request := fixture(t)
	candidates := &admittedCandidates{candidateStub: candidate}
	scoring := &admittedTarget{targetStub: target}
	publication, err := access.PinPartialPublication("partial", nil, []string{"optional"})
	if err != nil {
		t.Fatal(err)
	}
	request.Read, err = access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	if denied == "candidates" {
		candidates.err = access.UnsupportedCapability(nil)
	}
	if denied == "tensor" {
		scoring.err = access.UnsupportedCapability(nil)
	}
	search, err := tensorquery.New(
		tensorquery.Config[int, int]{
			Candidates:         candidates,
			Target:             scoring,
			CloneCandidateMeta: clone,
			Reference:          reference,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act: use public pipeline preflight, without payload dispatch.
	coverage, err := retrieval.InspectRead(
		t.Context(),
		request,
		retrieval.BackendNode[tensorquery.Intent, int, retrieval.NoExecutionMeta]{Backend: search},
	)
	// Assert: partial coverage survives; either leaf can veto before any I/O.
	if denied == "" &&
		(err != nil || !coverage.IsPartial() || candidates.admitted == 0 || scoring.admitted == 0) {
		t.Fatalf("admission: %v %v", coverage.State(), err)
	}
	if denied != "" && !access.IsUnsupportedCapability(err) {
		t.Fatalf("missing veto: %v", err)
	}
	if candidate.calls != 0 || target.calls != 0 {
		t.Fatal("admission dispatched payload")
	}
}

func checkChildRequestAdmission(t *testing.T, denied string) {
	t.Helper()
	// Arrange: candidate reports partial coverage; either typed leaf can veto.
	candidate, target, request := fixture(t)
	partial, err := retrieval.PartialReadCoverage("candidate-source")
	if err != nil {
		t.Fatal(err)
	}
	c := &requestAdmittedCandidates{candidateStub: candidate, coverage: partial}
	scoring := &requestAdmittedTarget{targetStub: target, coverage: retrieval.CompleteReadCoverage()}
	if denied == "candidates" {
		c.admissionErr = access.UnsupportedCapability(nil)
	}
	if denied == "tensor" {
		scoring.admissionErr = access.UnsupportedCapability(nil)
	}
	search, err := tensorquery.New(
		tensorquery.Config[int, int]{
			Candidates:         c,
			Target:             scoring,
			CloneCandidateMeta: clone,
			Reference:          reference,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act: public node preflight, followed by raw Search dispatch for a veto.
	coverage, err := retrieval.InspectRead(
		t.Context(),
		request,
		retrieval.BackendNode[tensorquery.Intent, int, retrieval.NoExecutionMeta]{Backend: search},
	)
	// Assert: leaf denial/partial survives before candidate materialization.
	if denied == "" && (err != nil || !coverage.IsPartial() || len(coverage.SkippedBranches()) != 1) {
		t.Fatalf("lost partial coverage: %v %v", coverage.State(), err)
	}
	if denied != "" {
		if !access.IsUnsupportedCapability(err) {
			t.Fatalf("missing veto: %v", err)
		}
		_, err = search.Query(t.Context(), request)
		if !errors.Is(err, ragy.ErrUnsupported) {
			t.Fatalf("raw dispatch missed veto: %v", err)
		}
	}
	if candidate.calls != 0 || target.calls != 0 {
		t.Fatal("preflight dispatched payload")
	}
}

type budgetAdmittedCandidates struct {
	*candidateStub

	admittedTopK       int
	admittedFetchLimit int
}

func (c *budgetAdmittedCandidates) AdmitRead(
	_ context.Context,
	r retrieval.Query[tensorquery.Intent],
) (retrieval.ReadCoverage, error) {
	c.admittedTopK, c.admittedFetchLimit = r.Options.TopK, r.Options.FetchLimit
	if r.Options.TopK > 1 || r.Options.FetchLimit > 1 {
		return retrieval.UnobservedReadCoverage(), access.UnsupportedCapability(ragy.ErrUnsupported)
	}
	return retrieval.CompleteReadCoverage(), nil
}
func TestSearchAdmitsActualCandidateBudgetBeforeDispatch(t *testing.T) {
	// Arrange: host denies the actual candidate budget, while output TopK is allowed.
	candidate, target, request := fixture(t)
	bounded := &budgetAdmittedCandidates{candidateStub: candidate}
	search, err := tensorquery.New(
		tensorquery.Config[int, int]{
			Candidates:         bounded,
			Target:             target,
			CloneCandidateMeta: clone,
			Reference:          reference,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, err = search.Query(t.Context(), request)
	// Assert: dispatch cannot expand a previously admitted request budget.
	if !errors.Is(err, ragy.ErrUnsupported) || bounded.admittedTopK != request.Intent.CandidateBudget ||
		bounded.admittedFetchLimit != request.Intent.CandidateBudget ||
		candidate.calls != 0 ||
		target.calls != 0 {
		t.Fatalf(
			"budget admission: %v topK=%d fetch=%d payload=%d",
			err,
			bounded.admittedTopK,
			bounded.admittedFetchLimit,
			candidate.calls,
		)
	}
}
