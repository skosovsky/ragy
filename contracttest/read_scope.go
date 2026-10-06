package contracttest

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

// ScopedReadFixture is a fresh host-owned adapter fixture for one scenario. Use a
// corpus with allowed and forbidden payloads, a scalar Eq/In/And binding, and
// contradictory/unsupported query conditions mapped through the host schema.
// IOCount measures payload target I/O, not entry into the adapter method. PayloadIDs
// observes payload materialization before output projection/filtering. Hooks update
// the existing authority/clock; they must not replace the immutable binding.
type ScopedReadFixture[TIntent, TRequestMeta, TMeta any] struct {
	Backend             retrieval.RequestBackend[TIntent, TRequestMeta, TMeta]
	Request             retrieval.Request[TIntent, TRequestMeta]
	Conflict            filter.Condition
	Unsupported         filter.Condition
	ExpectedIDs         []string
	ForbiddenPayloadIDs []string
	IOCount             func() int
	PayloadIDs          func() []string
	Revoke              func()
	Expire              func()
	RevokeDuringIO      func()
	ObservedDeadline    func() (time.Time, bool)
}

// ScopedReadFactory constructs an independent fixture for each scenario.
type ScopedReadFactory[TIntent, TRequestMeta, TMeta any] func(*testing.T) ScopedReadFixture[TIntent, TRequestMeta, TMeta]

// ScopedReadCase selects a required direct-backend enforcement scenario.
type ScopedReadCase string

const (
	ReadAllowed         ScopedReadCase = "allowed"
	ReadConflict        ScopedReadCase = "conflicting-query"
	ReadUnsupported     ScopedReadCase = "unsupported-query"
	ReadMissingBinding  ScopedReadCase = "missing-binding"
	ReadRevoked         ScopedReadCase = "revoked-before-io"
	ReadExpired         ScopedReadCase = "expired-before-io"
	ReadCanceled        ScopedReadCase = "canceled-before-io"
	ReadRevokedDuringIO ScopedReadCase = "revoked-during-io"
	ReadDeadline        ScopedReadCase = "deadline-propagation"
)

// ReadViolation contains stable QA codes, never forbidden payload IDs/counts or
// raw adapter errors. It is test output, not production authorization evidence.
type ReadViolation struct {
	Case ScopedReadCase
	Code string
}

const readFixtureDeadline = 5 * time.Second

// CheckScopedReadCase checks admission before dispatch, then independently invokes
// the raw adapter to detect missing leaf gates and post-filter-only implementations.
// Incompatible declarations reject before payload I/O. Freshness/error scenarios
// must be implemented by the adapter itself; pipeline wrappers cannot hide defects.
func CheckScopedReadCase[TIntent, TRequestMeta, TMeta any](
	ctx context.Context,
	fixture ScopedReadFixture[TIntent, TRequestMeta, TMeta],
	scenario ScopedReadCase,
) []ReadViolation {
	if !validReadFixture(fixture) {
		return []ReadViolation{{Case: scenario, Code: "invalid-fixture"}}
	}
	before := fixture.IOCount()
	node := retrieval.RequestBackendNode[TIntent, TRequestMeta, TMeta, retrieval.NoExecutionMeta]{
		Backend:  fixture.Backend,
		Resolver: nil,
		Name:     "conformance",
	}
	coverage, admitErr := retrieval.InspectRead(ctx, fixture.Request, node)
	if fixture.IOCount() != before {
		return []ReadViolation{{Case: scenario, Code: "admission-performed-payload-io"}}
	}
	if admitErr != nil || coverage.State() != retrieval.CoverageComplete {
		return []ReadViolation{{Case: scenario, Code: "unsupported-admission"}}
	}
	setup := setupReadCase(ctx, fixture, scenario)
	defer setup.cancel()
	if setup.invalid {
		return []ReadViolation{{Case: scenario, Code: "invalid-scenario"}}
	}
	// Act: call the adapter directly, without ragy's protective execution wrappers.
	result, err := fixture.Backend.Retrieve(setup.ctx, setup.request)
	// Assert: actual pre-projection payload observation is distinct from final IDs.
	violations := checkReadOutcome(fixture, scenario, result, err, setup.expectedErr, setup.noIO, before)
	if scenario == ReadDeadline {
		observed, ok := fixture.ObservedDeadline()
		if !ok || !observed.Equal(setup.deadline) {
			violations = append(violations, ReadViolation{Case: scenario, Code: "deadline-not-propagated"})
		}
	}
	return violations
}

func validReadFixture[TIntent, TRequestMeta, TMeta any](fixture ScopedReadFixture[TIntent, TRequestMeta, TMeta]) bool {
	if fixture.Backend == nil || !fixture.Request.Read.IsScoped() {
		return false
	}
	if len(fixture.ExpectedIDs) == 0 || len(fixture.ForbiddenPayloadIDs) == 0 {
		return false
	}
	if fixture.IOCount == nil || fixture.PayloadIDs == nil {
		return false
	}
	if fixture.Revoke == nil || fixture.Expire == nil || fixture.RevokeDuringIO == nil {
		return false
	}
	if fixture.ObservedDeadline == nil {
		return false
	}
	return !filter.IsEmpty(fixture.Conflict.IR()) && !filter.IsEmpty(fixture.Unsupported.IR())
}

type readCaseSetup[TIntent, TRequestMeta any] struct {
	request     retrieval.Request[TIntent, TRequestMeta]
	ctx         context.Context
	cancel      context.CancelFunc
	expectedErr error
	noIO        bool
	deadline    time.Time
	invalid     bool
}

func setupReadCase[TIntent, TRequestMeta, TMeta any](
	ctx context.Context,
	fixture ScopedReadFixture[TIntent, TRequestMeta, TMeta],
	scenario ScopedReadCase,
) readCaseSetup[TIntent, TRequestMeta] {
	request := retrieval.CopyRequestOptions(fixture.Request)
	callCtx, cancel := context.WithCancel(ctx)
	var expectedErr error
	noIO := false
	deadline := time.Time{}
	invalid := false
	switch scenario {
	case ReadAllowed:
		request.Options.Filters = filter.Condition{}
	case ReadConflict:
		request.Options.Filters = fixture.Conflict
	case ReadUnsupported:
		request.Options.Filters = fixture.Unsupported
		expectedErr, noIO = ragy.ErrUnsupported, true
	case ReadMissingBinding:
		request.Read = access.Binding{}
		expectedErr, noIO = ragy.ErrInvalidArgument, true
	case ReadRevoked:
		fixture.Revoke()
		expectedErr, noIO = ragy.ErrUnavailable, true
	case ReadExpired:
		fixture.Expire()
		expectedErr, noIO = ragy.ErrUnavailable, true
	case ReadCanceled:
		cancel()
		expectedErr, noIO = context.Canceled, true
	case ReadRevokedDuringIO:
		fixture.RevokeDuringIO()
		expectedErr = ragy.ErrUnavailable
	case ReadDeadline:
		cancel()
		deadline = time.Now().Add(readFixtureDeadline)
		if existing, present := ctx.Deadline(); present && existing.Before(deadline) {
			deadline = existing
		}
		callCtx, cancel = context.WithDeadline(ctx, deadline)
	default:
		invalid = true
	}
	return readCaseSetup[TIntent, TRequestMeta]{
		request:     request,
		ctx:         callCtx,
		cancel:      cancel,
		expectedErr: expectedErr,
		noIO:        noIO,
		deadline:    deadline,
		invalid:     invalid,
	}
}

func checkReadOutcome[TIntent, TRequestMeta, TMeta any](
	fixture ScopedReadFixture[TIntent, TRequestMeta, TMeta],
	scenario ScopedReadCase,
	result retrieval.ResultSet[TMeta],
	err, expectedErr error,
	noIO bool,
	before int,
) []ReadViolation {
	var violations []ReadViolation
	add := func(code string) { violations = append(violations, ReadViolation{Case: scenario, Code: code}) }
	if noIO && fixture.IOCount() != before {
		add("denied-payload-io")
	}
	if scenario == ReadRevokedDuringIO && fixture.IOCount() <= before {
		add("missing-mid-io-injection")
	}
	if materializedForbidden(fixture.PayloadIDs(), fixture.ForbiddenPayloadIDs) {
		add("forbidden-payload-materialized")
	}
	if expectedErr != nil {
		if !errors.Is(err, expectedErr) || !access.IsProtectionFailure(err) {
			add("missing-protected-failure")
		}
		if result != nil && !result.IsEmpty() {
			add("denied-payload-delivered")
		}
		return violations
	}
	if err != nil {
		add("unexpected-error")
	}
	if result == nil {
		add("nil-result")
		return violations
	}
	want := fixture.ExpectedIDs
	if scenario == ReadConflict {
		want = nil
	}
	ids := make([]string, 0, result.Len())
	for _, doc := range result.Documents() {
		ids = append(ids, doc.ID)
		if validateErr := retrieval.ValidateDocument(doc); validateErr != nil {
			add("invalid-document")
		}
	}
	if !sameReadIDs(ids, want) {
		add("unexpected-result-ids")
	}
	return violations
}

func sameReadIDs(a, b []string) bool {
	a = append([]string(nil), a...)
	b = append([]string(nil), b...)
	slices.Sort(a)
	slices.Sort(b)
	return slices.Equal(a, b)
}

// RunScopedReadSuite runs all required direct leaf scenarios using fresh fixtures.
// Adapter hosts must supply real I/O boundary instrumentation, not observations
// reconstructed from the already filtered output. Certification is fixture-scoped.
func RunScopedReadSuite[TIntent, TRequestMeta, TMeta any](
	t *testing.T,
	factory ScopedReadFactory[TIntent, TRequestMeta, TMeta],
) {
	t.Helper()
	for _, scenario := range []ScopedReadCase{ReadAllowed, ReadConflict, ReadUnsupported, ReadMissingBinding, ReadRevoked, ReadExpired, ReadCanceled, ReadRevokedDuringIO, ReadDeadline} {
		t.Run(string(scenario), func(t *testing.T) {
			// Arrange.
			fixture := factory(t)
			// Act.
			violations := CheckScopedReadCase(t.Context(), fixture, scenario)
			// Assert.
			if len(violations) > 0 {
				t.Fatalf("scoped read conformance: %+v", violations)
			}
		})
	}
}

func materializedForbidden(observed, forbidden []string) bool {
	for _, id := range observed {
		if slices.Contains(forbidden, id) {
			return true
		}
	}
	return false
}
