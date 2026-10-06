package recipe

import (
	"context"
	"errors"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/internal/nilvalue"
	"github.com/skosovsky/ragy/observation"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

type Recipe[TIntent, TRequestMeta, TMeta any] struct {
	config Config[TIntent, TRequestMeta, TMeta]
}

const maxSubquestions = 3

func New[TIntent, TRequestMeta, TMeta any](
	config Config[TIntent, TRequestMeta, TMeta],
) (*Recipe[TIntent, TRequestMeta, TMeta], error) {
	if !validConfig(config) {
		return nil, ragy.ErrInvalidArgument
	}

	maxQueries := queryLimit(config.Strategy)
	if config.MaxQueries <= 0 || config.MaxQueries > maxQueries {
		return nil, ragy.ErrInvalidArgument
	}
	if config.Artifact != nil {
		artifact := *config.Artifact
		artifact.Diagnostics = slices.Clone(artifact.Diagnostics)
		config.Artifact = &artifact
	}
	return &Recipe[TIntent, TRequestMeta, TMeta]{config: config}, nil
}

func validConfig[TIntent, TRequestMeta, TMeta any](config Config[TIntent, TRequestMeta, TMeta]) bool {
	ports := config.BackendModelFree && !nilPort(config.Backend) && !nilPort(config.Identity) &&
		config.Admission != nil &&
		config.Planner != nil &&
		config.Assessor != nil &&
		config.Pricing != nil
	cloning := config.CloneIntent != nil && config.CloneRequestMeta != nil && config.CloneMeta != nil &&
		config.Supports != nil
	bounds := config.Now != nil && config.Duration > 0 && config.MaxDocuments > 0 && config.FusionK > 0
	return ports && cloning && bounds && (config.QueryEncoder == nil || !nilPort(config.QueryEncoder)) &&
		config.Revision != "" &&
		utf8.ValidString(config.Revision)
}

func queryLimit(strategy Strategy) int {
	switch strategy {
	case SingleRewrite:
		return 1
	case MultiQuery:
		return 2
	case Decomposition:
		return maxSubquestions
	default:
		return 0
	}
}

func nilPort(port any) bool { return nilvalue.IsNil(port) }

type attempt[TIntent, TRequestMeta, TMeta any] struct {
	recipe   *Recipe[TIntent, TRequestMeta, TMeta]
	request  retrieval.Request[TIntent, TRequestMeta]
	ctx      context.Context
	parent   context.Context
	ledger   *budget.Ledger
	result   Result[TMeta]
	planned  int
	fusion   map[int][]retrieval.Document[TMeta]
	deadline time.Time
}

// Run performs a single bounded attempt. Parent cancellation/protection failure
// suppresses all payloads. An attempt-local deadline or budget exhaustion returns
// a typed partial/insufficient result only without independent callback or settlement
// failures. Mixed errors retain their causes and never become bounded success.
func (r *Recipe[TIntent, TRequestMeta, TMeta]) Run(
	ctx context.Context,
	request retrieval.Request[TIntent, TRequestMeta],
	ledger *budget.Ledger,
) (Result[TMeta], error) {
	result, err := r.RunObserved(ctx, request, ledger)
	if err != nil {
		return Result[TMeta]{}, err
	}
	return result, nil
}

// RunObserved retains owned observations and settled usage on ordinary attempt
// errors for explicit recording. An error is never a successful retrieval result.
// Protection failure and parent cancellation suppress the complete journal.
//
//nolint:nonamedreturns // Deferred diagnostics observe the final trusted delivery and failure result.
func (r *Recipe[TIntent, TRequestMeta, TMeta]) RunObserved(
	ctx context.Context,
	request retrieval.Request[TIntent, TRequestMeta],
	ledger *budget.Ledger,
) (output Result[TMeta], runErr error) {
	ctx, span := observation.Begin(ctx, observation.StagePipeline)
	defer func() { span.End(attemptCompletion(output, runErr)) }()
	if r == nil || ledger == nil {
		return Result[TMeta]{}, ragy.ErrInvalidArgument
	}
	if err := request.Read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	if err := request.Options.Validate(); err != nil {
		return Result[TMeta]{}, err
	}
	// Text variants cannot reuse a precomputed vector or graph seed selection.
	if len(request.Options.Vector) != 0 || request.Options.Graph != nil {
		return Result[TMeta]{}, ragy.ErrUnsupported
	}
	deadline := r.config.Now().Add(r.config.Duration)
	child, cancel := context.WithTimeout(ctx, r.config.Duration)
	defer cancel()
	child, ledgerCancel := ledger.Context(child)
	defer ledgerCancel()
	var err error

	a := attempt[TIntent, TRequestMeta, TMeta]{
		recipe:  r,
		request: request,
		ctx:     child,
		parent:  ctx,
		ledger:  ledger,
		result: Result[TMeta]{
			Outcome: "", Stop: "", Queries: nil, Selected: nil, Coverage: nil, Stages: nil,
			Admission: retrieval.UnobservedReadCoverage(), Budget: ledger.Snapshot(),
			Strategy:       r.config.Strategy,
			RecipeRevision: r.config.Revision,
			Publication:    request.Read.Publication().Reference(),
			Fusion:         FusionNotRun,
			Artifact:       nil, Encoding: nil, Sufficiency: nil,
			ArtifactRequested: r.config.Artifact != nil,
		},
		planned:  0,
		deadline: deadline,
		fusion:   make(map[int][]retrieval.Document[TMeta]),
	}
	coverage, err := r.config.Admission(child, retrieval.CopyRequestOptions(request))
	if err != nil {
		return Result[TMeta]{}, access.NonSkippable(err)
	}
	if err = request.Read.Check(child); err != nil {
		return Result[TMeta]{}, err
	}
	if _, err = coverage.MarshalJSON(); err != nil {
		return Result[TMeta]{}, ragy.ErrProtocol
	}
	if coverage.State() == retrieval.CoverageUnobserved {
		return Result[TMeta]{}, ragy.ErrProtocol
	}
	a.result.Admission = retrieval.BindPublicationCoverage(request.Read, coverage)
	a.request, err = a.copyRequest(request)
	if err != nil {
		return Result[TMeta]{}, err
	}
	selected, sufficient, err := a.execute()
	if stopErr := a.stop(err, deadline); stopErr != nil {
		return a.failedResult(stopErr)
	}
	if err = request.Read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	if a.result.Stop != Assessed {
		selected = allQueries(a.result.Queries)
	}
	a.result.Fusion = FusionMissing
	if err = a.observedFusion(selected); err != nil {
		return a.failedResult(err)
	}
	a.result.Fusion = FusionObserved
	a.result.Budget = ledger.Snapshot()
	if err = a.renderDelivery(); err != nil {
		if stopErr := a.stop(err, deadline); stopErr != nil {
			return a.failedResult(stopErr)
		}
	}
	a.finishOutcome(selected, sufficient)
	if err = request.Read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	return a.result, nil
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) failedResult(err error) (Result[TMeta], error) {
	if access.IsProtectionFailure(err) {
		return Result[TMeta]{}, err
	}
	if gateErr := a.request.Read.Check(a.parent); gateErr != nil {
		return Result[TMeta]{}, errors.Join(gateErr, err)
	}
	a.result.Outcome, a.result.Stop = Failure, StageFailure
	a.result.Selected = nil
	a.result.Budget = a.ledger.Snapshot()
	return a.result, err
}

// stageFailureError retains an independent callback or accounting failure. Its
// provenance prevents a joined local deadline from changing failure into success.
type stageFailureError struct{ cause error }

func (*stageFailureError) Error() string   { return "recipe stage failed" }
func (e *stageFailureError) Unwrap() error { return e.cause }

func boundedStop(err error) (StopReason, bool) {
	if !onlyBoundedCauses(err) {
		return "", false
	}
	switch {
	case errors.Is(err, budget.ErrExhausted):
		return BudgetExhausted, true
	case errors.Is(err, budget.ErrUnknownPrice):
		return PriceUnavailable, true
	case errors.Is(err, context.DeadlineExceeded):
		return DeadlineReached, true
	default:
		return "", false
	}
}

func onlyBoundedCauses(err error) bool {
	if err == nil || access.IsProtectionFailure(err) {
		return false
	}
	if _, failed := errors.AsType[*stageFailureError](err); failed {
		return false
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		if len(causes) == 0 {
			return false
		}
		for _, cause := range causes {
			if !onlyBoundedCauses(cause) {
				return false
			}
		}
		return true
	}
	if cause := errors.Unwrap(err); cause != nil {
		return onlyBoundedCauses(cause)
	}
	//nolint:errorlint // Only the exact known terminal causes prove bounded provenance.
	return err == context.DeadlineExceeded || err == budget.ErrExhausted || err == budget.ErrUnknownPrice
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) execute() ([]int, bool, error) {
	if a.recipe.config.Strategy != Decomposition {
		if err := a.retrieve(a.request); err != nil {
			return nil, false, err
		}
	}
	planning, err := a.plan()
	if err != nil {
		return nil, false, err
	}
	a.planned = len(planning.Queries)
	for _, text := range planning.Queries {
		request, copyErr := a.copyRequest(a.request)
		if copyErr != nil {
			return nil, false, copyErr
		}
		request.Text = text
		if request.Plan != nil {
			request.Plan.Text = text
			request.Plan.ExpandedText = ""
		}
		if err = a.retrieve(request); err != nil {
			return nil, false, err
		}
	}
	assessment, err := a.assess()
	if err != nil {
		return nil, false, err
	}
	signal := assessment.Sufficient
	a.result.Sufficiency = &signal
	a.result.Stop = Assessed
	return assessment.Selected, assessment.Sufficient, nil
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) reserve(operation Operation) (budget.Lease, Quote, error) {
	if err := a.gate(a.ctx); err != nil {
		return budget.Lease{}, Quote{}, err
	}
	quote, err := a.recipe.config.Pricing(a.ctx, operation)
	if err != nil {
		return budget.Lease{}, Quote{}, err
	}
	if err = a.gate(a.ctx); err != nil {
		return budget.Lease{}, Quote{}, err
	}
	kind := budget.Model
	if operation == Retrieve {
		kind = budget.Retrieval
		if quote.Usage.InputTokens != 0 || quote.Usage.OutputTokens != 0 {
			return budget.Lease{}, Quote{}, ragy.ErrInvalidArgument
		}
	} else if quote.Usage.InputTokens == 0 || quote.Usage.OutputTokens == 0 {
		return budget.Lease{}, Quote{}, ragy.ErrInvalidArgument
	}
	lease, err := a.ledger.Reserve(
		a.ctx,
		budget.Reservation{Kind: kind, Usage: quote.Usage, CostKnown: quote.CostKnown},
	)
	if err == nil {
		if gateErr := a.gate(a.ctx); gateErr != nil {
			settleErr := lease.Settle(quote.Usage, false)
			return budget.Lease{}, Quote{}, errors.Join(gateErr, settleErr)
		}
	}
	return lease, quote, err
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) settle(
	operation Operation,
	lease budget.Lease,
	quote Quote,
	usage Usage,
	callErr error,
) error {
	observed := usage
	observed.Known = usage.Known && quote.CostKnown
	a.result.Stages = append(a.result.Stages, Stage{Operation: operation, Usage: observed, Completed: false})
	settleErr := lease.Settle(usage.Value, observed.Known)
	if usage.Known &&
		(usage.Value.InputTokens > quote.Usage.InputTokens || usage.Value.OutputTokens > quote.Usage.OutputTokens) {
		settleErr = errors.Join(settleErr, budget.ErrUsageExceeded)
	}
	if settleErr != nil {
		settleErr = &stageFailureError{cause: settleErr}
	}
	callErr = a.callbackError(callErr)
	return errors.Join(callErr, settleErr, a.gate(a.ctx))
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) localCallbackDeadline(err error) bool {
	//nolint:errorlint // A wrapped or joined callback failure has independent provenance.
	return err == context.DeadlineExceeded && a.localExpired()
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) localExpired() bool {
	return errors.Is(a.ctx.Err(), context.DeadlineExceeded) || !a.recipe.config.Now().Before(a.deadline)
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) callbackError(err error) error {
	if err == nil || a.localCallbackDeadline(err) {
		return err
	}
	return &stageFailureError{cause: err}
}

//nolint:nonamedreturns // Deferred diagnostics require the validated output on every return.
func (a *attempt[TIntent, TRequestMeta, TMeta]) plan() (output Planning, stageErr error) {
	ctx, span := observation.Begin(a.ctx, observation.StagePlan)
	defer func() {
		span.End(
			diagnosticCompletion(
				stageErr,
				observation.Count{Known: stageErr == nil, Value: uint64(len(output.Queries))},
			),
		)
	}()
	request, err := a.copyRequest(a.request)
	if err != nil {
		return Planning{}, err
	}
	lease, quote, err := a.reserve(Plan)
	if err != nil {
		return Planning{}, err
	}
	modelCtx, modelSpan := observation.Begin(ctx, observation.StageModel)
	planning, callErr := a.recipe.config.Planner(
		modelCtx,
		request,
		ModelLimits{InputTokens: quote.Usage.InputTokens, OutputTokens: quote.Usage.OutputTokens},
	)
	modelSpan.End(modelCompletion(errors.Join(callErr, modelCtx.Err()), planning.Usage))
	if callErr == nil {
		callErr = a.validatePlanning(planning)
	}
	if err = a.settle(Plan, lease, quote, planning.Usage, callErr); err != nil {
		return Planning{}, err
	}
	planning.Queries = slices.Clone(planning.Queries)
	a.result.Stages[len(a.result.Stages)-1].Completed = true
	return planning, nil
}

//nolint:nonamedreturns // Deferred diagnostics require the validated output on every return.
func (a *attempt[TIntent, TRequestMeta, TMeta]) assess() (output Assessment, stageErr error) {
	ctx, span := observation.Begin(a.ctx, observation.StageAssess)
	defer func() {
		span.End(
			diagnosticCompletion(
				stageErr,
				observation.Count{Known: stageErr == nil, Value: uint64(len(output.Selected))},
			),
		)
	}()
	request, err := a.copyRequest(a.request)
	if err != nil {
		return Assessment{}, err
	}
	queries, err := a.copyQueries(a.result.Queries)
	if err != nil {
		return Assessment{}, err
	}
	lease, quote, err := a.reserve(Assess)
	if err != nil {
		return Assessment{}, err
	}
	modelCtx, modelSpan := observation.Begin(ctx, observation.StageModel)
	assessment, callErr := a.recipe.config.Assessor(
		modelCtx,
		AssessmentInput[TIntent, TRequestMeta, TMeta]{Original: request, Queries: queries},
		ModelLimits{InputTokens: quote.Usage.InputTokens, OutputTokens: quote.Usage.OutputTokens},
	)
	modelSpan.End(modelCompletion(errors.Join(callErr, modelCtx.Err()), assessment.Usage))
	if callErr == nil {
		callErr = a.validateAssessment(assessment)
	}
	if err = a.settle(Assess, lease, quote, assessment.Usage, callErr); err != nil {
		return Assessment{}, err
	}
	assessment.Selected = slices.Clone(assessment.Selected)
	a.result.Stages[len(a.result.Stages)-1].Completed = true
	return assessment, nil
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) validatePlanning(planning Planning) error {
	if len(planning.Queries) > a.recipe.config.MaxQueries {
		return ragy.ErrProtocol
	}
	seen := make(map[string]bool, len(planning.Queries))
	for _, text := range planning.Queries {
		if strings.TrimSpace(text) == "" || !utf8.ValidString(text) || seen[text] {
			return ragy.ErrProtocol
		}
		seen[text] = true
	}
	return nil
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) validateAssessment(assessment Assessment) error {
	seen := make(map[int]bool, len(assessment.Selected))
	for _, index := range assessment.Selected {
		if index < 0 || index >= len(a.result.Queries) || seen[index] {
			return ragy.ErrProtocol
		}
		seen[index] = true
	}
	return nil
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) stop(err error, deadline time.Time) error {
	if err == nil {
		return nil
	}
	if gateErr := a.request.Read.Check(a.parent); gateErr != nil {
		return errors.Join(gateErr, err)
	}
	reason, bounded := boundedStop(err)
	if !bounded {
		return err
	}
	if reason == DeadlineReached && a.ctx.Err() == nil && a.recipe.config.Now().Before(deadline) {
		return err
	}
	a.result.Stop = reason
	return nil
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) gate(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		// An expired child timer alone cannot establish parent/authority failure.
		if parentErr := a.request.Read.Check(a.parent); parentErr != nil {
			return errors.Join(parentErr, err)
		}
		if errors.Is(err, context.DeadlineExceeded) {
			return context.DeadlineExceeded
		}
		return access.NonSkippable(err)
	}
	if err := a.request.Read.Check(ctx); err != nil {
		return err
	}
	if !a.recipe.config.Now().Before(a.deadline) {
		return context.DeadlineExceeded
	}
	return nil
}

// RunOwn explicitly creates a private ledger for an independent attempt.
func (r *Recipe[TIntent, TRequestMeta, TMeta]) RunOwn(
	ctx context.Context,
	request retrieval.Request[TIntent, TRequestMeta],
) (Result[TMeta], error) {
	result, err := r.RunOwnObserved(ctx, request)
	if err != nil {
		return Result[TMeta]{}, err
	}
	return result, nil
}

func (r *Recipe[TIntent, TRequestMeta, TMeta]) RunOwnObserved(
	ctx context.Context,
	request retrieval.Request[TIntent, TRequestMeta],
) (Result[TMeta], error) {
	if r == nil {
		return Result[TMeta]{}, ragy.ErrInvalidArgument
	}
	ledger, err := budget.New(
		budget.Config{
			Limits:           r.config.Limits,
			Deadline:         r.config.Now().Add(r.config.Duration),
			Now:              r.config.Now,
			RequireKnownCost: r.config.RequireKnownCost,
		},
	)
	if err != nil {
		return Result[TMeta]{}, err
	}
	return r.RunObserved(ctx, request, ledger)
}
