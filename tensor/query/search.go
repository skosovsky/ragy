package query

import (
	"context"
	"reflect"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
)

type Config[TCandidateMeta, TMeta any] struct {
	Candidates         retrieval.Backend[Intent, TCandidateMeta]
	Target             Target[TMeta]
	CloneCandidateMeta func(TCandidateMeta) (TCandidateMeta, error)
	Reference          func(retrieval.Document[TCandidateMeta]) (source.Reference, error)
}

// Search dispatches one candidate query and one bounded tensor query. It neither
// retries nor fuses incompatible numeric scores. The original binding is retained.
type Search[TCandidateMeta, TMeta any] struct {
	config            Config[TCandidateMeta, TMeta]
	candidateProvider retrieval.ReadCapabilityProvider
}

func New[TCandidateMeta, TMeta any](config Config[TCandidateMeta, TMeta]) (*Search[TCandidateMeta, TMeta], error) {
	if nilPort(config.Candidates) || nilPort(config.Target) || config.CloneCandidateMeta == nil ||
		config.Reference == nil {
		return nil, ragy.ErrInvalidArgument
	}
	provider, ok := config.Candidates.(retrieval.ReadCapabilityProvider)
	if !ok {
		return nil, ragy.ErrUnsupported
	}
	return &Search[TCandidateMeta, TMeta]{config: config, candidateProvider: provider}, nil
}

func (s *Search[TCandidateMeta, TMeta]) Schema() filter.Schema {
	if s == nil || s.config.Target == nil {
		return filter.Schema{}
	}
	return s.config.Target.Schema()
}
func (s *Search[TCandidateMeta, TMeta]) ReadCapabilities() access.Capabilities {
	if s == nil || s.config.Target == nil || s.candidateProvider == nil {
		return access.Capabilities{ScopeProfile: false, PinnedPublication: false, RequirePinnedPublication: true}
	}
	candidate, target := s.candidateProvider.ReadCapabilities(), s.config.Target.ReadCapabilities()
	return access.Capabilities{
		RequirePinnedPublication: candidate.RequirePinnedPublication || target.RequirePinnedPublication,
		ScopeProfile:             candidate.ScopeProfile && target.ScopeProfile,
		PinnedPublication:        candidate.PinnedPublication && target.PinnedPublication,
	}
}

// AdmitPublication negotiates both candidate and tensor targets before I/O.
func (s *Search[TCandidateMeta, TMeta]) AdmitPublication(publication access.Publication) error {
	if s == nil || nilPort(s.config.Candidates) || nilPort(s.config.Target) {
		return access.Protect(ragy.ErrInvalidArgument)
	}
	for _, target := range []any{s.config.Candidates, s.config.Target} {
		admission, ok := target.(retrieval.PublicationAdmission)
		if !ok {
			return access.UnsupportedCapability(ragy.ErrUnsupported)
		}
		if err := admission.AdmitPublication(publication); err != nil {
			return err
		}
	}
	return nil
}

func (s *Search[TCandidateMeta, TMeta]) AdmitRead(
	ctx context.Context,
	request retrieval.Query[Intent],
) (retrieval.ReadCoverage, error) {
	if s == nil || s.config.Target == nil || s.candidateProvider == nil {
		return retrieval.UnobservedReadCoverage(), ragy.ErrInvalidArgument
	}
	prepared, err := s.plannedFilters(ctx, request)
	if err != nil {
		return retrieval.UnobservedReadCoverage(), err
	}
	request = prepared
	candidateRequest := s.candidateRequest(request)
	candidateCoverage, err := admitTarget(ctx, candidateRequest, s.candidateProvider)
	if err != nil {
		return retrieval.UnobservedReadCoverage(), err
	}
	targetRequest := CopyRequest(request)
	targetRequest.Options.Vector = nil
	targetCoverage, err := admitTarget(ctx, targetRequest, s.config.Target)
	if err != nil {
		return retrieval.UnobservedReadCoverage(), err
	}
	return retrieval.MergeReadCoverage(candidateCoverage, targetCoverage), nil
}

func (s *Search[TCandidateMeta, TMeta]) Retrieve(
	ctx context.Context,
	request retrieval.Query[Intent],
) (retrieval.ResultSet[TMeta], error) {
	result, err := s.Query(ctx, request)
	return result.Documents, err
}

func (s *Search[TCandidateMeta, TMeta]) Query(
	ctx context.Context,
	request retrieval.Query[Intent],
) (Result[TMeta], error) {
	empty := retrieval.NewResultSet[TMeta](nil, nil)
	result, err := s.query(ctx, CopyRequest(request))
	if err != nil {
		return Result[TMeta]{Documents: empty, Evidence: emptyEvidence()}, access.NonSkippable(err)
	}
	if err = request.Read.Check(ctx); err != nil {
		return Result[TMeta]{Documents: empty, Evidence: emptyEvidence()}, err
	}
	return result, nil
}

func emptyEvidence() tensor.RerankResult {
	return tensor.RerankResult{Ranking: nil, CandidateIDs: nil, CandidateBudget: 0}
}

func (s *Search[TCandidateMeta, TMeta]) query(
	ctx context.Context,
	request retrieval.Query[Intent],
) (Result[TMeta], error) {
	if s == nil || s.config.Target == nil || s.candidateProvider == nil {
		return Result[TMeta]{}, ragy.ErrInvalidArgument
	}
	if err := request.Read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	if err := s.validate(ctx, request); err != nil {
		return Result[TMeta]{}, err
	}
	var err error
	request, err = s.plannedFilters(ctx, request)
	if err != nil {
		return Result[TMeta]{}, err
	}

	if _, err = s.AdmitRead(ctx, request); err != nil {
		return Result[TMeta]{}, err
	}
	candidatesRequest := s.candidateRequest(request)
	candidates, err := s.config.Candidates.Retrieve(ctx, candidatesRequest)
	if err != nil {
		return Result[TMeta]{}, err
	}
	if err = request.Read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	if candidates == nil || candidates.Len() > request.Intent.CandidateBudget {
		return Result[TMeta]{}, ragy.ErrProtocol
	}
	docs := candidates.Documents()
	if err = request.Read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	if len(docs) > request.Intent.CandidateBudget || len(docs) != candidates.Len() {
		return Result[TMeta]{}, ragy.ErrProtocol
	}
	refs, err := s.project(ctx, request.Read, docs)
	if err != nil {
		return Result[TMeta]{}, err
	}
	request.Intent.Candidates = refs
	request.Options.Vector = nil
	return s.config.Target.Query(ctx, request)
}

func (s *Search[TCandidateMeta, TMeta]) validate(ctx context.Context, request retrieval.Query[Intent]) error {
	caps := s.config.Target.QueryCapabilities()
	if len(request.Intent.Candidates) != 0 || request.Intent.CandidateBudget <= 0 ||
		request.Intent.CandidateBudget > caps.CandidateLimit ||
		request.Options.TopK <= 0 ||
		request.Options.TopK > request.Intent.CandidateBudget {
		return ragy.ErrInvalidArgument
	}
	if !caps.ExactWithinCandidates || caps.Exhaustive || caps.ScoreSemantics == "" {
		return ragy.ErrUnsupported
	}
	if request.Intent.Embedding.Space != caps.Space {
		return ragy.ErrInvalidArgument
	}
	if err := request.Intent.Embedding.ValidateContext(ctx); err != nil {
		return err
	}
	if request.Options.Threshold != nil || request.Options.Graph != nil {
		return ragy.ErrUnsupported
	}
	return request.Options.Validate()
}

func (s *Search[TCandidateMeta, TMeta]) project(
	ctx context.Context,
	read access.Binding,
	docs []retrieval.Document[TCandidateMeta],
) ([]source.Reference, error) {
	var refs []source.Reference
	seen := make(map[source.Reference]struct{}, len(docs))
	for _, doc := range docs {
		if err := read.Check(ctx); err != nil {
			return nil, err
		}
		meta, err := s.config.CloneCandidateMeta(doc.Meta)
		if err != nil {
			return nil, err
		}
		if err = read.Check(ctx); err != nil {
			return nil, err
		}
		doc.Meta = meta
		doc.ScoreHistory = slices.Clone(doc.ScoreHistory)
		doc.SourceSupports = slices.Clone(doc.SourceSupports)
		ref, err := s.config.Reference(doc)
		if err != nil {
			return nil, err
		}
		if err = read.Check(ctx); err != nil {
			return nil, err
		}
		if err = ref.Validate(); err != nil {
			return nil, err
		}
		if _, exists := seen[ref]; exists {
			continue
		}
		seen[ref] = struct{}{}
		refs = append(refs, ref)
	}
	return refs, nil
}

func nilPort(port any) bool {
	if port == nil {
		return true
	}
	value := reflect.ValueOf(port)
	kinds := []reflect.Kind{reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice}
	return slices.Contains(kinds, value.Kind()) && value.IsNil()
}

func (s *Search[TCandidateMeta, TMeta]) plannedFilters(
	ctx context.Context,
	request retrieval.Query[Intent],
) (retrieval.Query[Intent], error) {
	return retrieval.PrepareRead(ctx, request, s)
}

func (s *Search[TCandidateMeta, TMeta]) candidateRequest(request retrieval.Query[Intent]) retrieval.Query[Intent] {
	candidate := CopyRequest(request)
	candidate.Options.TopK, candidate.Options.FetchLimit = request.Intent.CandidateBudget, request.Intent.CandidateBudget
	return candidate
}

func admitTarget(
	ctx context.Context,
	request retrieval.Query[Intent],
	provider retrieval.ReadCapabilityProvider,
) (retrieval.ReadCoverage, error) {
	if _, err := retrieval.PrepareRead(ctx, request, provider); err != nil {
		return retrieval.UnobservedReadCoverage(), err
	}
	if admission, ok := provider.(retrieval.RequestReadAdmission[Intent, retrieval.NoRequestMeta]); ok {
		return retrieval.InspectRead(ctx, request, admission)
	}
	return retrieval.BindPublicationCoverage(request.Read, retrieval.CompleteReadCoverage()), nil
}
