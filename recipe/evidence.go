package recipe

import (
	"context"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func (a *attempt[TIntent, TRequestMeta, TMeta]) copyRequest(
	request retrieval.Request[TIntent, TRequestMeta],
) (retrieval.Request[TIntent, TRequestMeta], error) {
	if err := a.gate(a.ctx); err != nil {
		return request, err
	}
	owned := retrieval.CopyRequestOptions(request)
	var err error
	owned.Intent, err = a.recipe.config.CloneIntent(request.Intent)
	if err != nil {
		return owned, err
	}
	if err = a.gate(a.ctx); err != nil {
		return owned, err
	}
	owned.Meta, err = a.recipe.config.CloneRequestMeta(request.Meta)
	if err != nil {
		return owned, err
	}
	if err = a.gate(a.ctx); err != nil {
		return owned, err
	}
	if owned.Plan != nil {
		owned.Plan.Intent, err = a.recipe.config.CloneIntent(owned.Plan.Intent)
		if err != nil {
			return owned, err
		}
	}
	owned.Read = a.request.Read
	return owned, a.gate(a.ctx)
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) copyDocument(
	ctx context.Context,
	doc retrieval.Document[TMeta],
) (retrieval.Document[TMeta], error) {
	if err := a.gate(ctx); err != nil {
		return doc, err
	}
	var err error
	doc.Meta, err = a.recipe.config.CloneMeta(doc.Meta)
	if err != nil {
		return doc, err
	}
	doc.ScoreHistory = slices.Clone(doc.ScoreHistory)
	doc.SourceSupports = slices.Clone(doc.SourceSupports)
	return doc, a.gate(ctx)
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) retrieve(request retrieval.Request[TIntent, TRequestMeta]) error {
	if err := a.gate(a.ctx); err != nil {
		return err
	}
	coverage, err := a.recipe.config.Admission(a.ctx, request)
	if err != nil {
		return access.NonSkippable(err)
	}
	if coverage.State() != a.result.Admission.State() ||
		!slices.Equal(coverage.SkippedBranches(), a.result.Admission.SkippedBranches()) {
		return ragy.ErrProtocol
	}
	lease, quote, err := a.reserve(Retrieve)
	if err != nil {
		return err
	}
	set, callErr := a.recipe.config.Backend.Retrieve(a.ctx, request)
	usage := Usage{Value: quote.Usage, Known: quote.CostKnown}
	var noUsage budget.Usage
	usage.Known = quote.CostKnown && quote.Usage == noUsage
	if err = a.settle(Retrieve, lease, quote, usage, callErr); err != nil {
		return err
	}
	docs, err := a.readSet(set)
	if err != nil {
		return err
	}
	query, err := a.captureQuery(request.EffectiveText(), docs)
	if err != nil {
		return err
	}
	a.result.Queries = append(a.result.Queries, query)
	return nil
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) readSet(
	set retrieval.ResultSet[TMeta],
) ([]retrieval.Document[TMeta], error) {
	if nilPort(set) {
		return nil, ragy.ErrProtocol
	}
	if err := a.gate(a.ctx); err != nil {
		return nil, err
	}
	count := set.Len()
	if err := a.gate(a.ctx); err != nil {
		return nil, err
	}
	if count < 0 || count > a.recipe.config.MaxDocuments {
		return nil, ragy.ErrProtocol
	}
	docs := set.Documents()
	if err := a.gate(a.ctx); err != nil {
		return nil, err
	}
	if len(docs) != count {
		return nil, ragy.ErrProtocol
	}
	return docs, nil
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) captureQuery(
	text string,
	docs []retrieval.Document[TMeta],
) (QueryEvidence[TMeta], error) {
	query := QueryEvidence[TMeta]{
		Keys:      make([]string, len(docs)),
		Index:     len(a.result.Queries),
		Text:      text,
		Documents: make([]retrieval.Document[TMeta], len(docs)),
		Supports:  make([][]source.Locator, len(docs)),
	}
	for i, doc := range docs {
		captured, err := a.captureOne(doc)
		if err != nil {
			return QueryEvidence[TMeta]{}, err
		}
		query.Documents[i], query.Keys[i], query.Supports[i] = captured.document, captured.key, captured.supports
	}
	selection := make([]retrieval.Document[TMeta], len(query.Documents))
	for i, doc := range query.Documents {
		var err error
		selection[i], err = a.copyDocument(a.ctx, doc)
		if err != nil {
			return QueryEvidence[TMeta]{}, err
		}
	}
	a.fusion[query.Index] = selection
	return query, nil
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) copyQueries(
	input []QueryEvidence[TMeta],
) ([]QueryEvidence[TMeta], error) {
	out := make([]QueryEvidence[TMeta], len(input))
	for i, query := range input {
		out[i] = QueryEvidence[TMeta]{
			Keys:      slices.Clone(query.Keys),
			Index:     query.Index,
			Text:      query.Text,
			Documents: make([]retrieval.Document[TMeta], len(query.Documents)),
			Supports:  make([][]source.Locator, len(query.Supports)),
		}
		for j, doc := range query.Documents {
			var err error
			out[i].Documents[j], err = a.copyDocument(a.ctx, doc)
			if err != nil {
				return nil, err
			}
			out[i].Supports[j] = slices.Clone(query.Supports[j])
		}
	}
	return out, nil
}

type fusionMeta[TMeta any] struct {
	value TMeta
	key   string
}
type capturedIdentity[TMeta any] struct{}

func (capturedIdentity[TMeta]) Resolve(doc retrieval.Document[fusionMeta[TMeta]]) retrieval.Identity {
	return retrieval.Identity{DocumentID: doc.ID, MergeKey: doc.Meta.key}
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) selectEvidence(indices []int) error {
	resolver := capturedIdentity[TMeta]{}
	merger, err := retrieval.NewReciprocalRankFusion(a.recipe.config.FusionK, resolver)
	if err != nil {
		return err
	}
	sets := make([]retrieval.ResultSet[fusionMeta[TMeta]], 0, len(indices))
	contributors := make(map[string][]Contribution)
	for _, index := range indices {
		query := a.result.Queries[index]
		docs := make([]retrieval.Document[fusionMeta[TMeta]], len(query.Documents))
		for i, doc := range a.fusion[index] {
			key := query.Keys[i]
			docs[i] = retrieval.Document[fusionMeta[TMeta]]{
				ID: doc.ID, Content: doc.Content, Score: doc.Score, ScoreState: doc.ScoreState,
				ScoreSemantics: doc.ScoreSemantics, ScoreHistory: slices.Clone(doc.ScoreHistory), Rank: doc.Rank,
				Meta:          fusionMeta[TMeta]{value: doc.Meta, key: key},
				SourceMapping: doc.SourceMapping, SourceSupports: slices.Clone(doc.SourceSupports),
			}
			contributors[key] = append(
				contributors[key],
				Contribution{
					QueryIndex: index,
					DocumentID: doc.ID,
					Rank:       i + 1,
					Supports:   slices.Clone(query.Supports[i]),
				},
			)
		}
		sets = append(sets, retrieval.NewResultSet(docs, resolver))
	}
	merged, err := merger.Merge(a.parent, sets...)
	if err != nil {
		return err
	}
	limit := a.request.Options.TopK
	if limit == 0 {
		limit = a.request.Options.FetchLimit
	}
	docs := merged.Documents()
	if len(docs) > limit {
		docs = docs[:limit]
	}
	for _, doc := range docs {
		owned := retrieval.Document[TMeta]{
			ID:             doc.ID,
			Content:        doc.Content,
			Score:          doc.Score,
			ScoreState:     doc.ScoreState,
			ScoreSemantics: doc.ScoreSemantics,
			ScoreHistory:   slices.Clone(doc.ScoreHistory),
			SourceMapping:  doc.SourceMapping, SourceSupports: slices.Clone(doc.SourceSupports),
			Rank: doc.Rank,
			Meta: doc.Meta.value,
		}
		a.result.Selected = append(
			a.result.Selected,
			SelectedEvidence[TMeta]{Document: owned, Contributors: contributors[doc.Meta.key]},
		)
	}
	return nil
}

func allQueries[TMeta any](queries []QueryEvidence[TMeta]) []int {
	out := make([]int, len(queries))
	for i := range queries {
		out[i] = i
	}
	return out
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) finishOutcome(selected []int, sufficient bool) {
	a.result.Outcome = Insufficient
	if len(a.result.Selected) > 0 {
		a.result.Outcome = Partial
		if a.result.Stop == Assessed && sufficient && a.result.Admission.State() != retrieval.CoveragePartial {
			a.result.Outcome = Complete
		}
	}
	if a.recipe.config.Strategy != Decomposition {
		return
	}
	for index := range a.planned {
		coverage := Subquestion{
			Index:       index,
			Retrieved:   index < len(a.result.Queries),
			Selected:    slices.Contains(selected, index),
			HasEvidence: false,
		}
		if coverage.Retrieved {
			coverage.HasEvidence = len(a.result.Queries[index].Documents) > 0
		}
		if !coverage.Selected || !coverage.HasEvidence {
			if a.result.Outcome == Complete {
				a.result.Outcome = Partial
			}
		}
		a.result.Coverage = append(a.result.Coverage, coverage)
	}
}

type capturedDocument[TMeta any] struct {
	document retrieval.Document[TMeta]
	key      string
	supports []source.Locator
}

func (a *attempt[TIntent, TRequestMeta, TMeta]) captureOne(
	doc retrieval.Document[TMeta],
) (capturedDocument[TMeta], error) {
	if err := retrieval.ValidateDocument(doc); err != nil {
		return capturedDocument[TMeta]{}, err
	}
	owned, err := a.copyDocument(a.ctx, doc)
	if err != nil {
		return capturedDocument[TMeta]{}, err
	}
	identityDoc, err := a.copyDocument(a.ctx, doc)
	if err != nil {
		return capturedDocument[TMeta]{}, err
	}
	key := a.recipe.config.Identity.Resolve(identityDoc).MergeKey
	if err = a.gate(a.ctx); err != nil {
		return capturedDocument[TMeta]{}, err
	}
	if key == "" {
		return capturedDocument[TMeta]{}, ragy.ErrProtocol
	}
	supportDoc, err := a.copyDocument(a.ctx, doc)
	if err != nil {
		return capturedDocument[TMeta]{}, err
	}
	supports, err := a.recipe.config.Supports(a.ctx, a.request.Read, supportDoc)
	if err != nil {
		return capturedDocument[TMeta]{}, err
	}
	if err = a.gate(a.ctx); err != nil {
		return capturedDocument[TMeta]{}, err
	}
	supports = slices.Clone(supports)
	if err = validateSupports(a.request.Read.Publication(), supports); err != nil {
		return capturedDocument[TMeta]{}, err
	}
	for _, location := range owned.SourceLocations() {
		if !slices.Contains(supports, location) {
			return capturedDocument[TMeta]{}, access.NonSkippable(ragy.ErrUnavailable)
		}
	}
	return capturedDocument[TMeta]{document: owned, key: key, supports: supports}, nil
}

func validateSupports(publication access.Publication, supports []source.Locator) error {
	if len(supports) == 0 {
		return ragy.ErrProtocol
	}
	for _, location := range supports {
		if err := location.Validate(); err != nil {
			return err
		}
		if !supportedByPublication(publication, location.Reference) {
			return access.NonSkippable(ragy.ErrUnavailable)
		}
	}
	return nil
}

func supportedByPublication(publication access.Publication, reference source.Reference) bool {
	if publication.IsCurrent() {
		return true
	}
	for _, target := range publication.Targets() {
		if target.Namespace == reference.Namespace && target.Source == reference.Source &&
			target.Revision == reference.Revision &&
			target.AccessFingerprint == reference.AccessFingerprint {
			return true
		}
	}
	return false
}
