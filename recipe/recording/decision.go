package recording

import (
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/recipe"
)

func decisions[TMeta any](result recipe.Result[TMeta], revisions evidence.HostRevisions) *evidence.DecisionInput {
	out := &evidence.DecisionInput{
		Queries:     nil,
		Selected:    nil,
		Fusion:      evidence.MissingObservation,
		Sufficiency: result.Sufficiency,
		Stop:        string(result.Stop),
		Revisions:   revisions,
	}
	if out.Stop == "" {
		return nil
	}
	switch result.Fusion {
	case recipe.FusionObserved:
		out.Fusion = evidence.StageObserved
	case recipe.FusionNotRun:
		out.Fusion = evidence.NotRun
	case recipe.FusionMissing:
	}
	for _, query := range result.Queries {
		item := evidence.DecisionQueryInput{
			Index:     query.Index,
			Text:      query.Text,
			Retrieved: true,
			Selected:  false,
			Delivered: false,
			Uncertain: false,
		}
		item = queryDecision(result, item)

		out.Queries = append(out.Queries, item)
	}
	for index, selected := range result.Selected {
		item := evidence.DecisionSelection{Index: index, Contributors: nil, Delivered: false, Uncertain: false}
		for _, contributor := range selected.Contributors {
			item.Contributors = append(
				item.Contributors,
				evidence.DecisionContributor{QueryIndex: contributor.QueryIndex, Rank: contributor.Rank},
			)
		}
		item.Delivered, item.Uncertain = selectionDelivery(result, index)

		out.Selected = append(out.Selected, item)
	}
	return out
}

func queryDecision[TMeta any](
	result recipe.Result[TMeta],
	item evidence.DecisionQueryInput,
) evidence.DecisionQueryInput {
	for index, selected := range result.Selected {
		for _, contributor := range selected.Contributors {
			if contributor.QueryIndex == item.Index {
				item.Selected = true
				delivered, uncertain := selectionDelivery(result, index)
				item.Delivered = item.Delivered || delivered
				item.Uncertain = item.Uncertain || uncertain
			}
		}
	}
	for _, coverage := range result.Coverage {
		if coverage.Index == item.Index {
			item.Retrieved = coverage.Retrieved
			item.Selected = coverage.SelectedEvidence
			item.Delivered = coverage.DeliveredEvidence
			item.Uncertain = coverage.DeliveryUncertain
		}
	}
	return item
}
func selectionDelivery[TMeta any](result recipe.Result[TMeta], index int) (bool, bool) {
	if result.Artifact == nil {
		return !result.ArtifactRequested, result.ArtifactRequested
	}
	delivered, uncertain := false, false
	for _, snippet := range result.Artifact.Snippets {
		for _, packed := range snippet.Contributors {
			if packed.InputIndex == index {
				delivered = true
				uncertain = uncertain || packed.DeliveryUncertain || !packed.FullDocument
			}
		}
	}
	return delivered, uncertain
}
