package recording

import (
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/recipe"
)

func validateQueryContributions[TMeta any](result recipe.Result[TMeta]) error {
	for _, selected := range result.Selected {
		for _, item := range selected.Contributors {
			if item.QueryIndex < 0 || item.QueryIndex >= len(result.Queries) {
				return ragy.ErrProtocol
			}
			query := result.Queries[item.QueryIndex]
			if item.Rank <= 0 || item.Rank > len(query.Documents) || item.Rank > len(query.Supports) {
				return ragy.ErrProtocol
			}
			if item.DocumentID != query.Documents[item.Rank-1].ID ||
				!slices.Equal(item.Supports, query.Supports[item.Rank-1]) {
				return ragy.ErrProtocol
			}
		}
	}
	return nil
}
