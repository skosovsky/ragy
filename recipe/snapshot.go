package recipe

import (
	"context"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// SnapshotResult captures BYOT metadata and every mutable slice for downstream
// processing/export. It never refreshes a publication or source locator.
func SnapshotResult[TMeta any](
	ctx context.Context,
	read access.Binding,
	input Result[TMeta],
	clone func(TMeta) (TMeta, error),
) (Result[TMeta], error) {
	if clone == nil {
		return Result[TMeta]{}, ragy.ErrInvalidArgument
	}
	if err := read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	out := input
	out.Stages = slices.Clone(input.Stages)
	out.Coverage = slices.Clone(input.Coverage)
	out.Queries = make([]QueryEvidence[TMeta], len(input.Queries))
	for i, query := range input.Queries {
		if query.Index != i || len(query.Keys) != len(query.Documents) || len(query.Supports) != len(query.Documents) {
			return Result[TMeta]{}, ragy.ErrProtocol
		}
		owned, err := snapshotQuery(ctx, read, query, clone)
		if err != nil {
			return Result[TMeta]{}, err
		}
		out.Queries[i] = owned
	}
	out.Selected = make([]SelectedEvidence[TMeta], len(input.Selected))
	for i, selected := range input.Selected {
		doc, err := snapshotDocument(ctx, read, selected.Document, clone)
		if err != nil {
			return Result[TMeta]{}, err
		}
		contributors := slices.Clone(selected.Contributors)
		for j := range contributors {
			contributors[j].Supports = slices.Clone(contributors[j].Supports)
			if contributors[j].QueryIndex < 0 || contributors[j].QueryIndex >= len(input.Queries) {
				return Result[TMeta]{}, ragy.ErrProtocol
			}
		}
		out.Selected[i] = SelectedEvidence[TMeta]{Document: doc, Contributors: contributors}
	}
	if err := read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	return out, nil
}

func snapshotQuery[TMeta any](
	ctx context.Context,
	read access.Binding,
	input QueryEvidence[TMeta],
	clone func(TMeta) (TMeta, error),
) (QueryEvidence[TMeta], error) {
	out := QueryEvidence[TMeta]{
		Index:     input.Index,
		Text:      input.Text,
		Keys:      slices.Clone(input.Keys),
		Documents: make([]retrieval.Document[TMeta], len(input.Documents)),
		Supports:  make([][]source.Locator, len(input.Supports)),
	}
	for i, doc := range input.Documents {
		owned, err := snapshotDocument(ctx, read, doc, clone)
		if err != nil {
			return QueryEvidence[TMeta]{}, err
		}
		out.Documents[i] = owned
		out.Supports[i] = slices.Clone(input.Supports[i])
	}
	return out, nil
}

func snapshotDocument[TMeta any](
	ctx context.Context,
	read access.Binding,
	doc retrieval.Document[TMeta],
	clone func(TMeta) (TMeta, error),
) (retrieval.Document[TMeta], error) {
	if err := read.Check(ctx); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	var err error
	doc.Meta, err = clone(doc.Meta)
	if err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	doc.ScoreHistory = slices.Clone(doc.ScoreHistory)
	doc.SourceSupports = slices.Clone(doc.SourceSupports)
	if err = read.Check(ctx); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	return doc, retrieval.ValidateDocument(doc)
}
