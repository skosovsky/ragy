package contracttest

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
)

// QueryRerankerLike is the minimal rerank surface used by contract suites.
type QueryRerankerLike[TMeta any] interface {
	Rerank(
		ctx context.Context,
		query string,
		rs retrieval.ResultSet[TMeta],
	) (retrieval.ResultSet[TMeta], error)
}

// RerankFactory builds a query-aware reranker for contract tests.
type RerankFactory func(t *testing.T) QueryRerankerLike[StructMeta]

const partialInvalidDocumentScore = 1.5

// RunRerankPartialSuite checks partial validate, resolver parity, and error preservation.
func RunRerankPartialSuite(t *testing.T, factory RerankFactory) {
	t.Helper()

	t.Run("empty query preserves input resolver", func(t *testing.T) {
		reranker := factory(t)
		resolver := ContentMergeResolver[StructMeta]{}
		rs := retrieval.NewResultSet(
			[]retrieval.Document[StructMeta]{{ID: "a", Content: "merge-key"}},
			resolver,
		)
		out, err := reranker.Rerank(context.Background(), "", rs)
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrEmptyText) {
			t.Fatalf("Rerank() error = %v, want empty text", err)
		}
		merged, mergeErr := retrieval.NewResultSet(
			[]retrieval.Document[StructMeta]{{ID: "b", Content: "merge-key", Score: partialMergeProbeScore}},
			resolver,
		).Merge(out)
		if mergeErr != nil {
			t.Fatalf("Merge(): %v", mergeErr)
		}
		if merged.Len() != 1 {
			t.Fatalf("merged Len() = %d, want 1 doc under custom merge key", merged.Len())
		}
	})

	t.Run("invalid document returns protocol error", func(t *testing.T) {
		reranker := factory(t)
		rs := retrieval.NewResultSet(
			[]retrieval.Document[StructMeta]{{Content: "missing-id"}},
			retrieval.DocumentIDResolver[StructMeta]{},
		)
		out, err := reranker.Rerank(context.Background(), "q", rs)
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrProtocol) {
			t.Fatalf("Rerank() error = %v, want protocol", err)
		}
	})

	t.Run("mixed valid and invalid preserves partial", func(t *testing.T) {
		reranker := factory(t)
		rs := retrieval.NewResultSet(
			[]retrieval.Document[StructMeta]{
				{ID: "ok", Content: "alpha"},
				{ID: "bad", Content: "beta", Score: partialInvalidDocumentScore},
			},
			retrieval.DocumentIDResolver[StructMeta]{},
		)
		out, err := reranker.Rerank(context.Background(), "q", rs)
		if !errors.Is(err, ragy.ErrProtocol) {
			t.Fatalf("Rerank() error = %v, want protocol", err)
		}
		if out.Len() != 1 || out.Documents()[0].ID != "ok" {
			t.Fatalf("Rerank() docs = %#v, want preserved valid doc", out.Documents())
		}
	})
}
