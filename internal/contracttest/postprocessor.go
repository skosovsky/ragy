package contracttest

import (
	"context"
	"strings"
	"testing"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

const (
	postProcessorContractTopK = 5
	groupByContractTopK       = 10
	groupByWantLen            = 2
	groupDocScoreHigh         = 0.9
	groupDocScoreMid          = 0.8
	groupDocScoreLow          = 0.7
)

// PostProcessorChainConfig supplies inputs for RunPostProcessorChainSuite.
type PostProcessorChainConfig struct {
	CustomProcessor retrieval.PostProcessor[struct{}]
	BackendDocs     []retrieval.Document[struct{}]
	WantLen         int
}

type groupMeta struct {
	Group string
}

type groupBackend struct {
	docs []retrieval.Document[groupMeta]
}

func (b groupBackend) Schema() filter.Schema { return filter.EmptySchema() }

func (b groupBackend) Retrieve(
	_ context.Context,
	_ string,
	_ retrieval.RetrieveOptions,
) (retrieval.ResultSet[groupMeta], error) {
	return retrieval.NewResultSet(b.docs, retrieval.DocumentIDResolver[groupMeta]{}), nil
}

// RunPostProcessorChainSuite checks post-processor chain resolver injection semantics.
func RunPostProcessorChainSuite(t *testing.T, cfg PostProcessorChainConfig) {
	t.Helper()

	t.Run("custom processor passthrough without resolver injection", func(t *testing.T) {
		t.Parallel()
		testPostProcessorCustomPassthrough(t, cfg)
	})

	t.Run("builtin GroupBy receives pipeline resolver", func(t *testing.T) {
		t.Parallel()
		testPostProcessorGroupByContract(t)
	})

	t.Run("second WithPostProcessors overwrites chain", func(t *testing.T) {
		t.Parallel()
		testPostProcessorChainOverwrites(t)
	})
}

type suffixPostProcessor[TMeta any] struct {
	suffix string
}

func (p suffixPostProcessor[TMeta]) Process(rs retrieval.ResultSet[TMeta]) (retrieval.ResultSet[TMeta], error) {
	docs := rs.Documents()
	if len(docs) == 0 {
		return rs, nil
	}
	doc := docs[0]
	doc.Content += p.suffix
	return retrieval.NewResultSet([]retrieval.Document[TMeta]{doc}, retrieval.DocumentIDResolver[TMeta]{}), nil
}

func testPostProcessorChainOverwrites(t *testing.T) {
	t.Helper()

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.RetrieverNode[struct{}, struct{}]{
			Backend: postProcessorStubBackend[struct{}]{
				docs: []retrieval.Document[struct{}]{{ID: "a", Content: "base", Score: 1}},
			},
		}).
		WithPostProcessors(suffixPostProcessor[struct{}]{suffix: "-first"}).
		WithPostProcessors(suffixPostProcessor[struct{}]{suffix: "-second"}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Text:    "q",
		Options: DefaultRetrieveOptions(),
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	got := rs.Documents()[0].Content
	if got != "base-second" {
		t.Fatalf("Content = %q, want base-second (second WithPostProcessors overwrote first)", got)
	}
	if strings.Contains(got, "-first") {
		t.Fatalf("Content = %q, first processor must not run", got)
	}
}

func testPostProcessorCustomPassthrough(t *testing.T, cfg PostProcessorChainConfig) {
	t.Helper()

	mergeResolver := ContentMergeResolver[struct{}]{}
	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.RetrieverNode[struct{}, struct{}]{
			Backend: postProcessorStubBackend[struct{}]{docs: cfg.BackendDocs},
		}).
		WithPostProcessors(cfg.CustomProcessor).
		WithResolver(mergeResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Text:    "q",
		Options: retrieval.RetrieveOptions{TopK: postProcessorContractTopK},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != cfg.WantLen {
		t.Fatalf(
			"Len() = %d, want %d (custom processor not merged by pipeline WithResolver)",
			rs.Len(),
			cfg.WantLen,
		)
	}
}

func testPostProcessorGroupByContract(t *testing.T) {
	t.Helper()

	backend := groupBackend{docs: []retrieval.Document[groupMeta]{
		{ID: "1", Content: "a", Score: groupDocScoreHigh, Meta: groupMeta{Group: "g1"}},
		{ID: "2", Content: "b", Score: groupDocScoreMid, Meta: groupMeta{Group: "g1"}},
		{ID: "3", Content: "c", Score: groupDocScoreLow, Meta: groupMeta{Group: "g2"}},
	}}
	pipeline, err := retrieval.NewPipelineBuilder[struct{}, groupMeta]().
		WithRoot(retrieval.RetrieverNode[struct{}, groupMeta]{Backend: backend}).
		WithPostProcessors(retrieval.GroupBy(
			func(m groupMeta) string { return m.Group },
			retrieval.DefaultMergeStrategy[groupMeta](),
		)).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Text:    "query",
		Options: retrieval.RetrieveOptions{TopK: groupByContractTopK},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != groupByWantLen {
		t.Fatalf("Len() = %d, want %d after GroupBy", rs.Len(), groupByWantLen)
	}
}

type postProcessorStubBackend[TMeta any] struct {
	docs []retrieval.Document[TMeta]
}

func (b postProcessorStubBackend[TMeta]) Schema() filter.Schema { return filter.EmptySchema() }

func (b postProcessorStubBackend[TMeta]) Retrieve(
	_ context.Context,
	_ string,
	_ retrieval.RetrieveOptions,
) (retrieval.ResultSet[TMeta], error) {
	return retrieval.NewResultSet(b.docs, retrieval.DocumentIDResolver[TMeta]{}), nil
}
