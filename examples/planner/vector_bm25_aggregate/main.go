// Command vector_bm25_aggregate demonstrates dense + lexical fusion with AggregateNode and RRF.
// Replace the BM25 branch with adapters/elasticsearch.Store for production lexical search (drop-in Backend).
package main

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

type searchIntent struct{}

type vectorBackend struct {
	schema filter.Schema
	hits   []retrieval.Document[struct{}]
}

func (m vectorBackend) Schema() filter.Schema { return m.schema }

func (m vectorBackend) Retrieve(
	_ context.Context,
	req retrieval.Query[searchIntent],
) (retrieval.ResultSet[struct{}], error) {
	opts := req.Options
	if len(opts.Vector) == 0 {
		return retrieval.NewResultSet[struct{}](nil, retrieval.DocumentIDResolver[struct{}]{}),
			ragy.ErrEmptyVector
	}
	return retrieval.NewResultSet(m.hits, retrieval.DocumentIDResolver[struct{}]{}), nil
}

const (
	vectorHitScore = 0.92
	exampleTopK    = 5
)

func buildPipeline(
	vector vectorBackend,
	bm25 retrieval.Backend[struct{}, struct{}],
) (*retrieval.Pipeline[searchIntent, struct{}], error) {
	bm25Branch := retrieval.ProjectedBackend[searchIntent, retrieval.NoRequestMeta, struct{}, retrieval.NoRequestMeta, struct{}]{
		Next: bm25,
		Project: func(req retrieval.Query[searchIntent]) retrieval.Query[struct{}] {
			return retrieval.Query[struct{}]{
				Text:    req.Text,
				Options: req.Options,
				Plan:    retrieval.ProjectPlannedQuery(req.Plan, struct{}{}),
			}
		},
	}
	return retrieval.NewPipelineBuilder[searchIntent, struct{}]().
		WithRoot(retrieval.AggregateNode[searchIntent, struct{}]{
			Nodes: []retrieval.Node[searchIntent, struct{}]{
				retrieval.RetrieverNode[searchIntent, struct{}]{Backend: vector},
				retrieval.RetrieverNode[searchIntent, struct{}]{Backend: bm25Branch},
			},
		}).
		Build()
}

func main() {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		panic(err)
	}

	bm25, err := lexical.NewBM25Index[struct{}](
		schema,
		lexical.Config[struct{}]{SearchFields: []string{"content"}},
		lexical.DefaultTokenizer{},
		nil,
	)
	if err != nil {
		panic(err)
	}
	if upsertErr := bm25.Upsert(retrieval.Document[struct{}]{
		ID: "lex-1", Content: "bm25 keyword match", Score: 0,
	}); upsertErr != nil {
		panic(upsertErr)
	}

	vector := vectorBackend{
		schema: schema,
		hits: []retrieval.Document[struct{}]{
			{ID: "vec-1", Content: "dense neighbor", Score: vectorHitScore},
		},
	}

	pipeline, err := buildPipeline(vector, bm25)
	if err != nil {
		panic(err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[searchIntent]{
		Text:    "keyword",
		Options: retrieval.RetrieveOptions{TopK: exampleTopK, Vector: []float32{0.1, 0.2}},
	})
	if err != nil {
		panic(err)
	}
	if rs.IsEmpty() {
		panic("expected fused hits from vector and BM25 branches")
	}
	fmt.Printf("fused docs: %d\n", rs.Len())
	for _, doc := range rs.Documents() {
		fmt.Printf("- %s score=%.3f\n", doc.ID, doc.Score)
	}
}
