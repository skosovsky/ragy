//go:build ignore

package main

import (
	"context"
	"fmt"
	"math"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

func main() {
	for _, cfg := range []lexical.Config[struct{}]{{SearchFields: []string{"content"}, K1: math.NaN()}, {SearchFields: []string{"content"}, K1: math.Inf(1)}, {SearchFields: []string{"content"}, B: math.NaN()}} {
		idx, err := lexical.NewBM25Index(filter.EmptySchema(), cfg, nil, nil)
		fmt.Println("construct", err)
		if err != nil {
			continue
		}
		if err = idx.Index([]retrieval.Document[struct{}]{{ID: "d", Content: "hello"}}); err != nil {
			panic(err)
		}
		rs, err := idx.Retrieve(
			context.Background(),
			retrieval.Query[struct{}]{
				Read:    retrieval.UnrestrictedRead(),
				Text:    "hello",
				Options: retrieval.RetrieveOptions{TopK: 1},
			},
		)
		fmt.Println(
			"retrieve",
			err,
			"score",
			rs.Documents()[0].Score,
			"validation",
			retrieval.ValidateDocument(rs.Documents()[0]),
		)
	}
}
