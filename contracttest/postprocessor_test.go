package contracttest_test

import (
	"context"

	"github.com/skosovsky/ragy/access"

	"testing"

	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

type passthroughProcessor struct{}

func (passthroughProcessor) Process(_ context.Context, _ access.Binding,
	rs retrieval.ResultSet[struct{}],
) (retrieval.ResultSet[struct{}], error) {
	return rs, nil
}

func TestPostProcessorChainContractConformance(t *testing.T) {
	contracttest.RunPostProcessorChainSuite(t, contracttest.PostProcessorChainConfig{
		CustomProcessor: passthroughProcessor{},
		BackendDocs: []retrieval.Document[struct{}]{
			{
				ScoreSemantics: "fixture-similarity",
				ScoreState:     retrieval.ScorePresent,
				ID:             "a",
				Content:        "same-key",
				Score:          0.9,
			},
			{
				ScoreSemantics: "fixture-similarity",
				ScoreState:     retrieval.ScorePresent,
				ID:             "b",
				Content:        "same-key",
				Score:          0.5,
			},
		},
		WantLen: 2,
	})
}
