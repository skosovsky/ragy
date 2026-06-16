package contracttest_test

import (
	"testing"

	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

type passthroughProcessor struct{}

func (passthroughProcessor) Process(
	rs retrieval.ResultSet[struct{}],
) (retrieval.ResultSet[struct{}], error) {
	return rs, nil
}

func TestPostProcessorChainContractConformance(t *testing.T) {
	contracttest.RunPostProcessorChainSuite(t, contracttest.PostProcessorChainConfig{
		CustomProcessor: passthroughProcessor{},
		BackendDocs: []retrieval.Document[struct{}]{
			{ID: "a", Content: "same-key", Score: 0.9},
			{ID: "b", Content: "same-key", Score: 0.5},
		},
		WantLen: 2,
	})
}
