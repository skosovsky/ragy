package tensor_test

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/skosovsky/ragy/tensor"
)

// checkpointContext cancels at a deterministic cooperative checkpoint. These
// tests exercise cancellation during validation without scheduler timing.
type checkpointContext struct {
	context.Context

	remaining int
}

func (c *checkpointContext) Err() error {
	c.remaining--
	if c.remaining <= 0 {
		return context.Canceled
	}
	return nil
}

func TestTensorValidationCancellationDiscardsResult(t *testing.T) {
	// Arrange: valid initial rows followed by malformed data. Cancellation must
	// be observed while validating rows, before reaching the malformed last row.
	malformed := embedding(tensor.Tensor{{1, 0}, {0, 1}, {1}})
	valid := embedding(tensor.Tensor{{1, 0}})
	for _, operation := range []string{"validate", "maxsim", "rerank"} {
		t.Run(operation, func(t *testing.T) {
			ctx := &checkpointContext{Context: context.Background(), remaining: 4}
			// Act.
			var err error
			switch operation {
			case "validate":
				err = malformed.ValidateContext(ctx)
			case "maxsim":
				var score float64
				score, err = tensor.MaxSim(ctx, malformed, valid)
				if score != 0 {
					t.Fatal("canceled score delivered", score)
				}
			case "rerank":
				var result tensor.RerankResult
				result, err = tensor.Rerank(
					ctx,
					malformed,
					[]tensor.Candidate{{ID: "d", Embedding: valid}},
					tensor.RerankOptions{CandidateBudget: 1, TopK: 1},
				)
				if !reflect.DeepEqual(result, tensor.RerankResult{}) {
					t.Fatal("canceled ranking delivered", result)
				}
			}
			// Assert.
			if !errors.Is(err, context.Canceled) {
				t.Fatal("validation ignored cancellation", err)
			}
		})
	}
}

func TestTensorCandidateValidationChecksCancellation(t *testing.T) {
	// Arrange: cancellation arrives between candidate rows, before invalid input.
	valid := embedding(tensor.Tensor{{1, 0}})
	candidates := []tensor.Candidate{{ID: "a", Embedding: valid}, {ID: "b", Embedding: embedding(tensor.Tensor{{1}})}}
	ctx := &checkpointContext{Context: context.Background(), remaining: 8}
	// Act.
	result, err := tensor.Rerank(ctx, valid, candidates, tensor.RerankOptions{CandidateBudget: 2, TopK: 1})
	// Assert.
	if !errors.Is(err, context.Canceled) || !reflect.DeepEqual(result, tensor.RerankResult{}) {
		t.Fatal("candidate cancellation", result, err)
	}
}
