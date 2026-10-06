package retrieval

import (
	"errors"
	"fmt"
	"testing"
)

func TestT12CompletenessEmptyPartialNeverRescued(t *testing.T) {
	// Arrange.
	cause := errors.New("partial cause")
	empty := NewResultSet[struct{}](nil, nil)
	partial := &PartialFailureError[struct{}]{Errors: []error{cause}, Result: empty}
	node := resultRescueNodeNoMeta[stubIntent, struct{}]{
		Primary: errorNode[stubIntent, struct{}]{
			err: errors.Join(fmt.Errorf("outer: %w", partial), errors.New("sibling")),
		},
		Secondary: stubNode[struct{}]{docs: []Document[struct{}]{{ID: "secondary", Content: "secondary"}}},
	}
	// Act.
	out, err := node.Retrieve(t.Context(), pipelineTestQuery("q"))
	// Assert.
	if err == nil || !errors.Is(err, cause) || !out.IsEmpty() {
		t.Fatalf("empty partial rescued: out=%v err=%v", out.Documents(), err)
	}
}
