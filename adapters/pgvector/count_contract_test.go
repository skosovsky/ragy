package pgvector

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestExactDeleteCountRejectsUnknownSentinel(t *testing.T) {
	for _, count := range []int64{-1, 0, 5} {
		// Arrange: -1 is a host's unknown sentinel, zero and five are exact outcomes.
		// Act.
		result, err := exactDeleteResult(count)
		// Assert: an unknown count is never exposed as a fabricated success.
		if count < 0 {
			if !errors.Is(err, ragy.ErrProtocol) || result.Deleted != 0 {
				t.Fatal("unknown count accepted", result, err)
			}
		} else if err != nil || int64(result.Deleted) != count {
			t.Fatal("exact count lost", result, err)
		}
	}
}
