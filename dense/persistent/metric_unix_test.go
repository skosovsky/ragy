//go:build darwin || linux

package persistent_test

import (
	"context"
	"strings"
	"testing"

	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

func TestPersistentProfileScoreRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		metric          embedding.Metric
		document, query []float32
		score           float64
	}{
		{embedding.NormalizedDot, []float32{1, 0}, []float32{1, 0}, 1},
		{embedding.Dot, []float32{3, 0}, []float32{2, 0}, 6},
		{embedding.Cosine, []float32{3, 0}, []float32{2, 0}, 1},
		{embedding.SquaredL2, []float32{3, 0}, []float32{2, 0}, -1},
	} {
		t.Run(string(tc.metric), func(t *testing.T) {
			// Arrange: persist an explicit profile with raw vectors intact.
			config := newConfig(t)
			config.Space.Metric = tc.metric
			input := records()[:1]
			input[0].Value.Space = config.Space
			input[0].Value.Vector = tc.document
			adapter := published(t, config, input)
			request := query(pin(t, config), input)
			request.Intent.Embedding = dense.Embedding{Space: config.Space, Vector: tc.query}
			// Act.
			result, err := adapter.Retrieve(context.Background(), request)
			// Assert.
			if err != nil || result.Len() != 1 {
				t.Fatalf("result=%v err=%v", result, err)
			}
			doc := result.Documents()[0]
			if doc.Score != tc.score ||
				!strings.HasPrefix(string(doc.ScoreSemantics), dense.ScoreSemantics(tc.metric)+":") {
				t.Fatalf("wrong score/semantics %+v", doc)
			}
			if adapter.QueryCapabilities().Space != config.Space {
				t.Fatal("lost profile")
			}
		})
	}
}
