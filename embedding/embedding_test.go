package embedding_test

import (
	"errors"
	"math"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
)

func profile(metric embedding.Metric) embedding.Space {
	return embedding.Space{
		Model:         "host-model",
		ModelRevision: "immutable-v1",
		Configuration: "host-tokenizer-v2",
		VectorSpace:   "retrieval-pair",
		Dimension:     2,
		Metric:        metric,
	}
}
func TestExplicitMetricConstraints(t *testing.T) {
	tests := []struct {
		name      string
		metric    embedding.Metric
		vector    []float32
		wantError bool
	}{
		{"raw dot keeps nonunit", embedding.Dot, []float32{3, 4}, false},
		{"L2 keeps zero", embedding.SquaredL2, []float32{0, 0}, false},
		{"cosine allows nonunit", embedding.Cosine, []float32{3, 4}, false},
		{"cosine rejects zero", embedding.Cosine, []float32{0, 0}, true},
		{"unit requirement explicit", embedding.NormalizedDot, []float32{3, 4}, true},
		{"unit vector", embedding.NormalizedDot, []float32{1, 0}, false},
		{"NaN", embedding.Dot, []float32{float32(math.NaN()), 1}, true},
		{"Inf", embedding.Dot, []float32{1, float32(math.Inf(1))}, true},
		{"dimension", embedding.Dot, []float32{1}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			space := profile(tt.metric)
			err := space.ValidateVector(tt.vector)
			if (err != nil) != tt.wantError {
				t.Fatalf("ValidateVector()=%v, wantError=%v", err, tt.wantError)
			}
		})
	}
}
func TestUsageDoesNotFabricateKnownCounters(t *testing.T) {
	tests := []struct {
		name      string
		usage     embedding.Usage
		wantError bool
	}{
		{"unknown", embedding.Usage{}, false},
		{"known zero", embedding.Usage{InputTokensKnown: true}, false},
		{
			"observed",
			embedding.Usage{InputTokens: 12, InputTokensKnown: true, BilledUnits: 1, BilledUnitsKnown: true},
			false,
		},
		{"unknown with value", embedding.Usage{InputTokens: 12}, true},
		{"negative", embedding.Usage{InputTokens: -1, InputTokensKnown: true}, true},
		{"unknown billing with value", embedding.Usage{BilledUnits: 2}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.usage.Validate()
			if (err != nil) != tt.wantError {
				t.Fatalf("Validate()=%v", err)
			}
		})
	}
}
func TestHostEncodingCanImplementStrictRemoteBound(t *testing.T) {
	request := embedding.Request[string]{
		Inputs:                  []string{"bounded host input"},
		Purpose:                 embedding.Query,
		RequireRemoteTokenBound: true,
	}
	err := request.Validate()
	if err != nil {
		t.Fatalf("core rejected capable host encoder request: %v", err)
	}
	request.Purpose = "arbitrary"
	if err = request.Validate(); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatalf("unsupported purpose=%v", err)
	}
}
