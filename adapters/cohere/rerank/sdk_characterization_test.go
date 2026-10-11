package rerank

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	cohere "github.com/cohere-ai/cohere-go/v2"
)

func TestSDKL04BillingPrecision(t *testing.T) {
	// Arrange: integer above float64 exact integer range.
	var dto cohere.V2RerankResponse
	// Act: official public decoder, with no corrective envelope parser.
	err := json.Unmarshal([]byte(`{"results":[],"meta":{"billed_units":{"search_units":9007199254740993}}}`), &dto)
	// Assert: known limitation of v2.19.0, not an upstream correctness assertion.
	if err != nil || dto.Meta == nil || dto.Meta.BilledUnits == nil || dto.Meta.BilledUnits.SearchUnits == nil ||
		*dto.Meta.BilledUnits.SearchUnits != 9007199254740992 {
		t.Fatalf("SDK-L04 changed: %v %+v", err, dto.Meta)
	}
}

func TestSDKL05AbsentScalars(t *testing.T) {
	// Arrange.
	var dto cohere.V2RerankResponse
	// Act.
	err := json.Unmarshal([]byte(`{"results":[{}]}`), &dto)
	// Assert: the public DTO loses required-field absence in v2.19.0.
	if err != nil || len(dto.Results) != 1 || dto.Results[0].Index != 0 || dto.Results[0].RelevanceScore != 0 {
		t.Fatalf("SDK-L05 changed: %v %+v", err, dto.Results)
	}
}

func TestSDKL08DecodeBarrier(t *testing.T) {
	// Arrange: valid counter coexists with an SDK-invalid typed rank field.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(
			w,
			`{"results":[{"index":0,"relevance_score":"bad"}],"meta":{"billed_units":{"search_units":7}}}`,
		)
	}))
	t.Cleanup(server.Close)
	sdk := newSDK("fixture", server.URL, nil)
	// Act: real SDK local HTTP fixture, no alternate envelope parser.
	dto, err := sdk.V2.Rerank(
		context.Background(),
		&cohere.V2RerankRequest{Model: "fixture-model", Query: "query", Documents: []string{"text"}},
	)
	// Assert: no completed DTO/counter available on typed decode failure.
	if err == nil || dto != nil {
		t.Fatalf("SDK-L08 changed: %+v %v", dto, err)
	}
}
