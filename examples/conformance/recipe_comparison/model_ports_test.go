package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/retrieval"
)

func TestModelPortsUseActualTransportAndHostCounter(t *testing.T) {
	// Arrange: HTTP protocol and local counter fixtures; no live model quality claim.
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, maxCounterRequest+1))
		if err != nil || len(body) > maxCounterRequest || strings.Contains(string(body), "fixture-key") {
			t.Error("invalid model request")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		calls.Add(1)
		content := `{"selected":[0],"sufficient":true}`
		if strings.Contains(string(body), "retrieval_planning") {
			content = `{"queries":["возврат оплаты срок"]}`
		}
		encoded, err := json.Marshal(content)
		if err != nil {
			t.Error(err)
			return
		}
		_, _ = w.Write(
			[]byte(
				`{"choices":[{"index":0,"message":{"role":"assistant","content":` + string(
					encoded,
				) + `},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`,
			),
		)
	}))
	defer server.Close()
	ctx, cancel := context.WithTimeout(t.Context(), attemptDuration)
	defer cancel()
	ports, err := newLiveModelPorts(
		ctx,
		liveModelConfig{
			apiKey:  "fixture-key",
			baseURL: server.URL,
			client:  server.Client(),
			counter: counterFixture(t, "valid"),
		},
		recipe.SingleRewrite,
	)
	if err != nil {
		t.Fatal(err)
	}
	request := retrieval.Query[struct{}]{Read: access.Unrestricted(), Text: "Когда вернут деньги?"}
	limits := recipe.ModelLimits{InputTokens: 1024, OutputTokens: 256}
	// Act: actual structured transport and executable token counter for both stages.
	planning, err := ports.planner.Plan(ctx, request, limits)
	if err != nil {
		t.Fatal(err)
	}
	assessment, err := ports.assessor.Assess(
		ctx,
		recipe.AssessmentInput[struct{}, retrieval.NoRequestMeta, comparisonMetadata]{
			Original: request,
			Queries: []recipe.QueryEvidence[comparisonMetadata]{
				{
					Index: 0,
					Text:  request.Text,
					Documents: []retrieval.Document[comparisonMetadata]{
						{
							ID:      "d1",
							Content: "Возврат оплаты за заказ: срок 10 дней.",
							Meta:    comparisonMetadata{Tenant: "a"},
						},
					},
				},
			},
		},
		limits,
	)
	// Assert: exact known fixture usage and one dispatch per stage, without retry.
	if err != nil || len(planning.Queries) != 1 || len(assessment.Selected) != 1 || !assessment.Sufficient ||
		calls.Load() != 2 {
		t.Fatal(planning, assessment, err, calls.Load())
	}
	for _, usage := range []recipe.Usage{planning.Usage, assessment.Usage} {
		if !usage.Known || usage.Value.InputTokens != 20 || usage.Value.OutputTokens != 5 ||
			usage.Value.Cost != fixtureCallCost {
			t.Fatal(usage)
		}
	}
}
func TestHostSchemaValidatorsRejectMissingNullAndUnknownFields(t *testing.T) {
	for _, data := range []string{`{}`, `{"queries":null}`, `{"queries":[],"unexpected":true}`} {
		if validatePlannerOutput([]byte(data)) == nil {
			t.Fatal("invalid planning schema accepted", data)
		}
	}
	for _, data := range []string{`{}`, `{"selected":[],"sufficient":null}`, `{"selected":null,"sufficient":false}`, `{"selected":[],"sufficient":false,"unexpected":true}`} {
		if validateAssessorOutput([]byte(data)) == nil {
			t.Fatal("invalid assessment schema accepted", data)
		}
	}
	if validatePlannerOutput([]byte(`{"queries":[]}`)) != nil ||
		validateAssessorOutput([]byte(`{"selected":[],"sufficient":false}`)) != nil {
		t.Fatal("valid empty/insufficient model output rejected")
	}
	if _, err := newLiveModelPorts(context.Background(), liveModelConfig{}, recipe.SingleRewrite); err == nil {
		t.Fatal("missing host attempt deadline accepted")
	}
}
