package main

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/skosovsky/ragy/adapters/openai/structured"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/retrieval"
)

const (
	fixtureCallCost            = 30
	attemptDuration            = 5 * time.Second
	maximumQueryBytes          = 1024
	maximumAssessmentQueries   = 3
	maximumAssessmentDocuments = 9
	maximumAssessmentBytes     = 8 << 10
)

type liveModelConfig struct {
	apiKey  string
	baseURL string
	client  *http.Client
	counter hostCounter
}
type liveModelPorts struct {
	planner  planningPort
	assessor assessmentPort
}
type planningPort interface {
	Plan(
		context.Context,
		retrieval.Request[struct{}, retrieval.NoRequestMeta],
		recipe.ModelLimits,
	) (recipe.Planning, error)
}
type assessmentPort interface {
	Assess(
		context.Context,
		recipe.AssessmentInput[struct{}, retrieval.NoRequestMeta, comparisonMetadata],
		recipe.ModelLimits,
	) (recipe.Assessment, error)
}

type comparisonMetadata struct {
	Tenant string `json:"tenant"`
}

const plannerSchema = `{"type":"object","properties":{"queries":{"type":"array","items":{"type":"string"}}},"required":["queries"],"additionalProperties":false}`

const assessorSchema = `{"type":"object","properties":{"selected":{"type":"array","items":{"type":"integer"}},"sufficient":{"type":"boolean"}},"required":["selected","sufficient"],"additionalProperties":false}`

// newLiveModelPorts requires the captured host attempt deadline before binding the
// context-free tokenizer callback. Its parent deadline bounds tokenization too.
// Credentials, qualification and model selection are explicit host configuration.
func newLiveModelPorts(ctx context.Context, cfg liveModelConfig, strategy recipe.Strategy) (liveModelPorts, error) {
	if ctx == nil {
		return liveModelPorts{}, errCounter
	}
	if _, present := ctx.Deadline(); !present {
		return liveModelPorts{}, errCounter
	}
	common := structured.Config{
		APIKey:           cfg.apiKey,
		Model:            cfg.counter.model,
		BaseURL:          cfg.baseURL,
		HTTPClient:       cfg.client,
		MaxRequestBytes:  maxCounterRequest,
		MaxResponseBytes: maxInputBytes,
		Duration:         attemptDuration,
		CountTokens:      func(request []byte) (uint64, error) { return cfg.counter.count(ctx, request) },
	}
	plannerConfig := common
	plannerConfig.Instructions = plannerInstructions
	plannerConfig.SchemaName = "retrieval_planning"
	plannerConfig.Schema = json.RawMessage(plannerSchema)
	plannerConfig.Validate = validatePlannerOutput
	planner, err := structured.NewPlanner[struct{}, retrieval.NoRequestMeta](
		plannerConfig,
		strategy,
		maximumQueryBytes,
		fixtureModelPrice,
	)
	if err != nil {
		return liveModelPorts{}, err
	}
	assessorConfig := common
	assessorConfig.Instructions = assessorInstructions
	assessorConfig.SchemaName = "retrieval_assessment"
	assessorConfig.Schema = json.RawMessage(assessorSchema)
	assessorConfig.Validate = validateAssessorOutput
	assessor, err := structured.NewAssessor[struct{}, retrieval.NoRequestMeta, comparisonMetadata](
		assessorConfig,
		maximumAssessmentQueries,
		maximumAssessmentDocuments,
		maximumAssessmentBytes,
		fixtureModelPrice,
	)
	if err != nil {
		return liveModelPorts{}, err
	}
	return liveModelPorts{planner: planner, assessor: assessor}, nil
}
func fixtureModelPrice(structured.Usage) (uint64, error) { return fixtureCallCost, nil }
func validatePlannerOutput(data json.RawMessage) error {
	var output structured.PlannerOutput
	if err := decodeStrict(data, &output); err != nil || output.Queries == nil {
		return errInvalid
	}
	return nil
}
func validateAssessorOutput(data json.RawMessage) error {
	var output struct {
		Selected   []int `json:"selected"`
		Sufficient *bool `json:"sufficient"`
	}
	if err := decodeStrict(data, &output); err != nil || output.Selected == nil || output.Sufficient == nil {
		return errInvalid
	}
	return nil
}
