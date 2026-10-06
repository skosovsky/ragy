package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

const (
	comparisonK1         = 1.2
	comparisonB          = 0.75
	plannerInstructions  = "Produce bounded search text variants for the stated strategy. Single rewrite returns one query; multi-query returns at most two; decomposition returns at most three independent subquestions. Preserve the original information need. Do not answer the question or invent facts."
	assessorInstructions = "Select executed query indices whose snippets support the original information need. Consider original evidence as well as variants. Set sufficient false when evidence cannot answer all requested parts. Reject evidence for an unrelated topic. Return no selected indices if there is no relevant evidence. Do not answer the question."
)

// Fixed configuration describes this consumer's actual reference execution.
// No provider seed is sent: uncontrolled sampling is explicit, not reproducibility.
type experimentConfiguration struct {
	SeedPolicy         string    `json:"seed_policy"`
	CorpusSHA256       string    `json:"corpus_sha256"`
	PlannerSHA256      string    `json:"planner_sha256"`
	AssessorSHA256     string    `json:"assessor_sha256"`
	BM25K1             float64   `json:"bm25_k1"`
	BM25B              float64   `json:"bm25_b"`
	TopK               int       `json:"top_k"`
	FusionK            int       `json:"fusion_k"`
	InputCap           uint64    `json:"input_token_cap"`
	OutputCap          uint64    `json:"output_token_cap"`
	CostCap            uint64    `json:"cost_unit_cap"`
	PerCallInput       uint64    `json:"per_call_input_tokens"`
	PerCallOutput      uint64    `json:"per_call_output_tokens"`
	PerCallCost        uint64    `json:"per_model_call_cost_units"`
	AttemptNanos       int64     `json:"attempt_duration_nanos"`
	CaptureNanos       int64     `json:"capture_duration_nanos"`
	CounterNanos       int64     `json:"counter_duration_nanos"`
	MaxRequestBytes    int       `json:"max_request_bytes"`
	MaxResponseBytes   int       `json:"max_response_bytes"`
	MaxQueryBytes      int       `json:"max_query_bytes"`
	MaxAssessmentBytes int       `json:"max_assessment_bytes"`
	MaxDocuments       int       `json:"max_documents"`
	QueryLimits        [4]int    `json:"query_limits_baseline_rewrite_multi_decomposition"`
	RetrievalLimits    [4]uint64 `json:"retrieval_limits_baseline_rewrite_multi_decomposition"`
	ModelLimits        [4]uint64 `json:"model_limits_baseline_rewrite_multi_decomposition"`
}

func digestConfigurationBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
func referenceConfiguration() experimentConfiguration {
	return experimentConfiguration{
		SeedPolicy: "provider-default-no-seed-requested", CorpusSHA256: digestConfigurationBytes(fixtureJSON),
		PlannerSHA256:  digestConfigurationBytes([]byte(plannerInstructions + "\n" + plannerSchema)),
		AssessorSHA256: digestConfigurationBytes([]byte(assessorInstructions + "\n" + assessorSchema)),
		BM25K1:         comparisonK1, BM25B: comparisonB, TopK: topK, FusionK: fusionK,
		InputCap: referenceInputCap, OutputCap: referenceOutputCap, CostCap: referenceCostCap,
		PerCallInput: perCallInput, PerCallOutput: perCallOutput, PerCallCost: fixtureCallCost,
		AttemptNanos: int64(attemptDuration), CaptureNanos: int64(captureDuration), CounterNanos: int64(counterTimeout),
		MaxRequestBytes: maxCounterRequest, MaxResponseBytes: maxInputBytes, MaxQueryBytes: maximumQueryBytes,
		MaxAssessmentBytes: maximumAssessmentBytes, MaxDocuments: maximumAssessmentDocuments,
		QueryLimits:     [4]int{0, 1, normalModelCalls, maximumAssessmentQueries},
		RetrievalLimits: [4]uint64{1, normalModelCalls, maximumRecipeCalls, maximumRecipeCalls},
		ModelLimits:     [4]uint64{0, normalModelCalls, normalModelCalls, normalModelCalls},
	}
}
func configurationIdentity(config experimentConfiguration) string {
	encoded, err := json.Marshal(config)
	if err != nil {
		return ""
	}
	return digestConfigurationBytes(encoded)
}
