//go:build darwin || linux

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

// Saved data records an explicit no-random-generation policy and binds the
// complete normalized corpus/qrels. No model sampling occurs.
type executionConfiguration struct {
	SeedPolicy             string  `json:"seed_policy"`
	FixtureSHA256          string  `json:"fixture_sha256"`
	DenseAdapter           string  `json:"dense_adapter"`
	TensorAdapter          string  `json:"tensor_adapter"`
	DenseScoring           string  `json:"dense_scoring"`
	TensorScoring          string  `json:"tensor_scoring"`
	TopK                   int     `json:"top_k"`
	CandidateBudget        int     `json:"candidate_budget"`
	Repetitions            int     `json:"repetitions"`
	DeadlineNanos          int64   `json:"deadline_nanos"`
	MaxRecords             int     `json:"max_records"`
	MaxScanRecords         int     `json:"max_scan_records"`
	MaxTargetFileBytes     int     `json:"max_target_file_bytes"`
	MaxManifestBytes       int     `json:"max_manifest_bytes"`
	MaxFixtureBytes        int     `json:"max_fixture_bytes"`
	FloatBytes             int     `json:"float_bytes"`
	MinimumNDCGGain        float64 `json:"minimum_ndcg_gain"`
	MinimumCandidateRecall float64 `json:"minimum_candidate_recall"`
	NegativeCandidate      string  `json:"negative_removed_candidate"`
}

func configuredExecution(input fixture) (executionConfiguration, string, error) {
	corpus, err := json.Marshal(input)
	if err != nil {
		return executionConfiguration{}, "", err
	}
	digest := sha256.Sum256(corpus)
	config := executionConfiguration{
		SeedPolicy:      "saved-hand-defined-data-no-random-generator",
		FixtureSHA256:   hex.EncodeToString(digest[:]),
		DenseAdapter:    "ragy/dense/persistent/local-filesystem",
		TensorAdapter:   "ragy/tensor/persistent/local-filesystem",
		DenseScoring:    "native-dot",
		TensorScoring:   "native-maxsim",
		TopK:            input.TopK,
		CandidateBudget: input.CandidateBudget,
		Repetitions:     input.Repetitions,
		DeadlineNanos: int64(
			experimentTimeout,
		),
		MaxRecords:             targetLimit,
		MaxScanRecords:         targetLimit,
		MaxTargetFileBytes:     targetFileBytes,
		MaxManifestBytes:       manifestBytes,
		MaxFixtureBytes:        maxFixtureBytes,
		FloatBytes:             float32Bytes,
		MinimumNDCGGain:        minimumNDCGGain,
		MinimumCandidateRecall: minimumCandidateRecall,
		NegativeCandidate:      "t1",
	}
	encoded, err := json.Marshal(config)
	if err != nil {
		return executionConfiguration{}, "", err
	}
	digest = sha256.Sum256(encoded)
	return config, hex.EncodeToString(digest[:]), nil
}
