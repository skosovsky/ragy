//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestPersistentComparisonActualRankingsAndQrels(t *testing.T) {
	// Arrange: saved embeddings, actual durable publication and fresh adapters.
	input, err := loadFixture("fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), experimentTimeout)
	defer cancel()
	// Act.
	result, err := experiment(ctx, t.TempDir(), input)
	// Assert: quality comes from retrieved artifacts, not scripted rankings.
	if err != nil {
		t.Fatal(err)
	}
	assertConfiguration(t, result)
	assertExecutionProvenance(t, result, input)
	if len(result.Samples) != 5 {
		t.Fatal("missing raw timing samples")
	}
	sample := result.Samples[0]
	if sample.Dense[0].ID != "t2" || sample.Reranked[0].ID != "t1" || sample.Reranked[0].Score != 2 ||
		sample.Reranked[1].Score != 1 ||
		sample.Reranked[2].Score != -1 {
		t.Fatal("actual oracle/ranking mismatch", sample)
	}
	if *result.Baseline.Recall != 1 || *result.Reranked.Recall != 1 || *result.Reranked.NDCG != 1 ||
		result.CandidateRecall != 1 ||
		*result.Negative.Recall != 0.5 ||
		result.NegativeCandidateRecall != 0.5 {
		t.Fatal("qrels/candidate loss mismatch")
	}
	expected := (1 + 7/math.Log2(3)) / (7 + 1/math.Log2(3))
	if math.Abs(*result.Baseline.NDCG-expected) > 1e-12 {
		t.Fatal("graded nDCG mismatch", result.Baseline.NDCG)
	}
	if result.DenseEmbeddingBytes != 24 || result.TensorEmbeddingBytes != 32 || result.DenseIndexBytes <= 0 ||
		result.TensorIndexBytes <= 0 ||
		result.ModelCalls != 0 ||
		result.P95 != nil ||
		result.P50 != nil ||
		result.RecommendDefault {
		t.Fatal("fabricated sizes/usage/percentiles/recommendation")
	}
	for _, row := range result.Samples {
		if row.DenseNanos <= 0 || row.CandidateRerankNanos <= 0 {
			t.Fatal("missing measured latency")
		}
	}
}

func TestMetricEmptyJudgmentsAndCandidateLoss(t *testing.T) {
	// Arrange.
	positive, zero := 2, 0
	input := fixture{
		TopK:      10,
		Documents: []fixtureDoc{{ID: "relevant", Grade: &positive}, {ID: "irrelevant", Grade: &zero}},
	}
	// Act.
	quality := metrics([]hit{{ID: "irrelevant"}}, input)
	empty := metrics(nil, fixture{TopK: 10})
	// Assert.
	if quality.Recall == nil || quality.NDCG == nil || *quality.Recall != 0 || *quality.NDCG != 0 ||
		empty.Recall != nil ||
		empty.NDCG != nil {
		t.Fatal("missing relevant evidence was rewarded", quality, empty)
	}
}

func assertConfiguration(t *testing.T, result report) {
	t.Helper()
	if result.DenseSpace.Model != savedModel || result.TensorSpace.Dimension != 2 || result.DensePublication == "" ||
		result.TensorPublication == "" ||
		!result.QualityGate ||
		result.NDCGGain < minimumNDCGGain {
		t.Fatal("declared configuration/quality gate lost")
	}
}

func TestAbsentQrelsAreNotZeroRelevance(t *testing.T) {
	// Arrange.
	input, err := loadFixture("fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	input.Documents[0].Grade = nil
	// Act.
	err = validateFixture(input)
	ungraded := metrics([]hit{{ID: "t1"}}, input)
	// Assert: absence is rejected for this fixed-qrels experiment, not silently grade 0.
	if err == nil || ungraded.Recall != nil || ungraded.NDCG != nil {
		t.Fatal("missing judgments became zero quality")
	}
}

func TestFixtureByteLimitIncludesTrailingWhitespace(t *testing.T) {
	// Arrange: a valid prefix below the limit, followed by excess whitespace.
	data, err := os.ReadFile("fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "oversized.json")
	data = append(data, bytes.Repeat([]byte(" "), maxFixtureBytes)...)
	if err = os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	// Act.
	_, err = loadFixture(path)
	// Assert: an artificial EOF at the byte limit cannot validate a larger file.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal(err)
	}
}

func TestExecutionIdentityBindsSavedDataAndControls(t *testing.T) {
	// Arrange: actual fixture plus a separately owned modified query vector.
	input, err := loadFixture("fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	config, id, err := configuredExecution(input)
	if err != nil {
		t.Fatal(err)
	}
	modified := input
	modified.Query.Dense = slices.Clone(input.Query.Dense)
	modified.Query.Dense[0]++
	// Act.
	changed, changedID, err := configuredExecution(modified)
	// Assert: identities describe actual data and stable no-random-generation policy.
	if err != nil || id == changedID || config.FixtureSHA256 == changed.FixtureSHA256 ||
		config.SeedPolicy != "saved-hand-defined-data-no-random-generator" || config.TopK != 10 || config.CandidateBudget != 100 || config.Repetitions != 5 {
		t.Fatal(config, id, changed, changedID, err)
	}
}

func assertExecutionProvenance(t *testing.T, result report, input fixture) {
	t.Helper()
	expectedConfig, expectedID, configErr := configuredExecution(input)
	if configErr != nil || result.Configuration != expectedConfig || result.ConfigIdentity != expectedID ||
		result.DenseScope.Identity == "" || result.TensorScope.Identity == "" {
		t.Fatal(result.Configuration, result.ConfigIdentity, configErr)
	}
	for _, row := range result.Samples {
		for _, item := range row.Dense {
			if item.Reference != reference(item.ID, "dense-vector") {
				t.Fatal(item)
			}
		}
		for _, item := range row.Reranked {
			if item.Reference != reference(item.ID, "token-matrix") {
				t.Fatal(item)
			}
		}
	}
}
