package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"time"

	"github.com/skosovsky/ragy/examples/conformance/internal/codexcall"

	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/retrieval"
)

const calibrationCallDuration = 45 * time.Second

func calibrateCodex(ctx context.Context, path, program, model string) error {
	if path == "" || !filepath.IsAbs(program) || model == "" {
		return errInvalid
	}
	config := codexcall.Config{Program: program, Model: model, CallDeadline: calibrationCallDuration}
	corpus, err := newCaptureCorpus(ctx)
	if err != nil {
		return err
	}
	ports := codexPorts{config: config, strategy: recipe.SingleRewrite}
	request := retrieval.Query[struct{}]{
		Read:    corpus.read,
		Text:    corpus.fixture.Queries[0].Text,
		Options: retrieval.RetrieveOptions{TopK: topK},
	}
	planned, planErr := ports.Plan(ctx, request, recipe.ModelLimits{})
	// Exercise calibration inputs through actual readonly scoped retrieval; no qrels
	// or expected IDs enter either model port.
	text := request.EffectiveText()
	if len(planned.Queries) > 0 {
		text = planned.Queries[0]
	}
	variant := request
	variant.Text = text
	docs, retrieveErr := corpus.index.Retrieve(ctx, variant)
	if retrieveErr != nil {
		return retrieveErr
	}
	_, assessErr := ports.Assess(ctx, recipe.AssessmentInput[struct{}, retrieval.NoRequestMeta, comparisonMetadata]{
		Original: request,
		Queries:  []recipe.QueryEvidence[comparisonMetadata]{{Index: 0, Text: text, Documents: docs.Documents()}},
	}, recipe.ModelLimits{})
	output := struct {
		Purpose                string             `json:"purpose"`
		Config                 codexcall.Config   `json:"config"`
		Calls                  []codexcall.Result `json:"calls"`
		PlanFailed             bool               `json:"plan_failed"`
		AssessFailed           bool               `json:"assess_failed"`
		MonetaryCostKnown      bool               `json:"monetary_cost_known"`
		HardTokenBoundVerified bool               `json:"hard_token_bound_verified"`
	}{Purpose: "calibration-only-not-comparative-acceptance", Config: config, Calls: ports.receipts, PlanFailed: planErr != nil, AssessFailed: assessErr != nil}
	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(path, append(data, '\n'), 0o600); err != nil {
		return err
	}
	if planErr != nil || assessErr != nil {
		return errInvalid
	}
	return nil
}
