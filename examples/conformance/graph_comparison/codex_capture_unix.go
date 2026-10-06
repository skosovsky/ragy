//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"example.com/ragyconsumer/internal/codexcall"

	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

type codexCalibration struct {
	Purpose string             `json:"purpose"`
	Config  codexcall.Config   `json:"config"`
	Calls   []codexcall.Result `json:"calls"`
	Failed  bool               `json:"failed"`
}

func calibrateCodex(ctx context.Context, path, program, model string) error {
	if path == "" || !filepath.IsAbs(program) || model == "" {
		return errInvalid
	}
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		return err
	}
	ports := codexGraphPorts{config: codexcall.Config{Program: program, Model: model, CallDeadline: cliCallDuration}}
	_, _, extractionErr := ports.extractionPorts().
		model(ctx, extraction.ModelInput{
			OntologyIdentity: referenceConfiguration().Ontology,
			Configuration:    extractionConfigurationIdentity(),
			Snippets:         []extraction.ModelSnippet{{Index: 0, Text: f.Sources[0].Text}},
			MaxInputTokens:   cliInputReservation,
			MaxOutputTokens:  cliOutputReservation,
		})
	_, _, summaryErr := ports.summaryPorts().
		model(ctx, graphsummary.ModelInput{
			Stage:    graphsummary.Map,
			Question: f.Queries[1].Text,
			Snippets: []graphsummary.ModelSnippet{
				{Index: 0, Text: f.Sources[0].Text},
			},
			MaxInputTokens:  cliInputReservation,
			MaxOutputTokens: cliOutputReservation,
		})
	data, err := json.MarshalIndent(
		codexCalibration{Purpose: "calibration-only-not-comparative-acceptance", Config: ports.config,
			Calls: ports.receipts(), Failed: extractionErr != nil || summaryErr != nil},
		"",
		"  ",
	)
	if err != nil {
		return err
	}
	if err = os.WriteFile(path, append(data, '\n'), 0o600); err != nil {
		return err
	}
	if extractionErr != nil || summaryErr != nil {
		return errInvalid
	}
	return nil
}
func readCodexCalibration(path, program, model string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil || len(data) > maxArtifactBytes {
		return nil, errInvalid
	}
	var checked codexCalibration
	if decodeStrict(data, &checked) != nil || checked.Purpose != "calibration-only-not-comparative-acceptance" ||
		checked.Failed ||
		checked.Config.Program != program ||
		checked.Config.Model != model ||
		checked.Config.CallDeadline != cliCallDuration ||
		len(checked.Calls) != 2 {
		return nil, errInvalid
	}
	for _, call := range checked.Calls {
		if !call.Success || call.Usage == nil || call.ToolActivity {
			return nil, errInvalid
		}
	}
	return data, nil
}
func captureCodexFile(ctx context.Context, path, program, model, calibrationPath string) error {
	if path == "" || !filepath.IsAbs(program) || model == "" {
		return errInvalid
	}
	calibration, err := readCodexCalibration(calibrationPath, program, model)
	if err != nil {
		return err
	}
	versionCtx, cancel := context.WithTimeout(ctx, cliVersionDuration)

	version, err := exec.CommandContext(versionCtx, program, "--version").Output()
	cancel()
	if err != nil || len(version) > 512 || strings.TrimSpace(string(version)) == "" {
		return errInvalid
	}
	profile := codexHostProfile{
		CLIVersion:        strings.TrimSpace(string(version)),
		Model:             model,
		CalibrationSHA256: digest(calibration),
		CallNanos:         int64(cliCallDuration),
		TokenPolicy:       "advisory-calibration-plus-input-byte-estimate",
		PricePolicy:       "unknown-provider-price",
		ToolIsolation:     "disabled-supported-tools-reject-observed-tools-additional-context-unverified",
	}
	controls, identity, err := cliConfigurationBytes(&profile)
	if err != nil {
		return err
	}
	manifest, err := json.MarshalIndent(struct {
		Profile       codexHostProfile `json:"host_profile"`
		Configuration json.RawMessage  `json:"configuration"`
		Identity      string           `json:"config_identity"`
	}{profile, controls, identity}, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(path+".profile.json", append(manifest, '\n'), 0o600); err != nil {
		return err
	}
	root, err := os.MkdirTemp("", "ragy-codex-graph-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(root) }()
	cfg := codexcall.Config{Program: program, Model: model, CallDeadline: cliCallDuration}
	factories := captureFactories{
		extraction: func(context.Context) (extractionPorts, error) {
			p := &codexGraphPorts{config: cfg}
			return p.extractionPorts(), nil
		},
		summary: func(context.Context) (summaryModelPorts, error) {
			p := &codexGraphPorts{config: cfg}
			return p.summaryPorts(), nil
		},
	}
	raw, runErr := executeGraphCapture(
		ctx,
		root,
		captureIdentity{execution: liveExecution, adapter: "persistent-dense+owned-bm25+managed-graph+codex-consumer",
			model: model, tokenizer: "unavailable-cli-advisory", profile: &profile},
		factories,
	)
	if raw.FixtureIdentity != "" {
		if err = saveCapture(path, raw); err != nil {
			return err
		}
	}
	return runErr
}
