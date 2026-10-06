package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"example.com/ragyconsumer/internal/codexcall"

	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

const (
	executionErrorStop       = "execution-error"
	cliAttemptDuration       = 120 * time.Second
	cliCaptureDuration       = 30 * time.Minute
	cliCallInputReservation  = 20000
	cliCallOutputReservation = 1024
	cliModelCalls            = 3
	cliVersionDeadline       = 3 * time.Second
)

type codexHostProfile struct {
	Kind                       string `json:"kind"`
	CLIVersion                 string `json:"cli_version"`
	Model                      string `json:"model"`
	CalibrationSHA256          string `json:"calibration_sha256"`
	AttemptNanos               int64  `json:"attempt_duration_nanos"`
	CallNanos                  int64  `json:"call_duration_nanos"`
	TokenPolicy                string `json:"token_policy"`
	PricePolicy                string `json:"price_policy"`
	HardTokenBoundVerified     bool   `json:"hard_token_bound_verified"`
	ProviderDispatchCountKnown bool   `json:"provider_dispatch_count_known"`
	ToolIsolation              string `json:"tool_isolation"`
}

func cliConfiguration() experimentConfiguration {
	value := referenceConfiguration()
	value.AttemptNanos, value.CaptureNanos = int64(cliAttemptDuration), int64(cliCaptureDuration)
	value.InputCap, value.OutputCap = cliCallInputReservation*cliModelCalls, cliCallOutputReservation*cliModelCalls
	value.PerCallInput, value.PerCallOutput, value.PerCallCost = cliCallInputReservation, cliCallOutputReservation, 0
	value.CostCap = 0
	value.ModelLimits[3] = cliModelCalls
	return value
}
func supportedCaptureConfiguration(raw capture) bool {
	if raw.HostProfile == nil {
		return raw.Configuration == referenceConfiguration()
	}
	h := raw.HostProfile
	digest, err := hex.DecodeString(h.CalibrationSHA256)
	if err != nil || len(digest) != 32 {
		return false
	}
	return raw.Configuration == cliConfiguration() && h.Kind == "codex-cli-calibrated" &&
		h.Model == raw.ModelIdentity &&
		h.CLIVersion != "" &&
		len(h.CalibrationSHA256) == 64 &&
		h.AttemptNanos == int64(cliAttemptDuration) &&
		h.CallNanos == int64(calibrationCallDuration) &&
		h.TokenPolicy == "advisory-reported-usage" &&
		h.PricePolicy == "advisory-unknown-provider-price" &&
		!h.HardTokenBoundVerified &&
		!h.ProviderDispatchCountKnown &&
		h.ToolIsolation == "disabled-supported-tools-reject-observed-tools-additional-context-unverified"
}
func captureCodexFile(ctx context.Context, path, program, model, calibrationPath string) error {
	if path == "" || !filepath.IsAbs(program) || model == "" || calibrationPath == "" {
		return errInvalid
	}
	calibration, err := readCodexCalibration(calibrationPath, program, model)
	if err != nil {
		return err
	}
	version, err := codexVersion(ctx, program)
	if err != nil {
		return err
	}
	profile := codexHostProfile{
		Kind:              "codex-cli-calibrated",
		CLIVersion:        version,
		Model:             model,
		CalibrationSHA256: digestConfigurationBytes(calibration),
		AttemptNanos: int64(
			cliAttemptDuration,
		),
		CallNanos:     int64(calibrationCallDuration),
		TokenPolicy:   "advisory-reported-usage",
		PricePolicy:   "advisory-unknown-provider-price",
		ToolIsolation: "disabled-supported-tools-reject-observed-tools-additional-context-unverified",
	}
	raw := capture{
		HostProfile:       &profile,
		Configuration:     cliConfiguration(),
		FixtureIdentity:   "",
		AdapterIdentity:   "ragy-scoped-readonly-bm25+codex-consumer-ports",
		ModelIdentity:     model,
		TokenizerIdentity: "unavailable-cli-advisory",

		ExecutionKind: liveExecution,
	}
	raw.ConfigIdentity = captureConfigurationIdentity(raw)
	// Freeze the complete host/config identity before any comparative sample.
	manifest := struct {
		Profile       codexHostProfile        `json:"host_profile"`
		Configuration experimentConfiguration `json:"configuration"`
	}{profile, raw.Configuration}
	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(path+".profile.json", append(manifestBytes, '\n'), 0o600); err != nil {
		return err
	}
	whole, cancel := context.WithTimeout(ctx, cliCaptureDuration)
	defer cancel()
	corpus, err := newCaptureCorpusLifetime(whole, cliCaptureDuration)
	if err != nil {
		return err
	}
	raw.FixtureIdentity = corpus.fixture.Identity
	cfg := codexcall.Config{Program: program, Model: model, CallDeadline: calibrationCallDuration}
	for _, strategy := range profiles() {
		for _, q := range corpus.fixture.Queries {
			sample, sampleErr := captureCodexSample(whole, cfg, corpus, strategy, q)
			raw.Samples = append(raw.Samples, sample)
			if err = saveCodexCapture(path, raw); err != nil {
				return err
			}
			if sampleErr != nil {
				return sampleErr
			}
		}
	}
	return nil
}
func codexVersion(parent context.Context, program string) (string, error) {
	ctx, cancel := context.WithTimeout(parent, cliVersionDeadline)
	defer cancel()

	output, err := exec.CommandContext(ctx, program, "--version").Output()
	if err != nil || len(output) > maxCounterResponse || strings.TrimSpace(string(output)) == "" {
		return "", errInvalid
	}
	return strings.TrimSpace(string(output)), nil
}
func saveCodexCapture(path string, raw capture) error {
	data, err := json.MarshalIndent(raw, "", "  ")
	if err != nil || len(data) > maxInputBytes {
		return errInvalid
	}
	return os.WriteFile(path, append(data, '\n'), 0o600)
}

func captureCodexSample(
	parent context.Context,
	cfg codexcall.Config,
	corpus captureCorpus,
	strategy string,
	q query,
) (observation, error) {
	ctx, cancel := context.WithTimeout(parent, cliAttemptDuration)
	defer cancel()
	started := time.Now()
	request := retrieval.Query[struct{}]{
		Read:    corpus.read,
		Text:    q.Text,
		Options: retrieval.RetrieveOptions{TopK: topK},
	}
	sample := observation{
		Query:       q.ID,
		Strategy:    strategy,
		Scope:       corpus.read.Snapshot().Identity,
		Publication: corpus.read.Publication().Reference(),
	}
	backend := &sampleBackend{index: corpus.index}
	if strategy == baselineProfile {
		return captureBaseline(ctx, backend, request, sample, started)
	}
	ports := &codexPorts{config: cfg, strategy: recipe.Strategy(strategy)}
	controls := comparisonRecipeConfig(
		backend,
		liveModelPorts{planner: ports, assessor: ports},
		recipe.Strategy(strategy),
		&sample,
	)
	configureCodexRecipe(&controls, strategy)
	instance, err := recipe.New(controls)
	if err != nil {
		return sample, err
	}
	result, runErr := instance.RunOwnObserved(ctx, request)
	sample.CLIReceipts = ports.receipts
	sample.ModelCalls = uint64(len(ports.receipts))
	sample.RetrievalCalls = backend.calls
	sample.ReportedTokensKnown = true
	sample.ProviderPriceState = "unavailable"
	// Unknown price stays unknown. Raw reported tokens remain independent of the
	// advisory ledger's conservatively held reservations; no price is inferred.
	for _, call := range ports.receipts {
		if call.Usage == nil {
			sample.ReportedTokensKnown = false
			continue
		}
		sample.InputTokens += call.Usage.Input
		sample.OutputTokens += call.Usage.Output
	}
	sample.Outcome, sample.Stop = string(result.Outcome), string(result.Stop)
	sample.Nanos = time.Since(started).Nanoseconds()
	if runErr != nil {
		sample.Failed = true
		sample.Outcome = failedOutcome
		sample.Stop = executionErrorStop
		return sample, nil
	}
	if err = corpus.read.Check(ctx); err != nil {
		return sample, err
	}
	for _, selected := range result.Selected {
		sample.IDs = append(sample.IDs, selected.Document.ID)
		for _, loc := range selected.Document.SourceLocations() {
			sample.SourceRefs = append(sample.SourceRefs, loc.Reference)
		}
	}
	return sample, nil
}

func configureCodexRecipe(
	config *recipe.Config[struct{}, retrieval.NoRequestMeta, comparisonMetadata],
	strategy string,
) {
	config.Revision = "task12-text-codex-advisory"
	config.Duration = cliAttemptDuration
	config.RequireKnownCost = false
	config.Limits.Usage = budget.Usage{
		InputTokens:  cliCallInputReservation * cliModelCalls,
		OutputTokens: cliCallOutputReservation * cliModelCalls,
	}
	if strategy == decompositionProfile {
		config.Limits.ModelCalls = cliModelCalls
	}
	config.Pricing = func(_ context.Context, operation recipe.Operation) (recipe.Quote, error) {
		if operation == recipe.Retrieve {
			return recipe.Quote{CostKnown: true}, nil
		}
		return recipe.Quote{
			Usage:     budget.Usage{InputTokens: cliCallInputReservation, OutputTokens: cliCallOutputReservation},
			CostKnown: false,
		}, nil
	}
}

func readCodexCalibration(calibrationPath, program, model string) ([]byte, error) {
	calibration, err := os.ReadFile(calibrationPath)
	if err != nil || len(calibration) > maxInputBytes {
		return nil, errInvalid
	}
	var checked struct {
		Purpose                string             `json:"purpose"`
		Config                 codexcall.Config   `json:"config"`
		Calls                  []codexcall.Result `json:"calls"`
		PlanFailed             bool               `json:"plan_failed"`
		AssessFailed           bool               `json:"assess_failed"`
		MonetaryCostKnown      bool               `json:"monetary_cost_known"`
		HardTokenBoundVerified bool               `json:"hard_token_bound_verified"`
	}
	if err = decodeStrict(
		calibration,
		&checked,
	); err != nil || checked.Config.Model != model || checked.Config.Program != program ||
		checked.Purpose != "calibration-only-not-comparative-acceptance" || checked.MonetaryCostKnown || checked.HardTokenBoundVerified || checked.Config.CallDeadline != calibrationCallDuration || checked.PlanFailed || checked.AssessFailed ||
		len(checked.Calls) != 2 {
		return nil, errInvalid
	}
	for _, call := range checked.Calls {
		if !call.Success || call.Usage == nil || call.ToolActivity {
			return nil, errInvalid
		}
	}
	return calibration, nil
}

func captureConfigurationIdentity(raw capture) string {
	if raw.HostProfile == nil {
		return configurationIdentity(raw.Configuration)
	}
	encoded, err := json.Marshal(struct {
		Configuration experimentConfiguration `json:"configuration"`
		Profile       *codexHostProfile       `json:"host_profile"`
	}{raw.Configuration, raw.HostProfile})
	if err != nil {
		return ""
	}
	return digestConfigurationBytes(encoded)
}
