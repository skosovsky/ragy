// Command recipe_comparison captures and evaluates text-recipe observations.
// Live capture is explicit; offline scoring performs no retrieval or model calls.
package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"

	"github.com/skosovsky/ragy/examples/conformance/internal/codexcall"

	"github.com/skosovsky/ragy/source"
)

//go:embed fixture.json
var fixtureJSON []byte

const (
	maxInputBytes        = 1 << 20
	baselineProfile      = "baseline"
	rewriteProfile       = "single-rewrite"
	multiProfile         = "multi-query"
	decompositionProfile = "decomposition"
	failedOutcome        = "failed"
	completeOutcome      = "complete"
	liveExecution        = "live-provider"
	contractExecution    = "contract-fixture"
	topK                 = 3
	normalModelCalls     = 2
	maximumRecipeCalls   = 3
)

func profiles() []string {
	return []string{baselineProfile, rewriteProfile, multiProfile, decompositionProfile}
}

type document struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}
type query struct {
	ID       string   `json:"id"`
	Text     string   `json:"text"`
	Relevant []string `json:"relevant"`
}
type fixture struct {
	Identity  string     `json:"identity"`
	Documents []document `json:"documents"`
	Queries   []query    `json:"queries"`
}

// observation is one actual execution. Empty hits are valid, missing rows are not.
// UsageKnown=false preserves unknown accounting; it cannot qualify budget acceptance.
type observation struct {
	CLIReceipts         []codexcall.Result `json:"cli_receipts,omitempty"`
	ReportedTokensKnown bool               `json:"reported_tokens_known,omitempty"`
	ProviderPriceState  string             `json:"provider_price_state,omitempty"`
	SourceRefs          []source.Reference `json:"source_refs"`
	ModelUsage          []modelUsage       `json:"model_usage"`
	Query               string             `json:"query"`
	Strategy            string             `json:"strategy"`
	IDs                 []string           `json:"ids"`
	Outcome             string             `json:"outcome"`
	Stop                string             `json:"stop"`
	Failed              bool               `json:"failed"`
	RetrievalCalls      uint64             `json:"retrieval_calls"`
	ModelCalls          uint64             `json:"model_calls"`
	InputTokens         uint64             `json:"input_tokens"`
	OutputTokens        uint64             `json:"output_tokens"`
	Cost                uint64             `json:"cost_units"`
	UsageKnown          bool               `json:"usage_known"`
	Nanos               int64              `json:"elapsed_nanos"`
	Scope               string             `json:"scope"`
	Publication         string             `json:"publication"`
}
type capture struct {
	HostProfile       *codexHostProfile       `json:"host_profile,omitempty"`
	Configuration     experimentConfiguration `json:"configuration"`
	FixtureIdentity   string                  `json:"fixture_identity"`
	AdapterIdentity   string                  `json:"adapter_identity"`
	ModelIdentity     string                  `json:"model_identity"`
	TokenizerIdentity string                  `json:"tokenizer_identity"`
	ConfigIdentity    string                  `json:"config_identity"`
	ExecutionKind     string                  `json:"execution_kind"`
	Samples           []observation           `json:"samples"`
}
type quality struct {
	Recall               float64 `json:"recall_at_3"`
	MRR                  float64 `json:"mrr_at_3"`
	NoAnswerErrors       int     `json:"no_answer_errors"`
	Failures             int     `json:"failed_executions"`
	BudgetsHonored       bool    `json:"budgets_honored"`
	NumericThresholdsMet bool    `json:"numeric_thresholds_met"`
}
type report struct {
	Fixture        fixture            `json:"fixture"`
	Capture        capture            `json:"capture"`
	Quality        map[string]quality `json:"quality"`
	DefaultProfile string             `json:"default_profile"`
	Limitation     string             `json:"limitation"`
}

var errInvalid = errors.New("invalid or incomplete experiment capture")

func main() {
	task19Corpus := flag.String("task19-corpus", "", "TASK19 versioned corpus")
	task19Split := flag.String("task19-split", "", "TASK19 split")
	task19Output := flag.String("task19-output", "", "TASK19 report")
	input := flag.String("input", "", "captured JSON observations")
	output := flag.String("output", "", "report path (stdout when empty)")
	executor := flag.String("executor", "http", "live model executor: http or codex")
	calibrationFile := flag.String("calibration", "", "successful CLI calibration artifact required before full run")
	calibrationPath := flag.String(
		"calibrate-codex",
		"",
		"calibrate real CLI planning/assessment; save full receipts, not a comparative run",
	)
	codexProgram := flag.String("codex-bin", "", "absolute path to trusted Codex CLI")
	capturePath := flag.String("capture", "", "execute actual BM25/model comparison and save capture")
	model := flag.String("model", os.Getenv("RAGY_EXPERIMENT_MODEL"), "configured live model")
	counter := flag.String(
		"counter",
		os.Getenv("RAGY_EXPERIMENT_TOKENIZER"),
		"absolute trusted tokenizer executable path",
	)
	tokenizer := flag.String(
		"tokenizer-id",
		os.Getenv("RAGY_EXPERIMENT_TOKENIZER_ID"),
		"qualified host tokenizer identity",
	)
	flag.Parse()
	if *task19Corpus != "" {
		if err := task19Command(*task19Corpus, *task19Split, *task19Output); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	if *calibrationPath != "" {
		if *capturePath != "" || *input != "" || *output != "" {
			fmt.Fprintln(os.Stderr, errInvalid)
			os.Exit(1)
		}
		if err := calibrateCodex(context.Background(), *calibrationPath, *codexProgram, *model); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *capturePath != "" {
		if *input != "" || *output != "" {
			fmt.Fprintln(os.Stderr, errInvalid)
			os.Exit(1)
		}
		var captureErr error
		switch *executor {
		case "codex":
			captureErr = captureCodexFile(context.Background(), *capturePath, *codexProgram, *model, *calibrationFile)
		case "http":
			captureErr = liveCaptureFile(context.Background(), *capturePath, *model, *counter, *tokenizer)
		default:
			captureErr = errInvalid
		}
		if captureErr != nil {
			fmt.Fprintln(os.Stderr, captureErr)
			os.Exit(1)
		}
		return
	}
	if err := evaluateFile(*input, *output); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func decodeStrict(data []byte, target any) error {
	if codexcall.UniqueJSON(data) != nil {
		return errInvalid
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return errInvalid
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		return errInvalid
	}
	return nil
}
func evaluateFile(input, output string) error {
	if input == "" {
		return errInvalid
	}
	file, err := os.Open(input)
	if err != nil {
		return err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxInputBytes+1))
	if err != nil {
		return err
	}
	if len(data) > maxInputBytes {
		return errInvalid
	}
	var raw capture
	if err = decodeStrict(data, &raw); err != nil {
		return err
	}
	result, err := evaluate(raw)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	encoded = append(encoded, '\n')
	if output != "" {
		return os.WriteFile(output, encoded, 0o600)
	}
	_, err = os.Stdout.Write(encoded)
	return err
}
func evaluate(raw capture) (report, error) {
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		return report{}, err
	}
	if raw.FixtureIdentity != f.Identity || raw.AdapterIdentity == "" || raw.ModelIdentity == "" ||
		raw.TokenizerIdentity == "" ||
		!supportedCaptureConfiguration(raw) ||
		raw.ConfigIdentity != captureConfigurationIdentity(raw) ||
		!slices.Contains([]string{liveExecution, contractExecution}, raw.ExecutionKind) {
		return report{}, errInvalid
	}
	if len(raw.Samples) != len(f.Queries)*len(profiles()) {
		return report{}, errInvalid
	}
	rows := make(map[string]observation, len(raw.Samples))
	for _, sample := range raw.Samples {
		if err := validateObservation(sample, f); err != nil {
			return report{}, err
		}
		key := sample.Strategy + "/" + sample.Query
		if _, exists := rows[key]; exists {
			return report{}, errInvalid
		}
		rows[key] = sample
	}
	values := make(map[string]quality, len(profiles()))
	for _, strategy := range profiles() {
		value, err := measure(strategy, f, rows)
		if err != nil {
			return report{}, err
		}
		values[strategy] = value
	}
	baseline := values[baselineProfile]
	for _, strategy := range profiles()[1:] {
		value := values[strategy]
		value.NumericThresholdsMet = value.Recall-baseline.Recall >= 0.10 && value.MRR >= baseline.MRR &&
			value.NoAnswerErrors <= baseline.NoAnswerErrors &&
			value.Failures == 0 &&
			baseline.Failures == 0 &&
			value.BudgetsHonored &&
			baseline.BudgetsHonored
		values[strategy] = value
	}
	limitation := "Offline scoring does not prove capture provenance, live tokenizer accuracy or completion of real-adapter experiments. Numeric thresholds alone do not authorize changing the default. Four answered queries and one no-answer query are too small for production quality or meaningful latency percentiles; retain all raw timings and usage. Cost is the declared 30-unit-per-call fixture, not provider billing."
	if raw.HostProfile != nil {
		limitation = "Actual CLI receipts preserve full reported tokens and end-to-end timings. Provider monetary price is unavailable; cost_units=0 is an unpriced ledger value, not a zero provider bill. CLI tokens are advisory and hard pre-dispatch bounds are unverified. budgets_honored evaluates the unchanged reference profile, not the CLI host profile. Supported tools are disabled and observed tool activity is rejected; additional context and provider retries cannot be proven absent. This is not an isolated retrieval comparison. No seed is supplied. Five cases do not establish production quality, meaningful percentiles or statistical significance. Default remains baseline."
	}
	return report{
		Fixture:        f,
		Capture:        raw,
		Quality:        values,
		DefaultProfile: baselineProfile,
		Limitation:     limitation,
	}, nil
}
func validateObservation(sample observation, f fixture) error {
	if !slices.Contains(profiles(), sample.Strategy) || sample.Nanos < 0 || sample.Scope == "" ||
		sample.Publication == "" ||
		sample.Outcome == "" ||
		sample.Stop == "" {
		return errInvalid
	}
	if !slices.ContainsFunc(f.Queries, func(q query) bool { return q.ID == sample.Query }) {
		return errInvalid
	}
	seen := make(map[string]bool)
	for _, id := range sample.IDs {
		if seen[id] || !slices.ContainsFunc(f.Documents, func(d document) bool { return d.ID == id }) {
			return errInvalid
		}
		seen[id] = true
	}
	if !slices.Contains([]string{completeOutcome, "partial", "insufficient", failedOutcome}, sample.Outcome) ||
		sample.Failed != (sample.Outcome == failedOutcome) {
		return errInvalid
	}
	if !validCapturedSources(sample) {
		return errInvalid
	}
	if sample.Failed && len(sample.IDs) != 0 {
		return errInvalid
	}
	return nil
}
func measure(strategy string, f fixture, rows map[string]observation) (quality, error) {
	out := quality{BudgetsHonored: true}
	answered := 0
	for _, q := range f.Queries {
		sample, exists := rows[strategy+"/"+q.ID]
		if !exists {
			return quality{}, errInvalid
		}
		if sample.Failed {
			out.Failures++
		}
		out.BudgetsHonored = out.BudgetsHonored && withinBudget(sample)
		if len(q.Relevant) == 0 {
			if len(sample.IDs) > 0 {
				out.NoAnswerErrors++
			}
			continue
		}
		answered++
		recall, mrr := rankMetrics(sample.IDs, q.Relevant)
		out.Recall += recall
		out.MRR += mrr
	}
	out.Recall /= float64(answered)
	out.MRR /= float64(answered)
	return out, nil
}
func rankMetrics(ids, relevant []string) (float64, float64) {
	found := 0
	reciprocal := 0.0
	for rank, id := range ids[:min(topK, len(ids))] {
		if slices.Contains(relevant, id) {
			found++
			if reciprocal == 0 {
				reciprocal = 1 / float64(rank+1)
			}
		}
	}
	return float64(found) / float64(len(relevant)), reciprocal
}
func withinBudget(s observation) bool {
	if !s.UsageKnown || s.Nanos > 5_000_000_000 || s.Cost > 100 || s.InputTokens > 2048 || s.OutputTokens > 512 {
		return false
	}
	if s.Strategy == baselineProfile {
		return s.RetrievalCalls <= 1 && s.ModelCalls == 0 && s.InputTokens == 0 && s.OutputTokens == 0 && s.Cost == 0
	}
	modelLimit, retrievalLimit := uint64(normalModelCalls), uint64(maximumRecipeCalls)
	if s.Strategy == rewriteProfile {
		retrievalLimit = 2
	}
	if s.Strategy == decompositionProfile {
		modelLimit = 3
	}
	return s.ModelCalls <= modelLimit && s.RetrievalCalls <= retrievalLimit
}

func validCapturedSources(sample observation) bool {
	for _, ref := range sample.SourceRefs {
		if ref != corpusReference(ref.Artifact) || !slices.Contains(sample.IDs, ref.Artifact) {
			return false
		}
	}
	for _, id := range sample.IDs {
		if !slices.Contains(sample.SourceRefs, corpusReference(id)) {
			return false
		}
	}
	return true
}
