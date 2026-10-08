// Command graph_comparison validates and evaluates external graph observations.
// The offline command dispatches no retrieval, extraction or model calls.
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

	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/source"
)

//go:embed fixture.json
var fixtureJSON []byte

const (
	executionErrorStop = "execution-error"
	failedOutcome      = "failed"
	completeOutcome    = "complete"
	maxArtifactBytes   = 1 << 20
	supportTopK        = 3
	baselineProfile    = "hybrid-baseline"
	localProfile       = "local-expansion"
	communityProfile   = "community-summary"
	globalProfile      = "global-summary"
	liveExecution      = "live-provider"
	contractExecution  = "contract-fixture"
)

var errInvalid = errors.New("invalid graph comparison artifact")

type sourceRow struct {
	Dense     []float32 `json:"dense"`
	ID        string    `json:"id"`
	Namespace string    `json:"namespace"`
	Text      string    `json:"text"`
}
type community struct {
	ID      string   `json:"id"`
	Members []string `json:"members"`
	Sources []string `json:"sources"`
}
type query struct {
	Dense    []float32 `json:"dense"`
	ID       string    `json:"id"`
	Recipe   string    `json:"recipe"`
	Text     string    `json:"text"`
	Relevant []string  `json:"relevant"`
}
type entity struct {
	Key      string   `json:"key"`
	Supports []string `json:"supports"`
}
type edge struct {
	From     string   `json:"from"`
	To       string   `json:"to"`
	Kind     string   `json:"kind"`
	Supports []string `json:"supports"`
}
type fixture struct {
	Identity    string      `json:"identity"`
	Sources     []sourceRow `json:"sources"`
	Communities []community `json:"communities"`
	Queries     []query     `json:"queries"`
	Entities    []entity    `json:"entities"`
	Edges       []edge      `json:"edges"`
}
type baselineMetadata struct {
	Tenant   string `json:"tenant"`
	SourceID string `json:"source_key"`
}
type observation struct {
	CLIReceipts         []codexcall.Result `json:"cli_receipts,omitempty"`
	ReportedTokensKnown bool               `json:"reported_tokens_known,omitempty"`
	ProviderPriceState  string             `json:"provider_price_state,omitempty"`
	stages              []evidence.Stage[baselineMetadata]
	Evidence            json.RawMessage    `json:"evidence,omitempty"`
	Configuration       string             `json:"configuration"`
	TransportCalls      uint64             `json:"transport_calls"`
	TransportCallsKnown bool               `json:"transport_calls_known"`
	SourceMetadataCalls uint64             `json:"source_metadata_calls"`
	SourcePayloadCalls  uint64             `json:"source_payload_calls"`
	CallsKnown          bool               `json:"calls_known"`
	Outcome             string             `json:"outcome"`
	Stop                string             `json:"stop"`
	Query               string             `json:"query"`
	Profile             string             `json:"profile"`
	Supports            []source.Reference `json:"supports"`
	Failed              bool               `json:"failed"`
	UsageKnown          bool               `json:"usage_known"`
	RetrievalCalls      uint64             `json:"retrieval_calls"`
	GraphCalls          uint64             `json:"graph_calls"`
	ModelCalls          uint64             `json:"model_calls"`
	InputTokens         uint64             `json:"input_tokens"`
	OutputTokens        uint64             `json:"output_tokens"`
	Cost                uint64             `json:"cost_units"`
	Nanos               int64              `json:"elapsed_nanos"`
	Scope               string             `json:"scope"`
	Publication         string             `json:"publication"`
}
type extractionObservation struct {
	CLIReceipts         []codexcall.Result `json:"cli_receipts,omitempty"`
	ReportedTokensKnown bool               `json:"reported_tokens_known,omitempty"`
	ProviderPriceState  string             `json:"provider_price_state,omitempty"`
	Configuration       string             `json:"configuration"`
	TransportCalls      uint64             `json:"transport_calls"`
	TransportCallsKnown bool               `json:"transport_calls_known"`
	Reference           source.Reference   `json:"reference"`
	Failed              bool               `json:"failed"`
	ModelCalls          uint64             `json:"model_calls"`
	InputTokens         uint64             `json:"input_tokens"`
	OutputTokens        uint64             `json:"output_tokens"`
	Cost                uint64             `json:"cost_units"`
	UsageKnown          bool               `json:"usage_known"`
	Nanos               int64              `json:"elapsed_nanos"`
}

type preparation struct {
	ResolutionHistoryID  string                  `json:"resolution_history_id"`
	Extractions          []extractionObservation `json:"extractions"`
	MembershipGraphCalls uint64                  `json:"membership_graph_calls"`
}
type capture struct {
	HostProfile       *codexHostProfile `json:"host_profile,omitempty"`
	Preparation       preparation       `json:"preparation"`
	FixtureIdentity   string            `json:"fixture_identity"`
	ExecutionKind     string            `json:"execution_kind"`
	AdapterIdentity   string            `json:"adapter_identity"`
	ModelIdentity     string            `json:"model_identity"`
	TokenizerIdentity string            `json:"tokenizer_identity"`
	Configuration     json.RawMessage   `json:"configuration"`
	ConfigIdentity    string            `json:"config_identity"`
	Samples           []observation     `json:"samples"`
}
type measured struct {
	BaselineRecall         float64 `json:"baseline_support_recall_at_3"`
	RecipeRecall           float64 `json:"recipe_support_recall_at_3"`
	Gain                   float64 `json:"support_recall_gain"`
	BaselineBudgetsHonored bool    `json:"baseline_budgets_honored"`
	RecipeBudgetsHonored   bool    `json:"recipe_budgets_honored"`
}
type report struct {
	PreparationBudgetsHonored bool                `json:"preparation_budgets_honored"`
	Fixture                   fixture             `json:"fixture"`
	Capture                   capture             `json:"capture"`
	Measurements              map[string]measured `json:"measurements"`
	DefaultProfile            string              `json:"default_profile"`
	Limitation                string              `json:"limitation"`
}

func main() {
	task19Corpus := flag.String("task19-corpus", "", "TASK19 versioned corpus JSON")
	task19Split := flag.String("task19-split", "", "TASK19 explicit query split JSON")
	task19Output := flag.String("task19-output", "", "TASK19 report JSON")
	input := flag.String("input", "", "complete external execution capture")
	output := flag.String("output", "", "output report; empty uses stdout")
	capturePath := flag.String("capture", "", "execute provider extraction and all graph recipes; save raw capture")
	model := flag.String("model", os.Getenv("RAGY_EXPERIMENT_MODEL"), "explicit provider model")
	tokenizer := flag.String(
		"tokenizer",
		os.Getenv("RAGY_EXPERIMENT_TOKENIZER"),
		"absolute trusted qualified tokenizer executable",
	)
	tokenizerID := flag.String(
		"tokenizer-id",
		os.Getenv("RAGY_EXPERIMENT_TOKENIZER_ID"),
		"qualified tokenizer identity",
	)
	executor := flag.String("executor", "http", "external executor: http or codex")
	program := flag.String("codex-bin", "", "absolute trusted Codex CLI executable")
	calibration := flag.String("calibration", "", "successful graph calibration artifact")
	calibrate := flag.String("calibrate-codex", "", "save calibration-only CLI responses")
	flag.Parse()
	if *task19Corpus != "" {
		if err := executeTask19Graph(*task19Corpus, *task19Split, *task19Output); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *calibrate != "" {
		if *input != "" || *output != "" || *capturePath != "" {
			fmt.Fprintln(os.Stderr, errInvalid)
			os.Exit(1)
		}
		if err := calibrateCodex(context.Background(), *calibrate, *program, *model); err != nil {
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
		case "http":
			captureErr = liveCaptureFile(context.Background(), *capturePath, *model, *tokenizer, *tokenizerID)
		case "codex":
			captureErr = captureCodexFile(context.Background(), *capturePath, *program, *model, *calibration)
		default:
			captureErr = errInvalid
		}
		if err := captureErr; err != nil {
			fmt.Fprintln(os.Stderr, err)
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
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(io.LimitReader(file, maxArtifactBytes+1))
	if err != nil {
		return err
	}
	if len(data) > maxArtifactBytes {
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
func originalReference(id string) source.Reference {
	return source.Reference{
		Namespace:         "n",
		Source:            id,
		Revision:          "v1",
		Transformation:    "original",
		AccessFingerprint: "a-public",
		Artifact:          "original-p1",
		Representation:    "utf8",
	}
}
func queryFor(f fixture, id string) (query, bool) {
	for _, q := range f.Queries {
		if q.ID == id {
			return q, true
		}
	}
	return query{}, false
}
func validateObservation(sample observation, f fixture) error {
	q, found := queryFor(f, sample.Query)
	if !validOutcome(sample) {
		return errInvalid
	}
	if !found || !slices.Contains([]string{baselineProfile, q.Recipe}, sample.Profile) || sample.Scope == "" ||
		sample.Publication == "" ||
		sample.Nanos <= 0 ||
		sample.Failed && len(sample.Supports) != 0 {
		return errInvalid
	}
	seen := make(map[source.Reference]bool)
	for _, ref := range sample.Supports {
		exists := false
		for _, row := range f.Sources {
			if ref == originalReference(row.ID) {
				exists = true
				break
			}
		}
		if !exists || seen[ref] {
			return errInvalid
		}
		seen[ref] = true
	}
	return validateEvidence(sample)
}
func recall(sample observation, q query) float64 {
	found := 0
	for _, id := range q.Relevant {
		for _, ref := range sample.Supports[:min(supportTopK, len(sample.Supports))] {
			if ref.Source == id {
				found++
				break
			}
		}
	}
	return float64(found) / float64(len(q.Relevant))
}
func budgetsHonored(sample observation) bool {
	if sample.TransportCallsKnown && sample.TransportCalls > sample.ModelCalls {
		return false
	}
	if sample.Failed || !sample.UsageKnown || !sample.CallsKnown || sample.Nanos > int64(attemptDuration) ||
		sample.InputTokens > referenceInputCap ||
		sample.OutputTokens > referenceOutputCap ||
		sample.Cost > referenceCostCap {
		return false
	}
	switch sample.Profile {
	case baselineProfile:
		return sample.RetrievalCalls <= 2 && sample.GraphCalls == 0 && sample.ModelCalls == 0 &&
			sample.InputTokens == 0 &&
			sample.OutputTokens == 0
	case localProfile:
		return sample.RetrievalCalls == 0 && sample.GraphCalls <= localCallCap && sample.ModelCalls == 0 &&
			sample.InputTokens == 0 &&
			sample.OutputTokens == 0
	case communityProfile:
		return sample.RetrievalCalls == 0 && sample.GraphCalls == 0 && sample.ModelCalls <= communityCallCap
	case globalProfile:
		return sample.RetrievalCalls == 0 && sample.GraphCalls == 0 && sample.ModelCalls <= globalCallCap
	}
	return false
}
func evaluate(raw capture) (report, error) {
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		return report{}, err
	}
	if raw.FixtureIdentity != f.Identity ||
		!slices.Contains([]string{liveExecution, contractExecution}, raw.ExecutionKind) ||
		raw.AdapterIdentity == "" ||
		raw.ModelIdentity == "" ||
		raw.TokenizerIdentity == "" ||
		len(raw.Samples) != 2*len(f.Queries) {
		return report{}, errInvalid
	}
	if err := validateCaptureConfiguration(raw); err != nil {
		return report{}, err
	}
	if err := validatePreparation(raw.Preparation, f, raw.ExecutionKind, raw.HostProfile != nil); err != nil {
		return report{}, err
	}
	rows := make(map[string]observation)
	for _, sample := range raw.Samples {
		if !validLiveModelObservation(raw, sample) {
			return report{}, errInvalid
		}
		if err := validateObservation(sample, f); err != nil {
			return report{}, err
		}
		key := sample.Query + "/" + sample.Profile
		if _, exists := rows[key]; exists {
			return report{}, errInvalid
		}
		rows[key] = sample
	}
	result := report{
		Fixture:                   f,
		Capture:                   raw,
		Measurements:              make(map[string]measured),
		DefaultProfile:            baselineProfile,
		PreparationBudgetsHonored: preparationBudgets(raw.Preparation),
		Limitation:                "Support retrieval only. Natural-language summary quality requires an external consumer evaluation. Scripted artifacts do not establish live model quality. One observation per profile/question has no meaningful latency percentile; raw timings are retained. No default recommendation or automatic promotion is made.",
	}
	if raw.HostProfile != nil {
		result.Limitation += " CLI host tokens are advisory; the counter is a calibration-plus-JSON-byte estimate, not a verified full input bound. Reference budgets remain unchanged and are evaluated separately. Full executor usage/startup timings are retained. Price is unavailable; cost_units=0 is an unpriced ledger value, not provider billing. Supported tools are disabled and observed tool activity rejected; additional context and internal provider dispatch/retries are unverified. This is not an isolated retrieval comparison."
	}
	for _, q := range f.Queries {
		baseline, bok := rows[q.ID+"/"+baselineProfile]
		recipe, rok := rows[q.ID+"/"+q.Recipe]
		if !bok || !rok || baseline.Scope != recipe.Scope || baseline.Publication != recipe.Publication {
			return report{}, errInvalid
		}
		b, r := recall(baseline, q), recall(recipe, q)
		result.Measurements[q.ID] = measured{
			BaselineRecall:         b,
			RecipeRecall:           r,
			Gain:                   r - b,
			BaselineBudgetsHonored: budgetsHonored(baseline),
			RecipeBudgetsHonored:   budgetsHonored(recipe),
		}
	}
	return result, nil
}

func validOutcome(sample observation) bool {
	if sample.Stop == "" {
		return false
	}
	if sample.Failed {
		return sample.Outcome == failedOutcome
	}
	return slices.Contains([]string{completeOutcome, "partial", "insufficient"}, sample.Outcome)
}
