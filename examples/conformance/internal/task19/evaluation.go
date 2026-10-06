package task19

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// Row is one actual execution, with failures retained as empty rankings.
const failedOutcome = "failed"

type Row struct {
	PackingStatus      retrieval.ArtifactPackingStatus `json:"packing_status"`
	PartialDelivery    bool                            `json:"partial_document_delivery"`
	DeliveryUncertain  bool                            `json:"delivery_uncertain"`
	RecipeOutcome      string                          `json:"recipe_outcome,omitempty"`
	StopReason         string                          `json:"stop_reason,omitempty"`
	Sufficiency        *bool                           `json:"host_sufficiency,omitempty"`
	DispatchCandidates []int                           `json:"dispatch_candidate_counts"`
	GraphContributors  []source.Reference              `json:"graph_contributors,omitempty"`
	GraphPublication   string                          `json:"graph_publication,omitempty"`
	CandidateIDs       []string                        `json:"candidate_ids,omitempty"`
	EncoderCalls       uint64                          `json:"local_encoder_calls"`
	RerankerCalls      uint64                          `json:"local_reranker_calls"`
	Query              string                          `json:"query"`
	Strategy           string                          `json:"strategy"`
	Repeat             int                             `json:"repeat"`
	Retrieved          []string                        `json:"retrieved"`
	Delivered          []string                        `json:"delivered"`
	Sources            []source.Locator                `json:"sources"`
	Publication        string                          `json:"publication"`
	Scope              string                          `json:"scope"`
	Outcome            string                          `json:"outcome"`
	Error              string                          `json:"error,omitempty"`
	RetrievalCalls     uint64                          `json:"retrieval_calls"`
	ModelCalls         uint64                          `json:"model_calls"`
	LocalInputUnits    uint64                          `json:"local_input_units"`
	LocalOutputUnits   uint64                          `json:"local_output_units"`
	UsageKnown         bool                            `json:"local_usage_known"`
	BillingState       string                          `json:"provider_billing_state"`
	ContextUnits       int64                           `json:"full_context_utf8_bytes"`
	Nanos              int64                           `json:"elapsed_nanos"`
	ScopeViolations    int                             `json:"scope_violations"`
	StaleViolations    int                             `json:"stale_violations"`
	CitationViolations int                             `json:"citation_violations"`
}
type Metrics struct {
	Answerable        int     `json:"answerable_denominator"`
	NoAnswer          int     `json:"no_answer_denominator"`
	Recall            float64 `json:"recall_at_3"`
	MRR               float64 `json:"mrr_at_3"`
	NDCG              float64 `json:"ndcg_at_3"`
	RetrievedCoverage float64 `json:"retrieved_coverage"`
	DeliveredCoverage float64 `json:"delivered_coverage"`
	NoAnswerErrors    int     `json:"no_answer_errors"`
	Abstentions       int     `json:"no_answer_abstentions"`
	Failures          int     `json:"failures"`
	Violations        int     `json:"violations"`
	PolicyKnown       bool    `json:"policy_known_compliant"`
}
type Report struct {
	RankingRepeatable    bool               `json:"ranking_repeatable"`
	NumericQualification map[string]bool    `json:"numeric_qualification"`
	CommandArguments     []string           `json:"command_arguments"`
	FreezeSHA256         string             `json:"freeze_sha256"`
	FreezeManifest       json.RawMessage    `json:"freeze_manifest,omitempty"`
	Schema               string             `json:"schema"`
	DatasetID            string             `json:"dataset_id"`
	Split                string             `json:"split"`
	CorpusDigest         string             `json:"corpus_digest"`
	SplitDigest          string             `json:"split_digest"`
	Profile              string             `json:"execution_profile"`
	Policy               map[string]any     `json:"configuration"`
	Rows                 []Row              `json:"rows"`
	Metrics              map[string]Metrics `json:"metrics"`
	Default              string             `json:"default"`
	Limitation           string             `json:"limitation"`
}
type Run func(context.Context, Corpus, Query, string, int) (Row, error)

func Policy() map[string]any {
	//nolint:gosec // Fixed profile identities are public accounting descriptions, not secrets.
	return map[string]any{
		"top_k":                            TopK,
		"candidate_limit":                  CandidateLimit,
		"candidate_limit_semantics":        "per-retrieval-dispatch materialized originals; union deduplicated",
		"whole_attempt_candidate_universe": UniverseLimit,
		"retrieval_slots":                  RetrievalSlots,
		"model_slots":                      ModelSlots,
		"full_context_utf8_bytes":          ContextBytes,
		"deadline_nanos":                   int64(Deadline),
		"repeats":                          Repeats,
		"local_input_bytes_cap":            InputBytes,
		"local_output_bytes_cap":           OutputBytes,
		"tokenizer":                        "serialized-bytes/v1 (UTF8 text plus float32 vector/tensor payloads, not provider tokens)",
		"model":                            "deterministic-host-script/v1",
		"encoder":                          "sha256-token-64-sign-bits/v1",
		"seed":                             "no RNG; exact deterministic algorithms",
		"provider":                         "none",
		"usage_source":                     "in-process dispatch counters and complete serialized-byte counters",
		"billing":                          "no provider invoked; provider prices unavailable",
		"hypothesis":                       "Recall gain >=0.05; no MRR/nDCG decline; no added no-answer/failures; zero violations; equal known policy compliance; local does not promote universal defaults",
	}
}

//nolint:gocognit,funlen // Explicit execution grid retains every failed row and preparation phase.
func Execute(corpusPath, splitPath, output string, strategies []string, run Run) error {
	c, s, e := Read(corpusPath, splitPath)
	if e != nil {
		return e
	}
	if e = Validate(c, s); e != nil {
		return e
	}
	cb, _ := os.ReadFile(corpusPath)
	sb, _ := os.ReadFile(splitPath)
	report := Report{
		CommandArguments: slices.Clone(os.Args[1:]),
		Schema:           "task19-evaluation/v1",
		DatasetID:        c.DatasetID,
		Split:            s.Split,
		CorpusDigest:     Digest(cb),
		SplitDigest:      Digest(sb),
		Profile:          "deterministic-local-consumer/library-composition",
		Policy:           Policy(),
		Default:          "baseline",
		Limitation:       "Host-authored graph facts, scripted planner/assessor and hash vectors are deterministic local consumer profiles, not provider or LLM effectiveness. No final answer judge, agent injection immunity, latency percentiles or stochastic confidence intervals. Tensor is candidate-only. Baseline stays universal default even if numeric hypothesis qualifies.",
	}
	freezePath := os.Getenv("RAGY_TASK19_FREEZE")
	if s.Split == "holdout" && freezePath == "" {
		return errors.New("holdout requires RAGY_TASK19_FREEZE manifest")
	}
	if freezePath != "" {
		//nolint:gosec // Explicit host-selected manifest path, never retrieved source text.
		f, err := os.ReadFile(freezePath)
		if err != nil {
			return err
		}
		if err := VerifyFreeze(f, cb, sb, s.Split); err != nil {
			return err
		}
		report.FreezeSHA256 = Digest(f)
		report.FreezeManifest = f
	}
	if setup, ok := PreparedSetup(c); ok {
		report.Policy["managed_index_setup"] = setup
	}
	setupMu.Lock()
	maps.Copy(report.Policy, extraSetups)
	setupMu.Unlock()
	for _, strategy := range strategies {
		for repeat := 1; repeat <= Repeats; repeat++ {
			for _, q := range s.Queries {
				ctx, cancel := context.WithTimeout(context.Background(), Deadline)
				start := time.Now()
				row, err := run(ctx, c, q, strategy, repeat)
				row.Query = q.ID
				row.Strategy = strategy
				row.Repeat = repeat
				row.Scope = q.Scope
				row.Nanos = time.Since(start).Nanoseconds()
				if err != nil {
					row.Error = err.Error()
					row.Outcome = failedOutcome
					row.Retrieved = nil
					row.Delivered = nil
					row.Sources = nil
				}
				if row.Outcome == "" {
					row.Outcome = "complete"
				}
				cancel()
				Audit(c, q, &row)
				report.Rows = append(report.Rows, row)
			}
		}
	}
	report.Metrics = Measure(c, s, report.Rows, strategies)
	report.RankingRepeatable = Repeatable(report.Rows)
	report.NumericQualification = map[string]bool{}
	if baseline, ok := report.Metrics["baseline"]; ok {
		for profile, value := range report.Metrics {
			if profile != "baseline" {
				report.NumericQualification[profile] = Qualifies(value, baseline)
			}
		}
	}
	encoded, e := json.MarshalIndent(report, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(output, append(encoded, '\n'), 0o600)
}
func Pack(ctx context.Context, c Corpus, b access.Binding, docs []retrieval.Document[Meta], row *Row) error {
	for _, d := range docs {
		if _, e := Supports(c)(ctx, b, d); e != nil {
			return e
		}
		row.Retrieved = append(row.Retrieved, d.ID)
	}
	artifact, e := (retrieval.DefaultArtifactRenderer[Meta]{}).Render(
		ctx,
		b,
		retrieval.NewResultSet(docs, retrieval.DocumentIDResolver[Meta]{}),
		ArtifactOptions(c),
	)
	if e != nil {
		return e
	}
	row.ContextUnits = artifact.Resource.Used
	row.PackingStatus = artifact.Resource.Packing
	row.Publication = b.Publication().Reference()
	for _, s := range artifact.Snippets {
		row.PartialDelivery = row.PartialDelivery || !s.FullDocument
		row.DeliveryUncertain = row.DeliveryUncertain || s.DeliveryUncertain
		row.Delivered = append(row.Delivered, s.DocumentID)
		row.Sources = append(row.Sources, s.Supports...)
	}
	return nil
}
func Eligible(c Corpus, q Query) []Document {
	out := []Document{}
	for _, d := range c.Documents {
		if d.Current && (d.Scope == PublicScope || d.Scope == q.Scope) {
			out = append(out, d)
		}
	}
	return out
}
func Validate(c Corpus, s Split) error {
	if c.SchemaVersion != "1.0.0" || s.SchemaVersion != "1.0.0" {
		return errors.New("unsupported dataset schema version")
	}
	if c.DatasetID == "" || c.DatasetID != s.DatasetID || len(c.Documents) == 0 || len(s.Queries) == 0 {
		return errors.New("invalid dataset identity")
	}
	docs := map[string]Document{}
	for _, d := range c.Documents {
		if _, ok := docs[d.ID]; ok || d.ID == "" || d.SourceID == "" || d.Revision == "" || d.Scope == "" {
			return errors.New("invalid document")
		}
		docs[d.ID] = d
	}
	seen := map[string]bool{}
	for _, q := range s.Queries {
		if seen[q.ID] || q.ID == "" || q.Scope == "" || q.Answerable != (len(q.Qrels) > 0) {
			return errors.New("invalid query")
		}
		seen[q.ID] = true
		rels := map[string]bool{}
		for _, r := range q.Qrels {
			d, ok := docs[r.DocumentID]
			if !ok || !d.Current || (d.Scope != PublicScope && d.Scope != q.Scope) || rels[r.DocumentID] ||
				r.Grade < 1 ||
				r.Grade > 3 {
				return fmt.Errorf("invalid qrel %s", q.ID)
			}
			rels[r.DocumentID] = true
		}
	}
	return nil
}

//nolint:gocognit // Independent identity and scope checks intentionally audit every evidence class.
func Audit(c Corpus, q Query, row *Row) {
	docs := map[string]Document{}
	for _, d := range c.Documents {
		docs[d.ID] = d
	}
	for _, id := range append(append(slices.Clone(row.Retrieved), row.Delivered...), row.CandidateIDs...) {
		d, ok := docs[id]
		if !ok || (d.Scope != PublicScope && d.Scope != q.Scope) {
			row.ScopeViolations++
		}
		if !ok || !d.Current {
			row.StaleViolations++
		}
	}
	for _, ref := range row.GraphContributors {
		d, ok := docs[ref.Artifact]
		if !ok || ref != Reference(c, d) || (d.Scope != PublicScope && d.Scope != q.Scope) || !d.Current {
			row.CitationViolations++
		}
	}
	for _, loc := range row.Sources {
		d, ok := docs[loc.Reference.Artifact]
		if !ok || loc != Locator(c, d) || !slices.Contains(row.Delivered, d.ID) {
			row.CitationViolations++
		}
	}
	for _, id := range row.Delivered {
		d, ok := docs[id]
		if !ok || !slices.Contains(row.Sources, Locator(c, d)) {
			row.CitationViolations++
		}
	}
}
func Compliant(r Row) bool {
	return r.UsageKnown && r.RetrievalCalls <= RetrievalSlots &&
		OperationCompliant(r) &&
		r.ContextUnits >= 0 && r.ContextUnits <= ContextBytes &&
		r.LocalInputUnits <= InputBytes &&
		r.LocalOutputUnits <= OutputBytes &&
		CandidateCompliant(r) &&
		r.Nanos >= 0 && r.Nanos <= int64(Deadline) &&
		r.Outcome != failedOutcome &&
		r.ScopeViolations == 0 && r.StaleViolations == 0 && r.CitationViolations == 0
}
func Rank(ids []string, qrels []Qrel) (float64, float64, float64) {
	grades := map[string]int{}
	ideal := []int{}
	for _, r := range qrels {
		grades[r.DocumentID] = r.Grade
		ideal = append(ideal, r.Grade)
	}
	if len(ideal) == 0 {
		return 0, 0, 0
	}
	slices.Sort(ideal)
	slices.Reverse(ideal)
	found := 0
	mrr, dcg, idcg := 0.0, 0.0, 0.0
	seen := map[string]bool{}
	for i, id := range ids[:min(TopK, len(ids))] {
		g := grades[id]
		if g > 0 && !seen[id] {
			found++
			if mrr == 0 {
				mrr = 1 / float64(i+1)
			}
			dcg += (math.Pow(2, float64(g)) - 1) / math.Log2(float64(i+2))
		}
		seen[id] = true
	}
	for i, g := range ideal[:min(TopK, len(ideal))] {
		idcg += (math.Pow(2, float64(g)) - 1) / math.Log2(float64(i+2))
	}
	return float64(found) / float64(len(ideal)), mrr, dcg / idcg
}

//nolint:gocognit // Grid denominators include missing, failed and no-answer executions.
func Measure(_ Corpus, s Split, rows []Row, strategies []string) map[string]Metrics {
	out := map[string]Metrics{}
	for _, strategy := range strategies {
		m := Metrics{PolicyKnown: true}
		for repeat := 1; repeat <= Repeats; repeat++ {
			for _, q := range s.Queries {
				r := Row{Outcome: failedOutcome}
				for _, candidate := range rows {
					if candidate.Query == q.ID && candidate.Strategy == strategy && candidate.Repeat == repeat {
						r = candidate
						break
					}
				}
				if r.Outcome == failedOutcome {
					m.Failures++
					r.Retrieved = nil
					r.Delivered = nil
					r.CandidateIDs = nil
				}
				m.PolicyKnown = m.PolicyKnown && Compliant(r)
				m.Violations += r.ScopeViolations + r.StaleViolations + r.CitationViolations
				if !q.Answerable {
					m.NoAnswer++
					if len(r.Delivered) > 0 {
						m.NoAnswerErrors++
					} else if r.Outcome != failedOutcome {
						m.Abstentions++
					}
					continue
				}
				m.Answerable++
				rec, mrr, ndcg := Rank(r.Retrieved, q.Qrels)
				m.Recall += rec
				m.MRR += mrr
				m.NDCG += ndcg
				m.RetrievedCoverage += Coverage(r.CandidateIDs, q.Qrels)
				del, _, _ := Rank(r.Delivered, q.Qrels)
				m.DeliveredCoverage += del
			}
		}
		if m.Answerable > 0 {
			n := float64(m.Answerable)
			m.Recall /= n
			m.MRR /= n
			m.NDCG /= n
			m.RetrievedCoverage /= n
			m.DeliveredCoverage /= n
		}
		out[strategy] = m
	}
	return out
}

func Coverage(ids []string, qrels []Qrel) float64 {
	if len(qrels) == 0 {
		return 0
	}
	found := 0
	for _, r := range qrels {
		if slices.Contains(ids, r.DocumentID) {
			found++
		}
	}
	return float64(found) / float64(len(qrels))
}

//nolint:gochecknoglobals // Narrow command-scoped preparation registry, guarded by setupMu.
var setupMu sync.Mutex

//nolint:gochecknoglobals // Explicit preparation observations owned by the current command.
var extraSetups = map[string]any{}

// RegisterSetup records an explicit consumer preparation observation before Execute.
func RegisterSetup(kind string, observation any) {
	setupMu.Lock()
	defer setupMu.Unlock()
	extraSetups[kind] = observation
}

func CandidateCompliant(r Row) bool {
	seen := map[string]bool{}
	for _, id := range r.CandidateIDs {
		if id == "" || seen[id] {
			return false
		}
		seen[id] = true
	}
	if len(seen) > UniverseLimit {
		return false
	}
	if uint64(len(r.DispatchCandidates)) != r.RetrievalCalls {
		return false
	}
	for _, n := range r.DispatchCandidates {
		if n < 0 || n > CandidateLimit {
			return false
		}
	}
	return true
}

func OperationCompliant(r Row) bool {
	if r.ModelCalls > ModelSlots || r.EncoderCalls > ModelSlots || r.RerankerCalls > ModelSlots {
		return false
	}
	return r.ModelCalls+r.EncoderCalls+r.RerankerCalls <= ModelSlots
}

func Qualifies(candidate, baseline Metrics) bool {
	return candidate.Recall-baseline.Recall >= 0.05 && candidate.MRR >= baseline.MRR &&
		candidate.NDCG >= baseline.NDCG &&
		candidate.NoAnswerErrors <= baseline.NoAnswerErrors &&
		candidate.Failures <= baseline.Failures &&
		candidate.Violations == 0 &&
		baseline.Violations == 0 &&
		candidate.PolicyKnown &&
		baseline.PolicyKnown
}
func Repeatable(rows []Row) bool {
	first := map[string]Row{}
	for _, row := range rows {
		key := row.Strategy + "/" + row.Query
		previous, ok := first[key]
		if !ok {
			first[key] = row
			continue
		}
		if !slices.Equal(previous.Retrieved, row.Retrieved) || !slices.Equal(previous.Delivered, row.Delivered) ||
			!slices.Equal(previous.CandidateIDs, row.CandidateIDs) ||
			previous.Outcome != row.Outcome ||
			previous.Error != row.Error {
			return false
		}
	}
	return true
}
