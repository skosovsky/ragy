//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

type evidenceSpy struct {
	calls int
	fail  bool
}

func (s *evidenceSpy) Write(context.Context, evidence.Record) error {
	s.calls++
	if s.fail {
		return errors.New("private sink message")
	}
	return nil
}
func TestActualGraphEvidenceRecordingModesExecuteOnceAndKeepPrivacy(t *testing.T) {
	for _, mode := range []evidence.Mode{evidence.Disabled, evidence.BestEffort, evidence.Required} {
		t.Run(string(mode), func(t *testing.T) {
			// Arrange: actual scoped graph expansion with an injected failed recording sink.
			f, corpus, baseline, read := publishedLocalFixture(t)
			prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
			if err != nil {
				t.Fatal(err)
			}
			sink := &evidenceSpy{fail: true}
			executed := 0
			q := f.Queries[0]
			// Act.
			result, err := prepared.recordObservation(
				t.Context(),
				read,
				q,
				mode,
				sink,
				func(ctx context.Context) (observation, error) { executed++; return corpus.local(ctx, read, q) },
			)
			// Assert: source facts/one graph call persist; failed required recording is not success.
			if executed != 1 || result.Result.GraphCalls != 1 || len(result.Result.Supports) != 2 {
				t.Fatal(result, err, executed)
			}
			assertGraphRecording(t, mode, result, err, sink, q, f)
		})
	}
}
func assertSupportEvidence(t *testing.T, record evidence.Record, want int) {
	t.Helper()
	snap, err := record.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if snap.Query.State != evidence.Omitted || len(snap.Stages) != 2 ||
		snap.Stages[0].Status != evidence.StageObserved ||
		snap.Stages[1].Status != evidence.StageObserved ||
		len(snap.Stages[1].Hits) != want {
		t.Fatal(snap)
	}
	for _, hit := range snap.Stages[1].Hits {
		if hit.Score.State != evidence.ScoreAbsent || hit.Snippet.State != evidence.Omitted ||
			hit.Judgment.State != evidence.Ungradable ||
			len(hit.Sources) != 1 {
			t.Fatal(hit)
		}
	}
}
func TestGraphEvidenceSourceRetirementBeforeExportSuppressesRecord(t *testing.T) {
	for _, src := range []string{"s1", "s2"} {
		t.Run(src, func(t *testing.T) {
			// Arrange: actual graph facts retrieved before original source retention changes.
			f, corpus, baseline, read := publishedLocalFixture(t)
			prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
			if err != nil {
				t.Fatal(err)
			}
			sink := &evidenceSpy{}
			// Act: ordinary result exists, but mandatory source admission must fail on capture.
			result, err := prepared.recordObservation(
				t.Context(),
				read,
				f.Queries[0],
				evidence.Required,
				sink,
				func(ctx context.Context) (observation, error) {
					sample, runErr := corpus.local(ctx, read, f.Queries[0])
					retireSummaryGraphSource(t, corpus, src)
					return sample, runErr
				},
			)
			// Assert: no sink/output exposes the stale result under the current host retention rule.
			if !access.IsProtectionFailure(err) || sink.calls != 0 || len(result.Result.Supports) != 0 {
				t.Fatal(result, err, sink.calls)
			}
		})
	}
}
func TestGraphDiagnosticNumbersDoNotRoundExactOverflowOrUnknownToZero(t *testing.T) {
	// Arrange/Act/Assert.
	for _, number := range []evidence.Number{diagnosticNumber(^uint64(0), true), diagnosticNumber(0, false), diagnosticRetrievalCalls(observation{CallsKnown: true, RetrievalCalls: ^uint64(0), GraphCalls: 1})} {
		if number.State != evidence.Unavailable || number.Value != nil {
			t.Fatal(number)
		}
	}
	if number := diagnosticNumber(
		0,
		true,
	); number.State != evidence.Observed || number.Value == nil ||
		*number.Value != 0 {
		t.Fatal(number)
	}
}

func assertGraphRecording(
	t *testing.T,
	mode evidence.Mode,
	result evidence.Execution[observation],
	err error,
	sink *evidenceSpy,
	q query,
	f fixture,
) {
	t.Helper()
	if mode == evidence.Disabled {
		if err != nil || sink.calls != 0 || result.Receipt.State != evidence.RecordingDisabled {
			t.Fatal(result, err)
		}
		return
	}
	if sink.calls != 1 || result.Receipt.State != evidence.RecordingFailed {
		t.Fatal(result, err)
	}
	if (mode == evidence.Required) != errors.Is(err, evidence.ErrRecordingFailed) {
		t.Fatal(err)
	}
	encoded, marshalErr := result.Receipt.Record.MarshalJSON()
	if marshalErr != nil || strings.Contains(string(encoded), q.Text) ||
		strings.Contains(string(encoded), f.Sources[0].Text) ||
		strings.Contains(string(encoded), "access_fingerprint") ||
		strings.Contains(string(encoded), "private sink") {
		t.Fatal("privacy", marshalErr)
	}
	assertSupportEvidence(t, result.Receipt.Record, 2)
}

func TestGraphSupportEvidenceCaptureStandalone(t *testing.T) {
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	sample, err := corpus.local(t.Context(), read, f.Queries[0])
	if err != nil {
		t.Fatal(err)
	}
	_, sample.Configuration, err = configurationBytes()
	if err != nil {
		t.Fatal(err)
	}
	record, err := evidence.Capture(
		t.Context(),
		read,
		prepared.evidenceInput(f.Queries[0], sample),
		supportPolicy(read, f.Queries[0], sample),
	)
	if err != nil {
		t.Fatal("direct capture", err)
	}
	assertSupportEvidence(t, record, 2)
	sample.Evidence, err = record.MarshalJSON()
	if err != nil || validateObservation(sample, f) != nil {
		t.Fatal("valid association", err)
	}
	assertGraphEvidenceTamperingRejected(t, sample, f)
}

func assertGraphEvidenceTamperingRejected(t *testing.T, sample observation, f fixture) {
	t.Helper()
	changes := map[string]func(*evidence.Snapshot){
		"scope":           func(s *evidence.Snapshot) { *s.Scope.Value = "foreign" },
		"publication":     func(s *evidence.Snapshot) { *s.Publication.Value = "foreign" },
		"recipe":          func(s *evidence.Snapshot) { *s.Recipe.Value = "foreign" },
		"retrieval":       func(s *evidence.Snapshot) { *s.RetrievalID.Value = "foreign" },
		"outcome":         func(s *evidence.Snapshot) { s.Outcome = evidence.Insufficient; s.Reason = evidence.MissingEvidence },
		"source":          func(s *evidence.Snapshot) { *s.Stages[len(s.Stages)-1].Hits[0].Sources[0].ID.Value = "s4" },
		"revision":        func(s *evidence.Snapshot) { *s.Stages[len(s.Stages)-1].Hits[0].Sources[0].Revision.Value = "r2" },
		"order":           func(s *evidence.Snapshot) { h := s.Stages[len(s.Stages)-1].Hits; h[0], h[1] = h[1], h[0] },
		"missing-stage":   func(s *evidence.Snapshot) { s.Stages = s.Stages[:len(s.Stages)-1] },
		"duplicate-stage": func(s *evidence.Snapshot) { s.Stages = append(s.Stages, s.Stages[len(s.Stages)-1]) },
	}
	for name, change := range changes {
		t.Run(name, func(t *testing.T) {
			// Arrange: mutate one association in a real captured immutable record.
			record, err := evidence.Decode(sample.Evidence)
			if err != nil {
				t.Fatal(err)
			}
			snapshot, err := record.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			change(&snapshot)
			altered := sample
			altered.Evidence, err = json.Marshal(snapshot)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			err = validateObservation(altered, f)
			// Assert: a structurally valid but misassociated record cannot enter the report.
			if !errors.Is(err, errInvalid) {
				t.Fatal("accepted tampering", err)
			}
		})
	}
	t.Run("invalid-json", func(t *testing.T) {
		altered := sample
		altered.Evidence = json.RawMessage(`{"unexpected":true}`)
		if !errors.Is(validateObservation(altered, f), errInvalid) {
			t.Fatal("accepted invalid record")
		}
	})
}

func TestGraphRequiredAllStageSourcesRejectsMissingInternalObservations(t *testing.T) {
	// Arrange: final delivered supports exist; internal per-stage source observations were not captured.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	sample, err := corpus.local(t.Context(), read, f.Queries[0])
	if err != nil {
		t.Fatal(err)
	}
	sample.stages = nil // Explicitly missing producer observations must still fail required capture.
	input := prepared.evidenceInput(f.Queries[0], sample)
	input.Required = append(input.Required, evidence.SourceField)
	// Act.
	_, err = evidence.Capture(t.Context(), read, input, supportPolicy(read, f.Queries[0], sample))
	// Assert: requesting full internal source observations fails honestly.
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal(err)
	}
}

func TestHybridEvidenceObservesActualLeafAndFusionScores(t *testing.T) {
	// Arrange: persistent dense, BM25 and RRF run with actual pinned source admission.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	sample, err := baseline.retrieve(t.Context(), f.Queries[0])
	if err != nil {
		t.Fatal(err)
	}
	_, sample.Configuration, err = configurationBytes()
	if err != nil {
		t.Fatal(err)
	}
	input := prepared.evidenceInput(f.Queries[0], sample)
	input.Required = append(input.Required, evidence.SourceField)
	// Act: all source observations are now available in the baseline path.
	record, err := evidence.Capture(t.Context(), read, input, supportPolicy(read, f.Queries[0], sample))
	// Assert: numeric native/normalized scores and ranks remain those of each real stage.
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := record.Snapshot()
	if err != nil || len(snapshot.Stages) != 4 {
		t.Fatal(snapshot, err)
	}
	assertHybridStageScores(t, snapshot, sample)
	sample.Evidence, err = record.MarshalJSON()
	if err != nil || validateObservation(sample, f) != nil {
		t.Fatal(err)
	}
	// Mutating the detached stage projection cannot mutate the completed immutable record.
	sample.stages[0].Hits[0].Document.Score = 999
	detached, err := record.Snapshot()
	if err != nil || *detached.Stages[0].Hits[0].Score.Value == 999 {
		t.Fatal(detached, err)
	}
}

func assertHybridStageScores(t *testing.T, snapshot evidence.Snapshot, sample observation) {
	t.Helper()
	for i, name := range []string{denseTarget, lexicalTarget, "rrf"} {
		stage := snapshot.Stages[i]
		if !observedText(stage.Name, name) || stage.Status != evidence.StageObserved || len(stage.Hits) == 0 {
			t.Fatal(stage)
		}
		for j, hit := range stage.Hits {
			actual := sample.stages[i].Hits[j].Document
			if hit.Score.Value == nil || *hit.Score.Value != actual.Score ||
				!observedText(hit.Score.Semantics, string(actual.ScoreSemantics)) ||
				hit.Rank.Value == nil || *hit.Rank.Value != float64(actual.Rank) ||
				hit.Judgment.State != evidence.Ungradable || hit.Snippet.State != evidence.Omitted ||
				len(hit.Sources) == 0 || hit.SourcesState != evidence.Observed {
				t.Fatal(name, hit)
			}
			for _, src := range hit.Sources {
				if !observedText(src.Transformation, "original") || observedText(src.ID, foreignSource) {
					t.Fatal(src)
				}
			}
		}
	}
}

func TestGraphRecipeStagesCaptureUnrankedFactsAndSummaryAssociations(t *testing.T) {
	for _, profile := range []string{localProfile, communityProfile, globalProfile} {
		t.Run(profile, func(t *testing.T) { checkUnrankedRecipeCapture(t, profile) })
	}
}
func checkUnrankedRecipeCapture(t *testing.T, profile string) {
	t.Helper()
	// Arrange: actual published graph, exact community membership and original payload reader.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	var q query
	for _, candidate := range f.Queries {
		if candidate.Recipe == profile {
			q = candidate
			break
		}
	}
	var sample observation
	if profile == localProfile {
		sample, err = corpus.local(t.Context(), read, q)
	} else {
		sample, err = prepared.summary(t.Context(), read, q, contractSummaryPorts())
	}
	if err != nil || sample.Failed {
		t.Fatal(sample, err)
	}
	_, sample.Configuration, err = configurationBytes()
	if err != nil {
		t.Fatal(err)
	}
	input := prepared.evidenceInput(q, sample)
	input.Required = append(input.Required, evidence.SourceField)
	// Act: observed graph/summary stages satisfy required source observation.
	record, err := evidence.Capture(t.Context(), read, input, supportPolicy(read, q, sample))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := record.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	// Assert: no fabricated ranking, numeric score, grade, query or generated prose.
	assertUnrankedGraphStages(t, snapshot, profile)
}
func assertUnrankedGraphStages(t *testing.T, snapshot evidence.Snapshot, profile string) {
	t.Helper()
	expected := []string{profile}
	if profile == globalProfile {
		expected = []string{"community-map-C1", "community-map-C2", "global-reduce"}
	}
	if len(snapshot.Stages) != len(expected)+1 || snapshot.Query.State != evidence.Omitted {
		t.Fatal(snapshot)
	}
	for i, name := range expected {
		stage := snapshot.Stages[i]
		if !observedText(stage.Name, name) || stage.Status != evidence.StageObserved || len(stage.Hits) == 0 {
			t.Fatal(stage)
		}
		for _, hit := range stage.Hits {
			assertUnrankedGraphHit(t, hit)
		}
	}
}
func assertUnrankedGraphHit(t *testing.T, hit evidence.WireHit) {
	t.Helper()
	if hit.Rank.State != evidence.Unavailable || hit.Rank.Value != nil ||
		hit.Score.State != evidence.ScoreAbsent || hit.Score.Value != nil ||
		hit.Judgment.State != evidence.Ungradable || hit.Snippet.State != evidence.Omitted ||
		hit.SourcesState != evidence.Observed || len(hit.Sources) == 0 {
		t.Fatal(hit)
	}
	for _, src := range hit.Sources {
		if !observedText(src.Transformation, "original") || observedText(src.ID, foreignSource) {
			t.Fatal(src)
		}
	}
}

func TestGlobalSummaryRejectsReduceOmittingCommunity(t *testing.T) {
	// Arrange: real C1/C2 maps complete; reduce illegally selects only C1.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	ports := contractSummaryPorts()
	originalModel := ports.model
	ports.model = func(ctx context.Context, input graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
		output, usage, runErr := originalModel(ctx, input)
		if input.Stage == graphsummary.Reduce {
			output.Selected = []int{0}
		}
		return output, usage, runErr
	}
	// Act.
	sample, err := prepared.summary(t.Context(), read, summaryQuery(t, f, globalProfile), ports)
	// Assert: the protocol failure cannot deliver partial global success or stage artifacts.
	if err != nil || !sample.Failed || sample.Outcome != failedOutcome ||
		len(sample.Supports) != 0 || len(sample.stages) != 0 || sample.ModelCalls != 3 || !sample.UsageKnown {
		t.Fatal(sample, err)
	}
}
