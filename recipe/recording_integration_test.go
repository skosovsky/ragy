package recipe_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/recording"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type recordSink struct {
	calls  int
	err    error
	hook   func()
	record evidence.Record
}

func (s *recordSink) Write(_ context.Context, record evidence.Record) error {
	s.calls++
	s.record = record
	if s.hook != nil {
		s.hook()
	}
	return s.err
}

func recordingConfig(
	t *testing.T,
	f *fixture,
	mode evidence.Mode,
	sink *recordSink,
) recording.Config[[]string, []string, meta] {
	t.Helper()
	schema := f.config.Backend.(interface{ Schema() filter.Schema }).Schema()
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	return recording.Config[[]string, []string, meta]{
		Recipe:      r,
		Mode:        mode,
		Sink:        sink,
		RetrievalID: "attempt",
		Schema:      schema,
		Codec:       retrieval.NewJSONCodec[meta](schema),
		CloneMeta:   f.config.CloneMeta,
		SourceAdmission: func(_ context.Context, _ access.Binding, ref source.Reference) error {
			if ref != location(ref.Artifact).Reference {
				return access.NonSkippable(ragy.ErrUnavailable)
			}
			return nil
		},
		Policy: evidence.Policy{
			AllowIdentifier: func(evidence.IdentifierKind, string) bool { return true },
			AllowNumbers:    true,
		},
	}
}

func recordedRequest(f *fixture) request {
	return request{
		Read:    f.read,
		Text:    "original",
		Intent:  []string{"intent"},
		Meta:    []string{"secret"},
		Options: retrieval.RetrieveOptions{TopK: 3},
	}
}

func TestRecipeRecordingActualBM25ModesAndNoRetry(t *testing.T) {
	for _, mode := range []evidence.Mode{evidence.Disabled, evidence.BestEffort, evidence.Required} {
		t.Run(string(mode), func(t *testing.T) {
			// Arrange: native scoped BM25, scripted model ports, failing sink.
			f := newFixture(t, recipe.SingleRewrite)
			configureActualBM25(t, f)
			f.planned, f.selected = []string{"возврат оплаты срок"}, []int{1}
			sink := &recordSink{err: errors.New("sink secret failure")}
			config := recordingConfig(t, f, mode, sink)
			// Act.
			output, err := recording.Run(context.Background(), recordedRequest(f), config)
			// Assert: sink failure never repeats the two model calls or retrieval attempt.
			if f.modelCalls != 2 || len(f.retrieved) != 2 || len(output.Result.Selected) == 0 {
				t.Fatal(f.modelCalls, err)
			}
			if !assertRecordingReceipt(t, mode, sink, output.Receipt, err) {
				return
			}
			assertRecordedResult(t, output)
		})
	}
}

func TestRecipeRecordingRequiredAdmissionAndRevocation(t *testing.T) {
	for _, at := range []string{"unsupported", "nil_codec", "source", "sink"} {
		t.Run(at, func(t *testing.T) {
			// Arrange.
			f := newFixture(t, recipe.SingleRewrite)
			configureActualBM25(t, f)
			f.planned, f.selected = []string{"возврат оплаты срок"}, []int{1}
			sink := &recordSink{}
			config := recordingConfig(t, f, evidence.Required, sink)
			switch at {
			case "unsupported":
				config.Required = []evidence.Field{evidence.JudgmentField}
			case "nil_codec":
				config.Codec = (*retrieval.JSONCodec[meta])(nil)
			case "source":
				config.SourceAdmission = func(context.Context, access.Binding, source.Reference) error { f.epoch++; return nil }
			case "sink":
				sink.hook = func() { f.epoch++ }
			}
			// Act.
			output, err := recording.Run(context.Background(), recordedRequest(f), config)
			// Assert.
			if err == nil || len(output.Result.Selected) != 0 || output.Receipt.State != "" {
				t.Fatal(output, err)
			}
			if (at == "unsupported" || at == "nil_codec") && (f.modelCalls != 0 || sink.calls != 0) {
				t.Fatal("unsupported export dispatched work")
			}
			if at == "source" && sink.calls != 0 {
				t.Fatal("revoked export reached sink")
			}
		})
	}
}

func TestRecipeRecordingBudgetStopReportsUnexecutedModels(t *testing.T) {
	// Arrange: baseline BM25 can return evidence, but no model call is admitted.
	f := newFixture(t, recipe.SingleRewrite)
	configureActualBM25(t, f)
	f.config.Limits.ModelCalls = 0
	sink := &recordSink{}
	config := recordingConfig(t, f, evidence.Required, sink)
	req := recordedRequest(f)
	req.Text = "возврат оплаты срок"
	// Act.
	output, err := recording.Run(context.Background(), req, config)
	// Assert.
	if err != nil || f.modelCalls != 0 || len(f.retrieved) != 1 || output.Result.Outcome != recipe.Partial ||
		sink.calls != 1 {
		t.Fatal(output.Result.Outcome, f.modelCalls, f.retrieved, err)
	}
	snapshot, err := output.Receipt.Record.Snapshot()
	if err != nil || snapshot.Reason != evidence.Budget || snapshot.Outcome != evidence.Partial {
		t.Fatal(snapshot, err)
	}
	notRun := 0
	for _, stage := range snapshot.Stages {
		if stage.Status == evidence.NotRun {
			notRun++
		}
	}
	if notRun != 2 {
		t.Fatal("unexecuted models not distinguished", snapshot.Stages)
	}
}

func assertRecordedResult(t *testing.T, output evidence.Execution[recipe.Result[meta]]) {
	t.Helper()
	snapshot, snapshotErr := output.Receipt.Record.Snapshot()
	if snapshotErr != nil {
		t.Fatal(snapshotErr)
	}
	if snapshot.Query.State != evidence.Omitted || snapshot.Publication.Value == nil ||
		*snapshot.Publication.Value != "pub1" {
		t.Fatal(snapshot)
	}
	assertRecordedScores(t, snapshot)
	encoded, _ := output.Receipt.Record.MarshalJSON()
	for _, secret := range []string{"sink secret failure", "owned", "secret", "Возврат оплаты"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatal("unexpected payload", secret)
		}
	}
	before := string(encoded)
	output.Result.Selected[0].Document.Meta.Tags[0] = "mutated"
	snapshot.Stages = nil
	encoded, _ = output.Receipt.Record.MarshalJSON()
	if string(encoded) != before {
		t.Fatal("record aliases result")
	}
}

func assertRecordedScores(t *testing.T, snapshot evidence.Snapshot) {
	t.Helper()
	native, fused := false, false
	for _, stage := range snapshot.Stages {
		for _, hit := range stage.Hits {
			assertRecordedHit(t, hit)
			if stage.Name.Value == nil {
				continue
			}
			native = native || (*stage.Name.Value == "retrieve/1" && hit.Score.State == evidence.ScoreNative)
			fused = fused || (*stage.Name.Value == "fusion" && hit.Score.State == evidence.ScoreNormalized)
		}
	}
	if !native || !fused {
		t.Fatal("native observations or fusion missing", snapshot.Stages)
	}
}

func assertRecordedHit(t *testing.T, hit evidence.WireHit) {
	t.Helper()
	if hit.ID.Value != nil && *hit.ID.Value == "private" {
		t.Fatal("private document escaped")
	}
	if hit.Snippet.State != evidence.Omitted {
		t.Fatal("snippet exported without policy")
	}
	for _, ref := range hit.Sources {
		if ref.Revision.Value == nil || *ref.Revision.Value != "r1" {
			t.Fatal(ref)
		}
	}
}

func TestSnapshotResultRevocationSuppressesEmptyEnvelope(t *testing.T) {
	// Arrange: the second freshness gate revokes even an envelope without documents.
	f := newFixture(t, recipe.SingleRewrite)
	schema := f.config.Backend.(backend).schema
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	checks := 0
	read, err := access.Scoped(access.ScopedConfig{Snapshot: f.read.Snapshot(), Schema: schema, Mandatory: mandatory,
		Publication: f.read.Publication(), Now: func() time.Time { return f.now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			checks++
			if checks == 2 {
				return ragy.ErrUnavailable
			}
			return nil
		})})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := recipe.SnapshotResult(
		context.Background(),
		read,
		recipe.Result[meta]{RecipeRevision: "private"},
		f.config.CloneMeta,
	)
	// Assert.
	if !errors.Is(err, ragy.ErrUnavailable) || result.RecipeRevision != "" {
		t.Fatal(result, err)
	}
}

func assertRecordingReceipt(
	t *testing.T,
	mode evidence.Mode,
	sink *recordSink,
	receipt evidence.Receipt,
	err error,
) bool {
	t.Helper()
	if mode == evidence.Disabled {
		if err != nil || sink.calls != 0 || receipt.State != evidence.RecordingDisabled {
			t.Fatal(receipt, err)
		}
		return false
	}
	if sink.calls != 1 || receipt.State != evidence.RecordingFailed {
		t.Fatal(sink.calls, receipt, err)
	}
	if mode == evidence.Required && !errors.Is(err, evidence.ErrRecordingFailed) {
		t.Fatal(err)
	}
	if mode == evidence.BestEffort && err != nil {
		t.Fatal(err)
	}
	return true
}

func TestRecipeRecordingExportsOriginalLocationsAfterDedup(t *testing.T) {
	// Arrange: actual scoped BM25; both variants select the same original source.
	f := newFixture(t, recipe.MultiQuery)
	configureActualBM25(t, f)
	f.planned, f.selected = []string{"сброс пароля", "восстановления доступа"}, []int{1, 2}
	sink := &recordSink{}
	cfg := recordingConfig(t, f, evidence.Required, sink)
	cfg.Policy.AllowLocation = func(source.Locator) bool { return true }
	cfg.Policy.AllowContribution = func(evidence.Contribution) bool { return true }
	// Act.
	execution, err := recording.Run(context.Background(), recordedRequest(f), cfg)
	// Assert: union locations survive RRF while typed contributors retain both queries.
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := sink.record.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	last := snapshot.Stages[len(snapshot.Stages)-1]
	found := false
	for _, hit := range last.Hits {
		if hit.ID.Value == nil || *hit.ID.Value != "d2" {
			continue
		}
		found = true
		if hit.ContributionsState != evidence.Observed || len(hit.Contributions) != 2 ||
			hit.Contributions[0].QueryIndex != 1 ||
			hit.Contributions[1].QueryIndex != 2 ||
			len(hit.Contributions[0].Locations) != 1 {
			t.Fatal("query-to-contributor wire association lost")
		}
		if hit.LocationsState != evidence.Observed || len(hit.Locations) != 1 ||
			hit.Locations[0].Source.Revision.Value == nil ||
			*hit.Locations[0].Source.Revision.Value != "r1" ||
			hit.Locations[0].Kind != source.DocumentLocation {
			t.Fatal("dedup original location lost")
		}
	}
	if !found || len(execution.Result.Selected) == 0 || len(execution.Result.Selected[0].Contributors) != 2 {
		t.Fatal("query contributors or fusion hit lost")
	}
}
