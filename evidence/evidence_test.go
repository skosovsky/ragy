package evidence_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type metadata struct {
	Secret []string `json:"-"`
	Tenant string   `json:"tenant"`
}

func reference() source.Reference {
	return source.Reference{
		Namespace:         "n",
		Source:            "policy",
		Revision:          "r2",
		Transformation:    "chunk",
		AccessFingerprint: "acl",
		Artifact:          "p3",
		Representation:    "text",
	}
}
func fixture(t *testing.T) (access.Binding, evidence.Input[metadata], evidence.Policy) {
	t.Helper()
	pub, err := access.PinPublication(
		"pub2",
		[]access.TargetRevision{
			{
				Target:            "dense",
				Namespace:         "n",
				Source:            "policy",
				Revision:          "r2",
				Transformation:    "chunk",
				AccessFingerprint: "acl",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.UnrestrictedAt(pub)
	if err != nil {
		t.Fatal(err)
	}
	input := evidence.Input[metadata]{
		RetrievalID:    "q1",
		RecipeRevision: "baseline",
		Query:          "private raw query",
		Outcome:        evidence.Complete,
		Reason:         evidence.NoReason,
		Coverage:       retrieval.CompleteReadCoverage(),
		Stages: []evidence.Stage[metadata]{
			{
				Name:      "dense",
				Status:    evidence.StageObserved,
				Scores:    evidence.Observed,
				Sources:   evidence.Observed,
				Judgments: evidence.Unavailable,
				Hits: []evidence.Hit[metadata]{
					{
						Document: retrieval.Document[metadata]{
							ID:             "p3",
							Content:        "private raw text",
							Rank:           1,
							Score:          2,
							ScoreState:     retrieval.ScorePresent,
							ScoreSemantics: "maxsim-space",
							Meta:           metadata{Secret: []string{"auth TOKEN"}, Tenant: "a"},
						},
						Sources: []source.Reference{reference()},
					},
				},
			},
			{Name: "rerank", Status: evidence.NotRun},
		},
	}
	policy := evidence.Policy{
		AllowIdentifier: func(_ evidence.IdentifierKind, value string) bool { return value != "denied" },
		AllowNumbers:    true,
	}
	return read, input, policy
}

func TestImmutableRecordPrivacyAndNativeScores(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	// Act.
	record, err := evidence.Capture(context.Background(), read, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	before, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	input.Stages[0].Hits[0].Document.Score = -1
	input.Stages[0].Hits[0].Sources[0].Revision = "r3"
	input.Stages[0].Hits[0].Document.Meta.Secret[0] = "changed auth"
	snapshot, err := record.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	*snapshot.Stages[0].Hits[0].Score.Value = 999
	encoded, err := record.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	encoded[0] = 'x'
	after, err := json.Marshal(record)
	// Assert.
	if err != nil || !bytes.Equal(before, after) {
		t.Fatal("record aliases caller/sink buffers", err)
	}
	for _, secret := range []string{"private raw query", "private raw text", "TOKEN", "changed auth", "r3"} {
		if bytes.Contains(after, []byte(secret)) {
			t.Fatal("private data leaked", secret)
		}
	}
	snapshot, err = record.Snapshot()
	if err != nil || *snapshot.Stages[0].Hits[0].Score.Value != 2 || snapshot.Stages[1].Status != evidence.NotRun ||
		snapshot.Stages[0].Hits[0].Judgment.State != evidence.Ungradable {
		t.Fatal(snapshot, err)
	}
}

func TestAllowlistedSnippetsJudgmentAndRequiredFields(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	input.Stages = input.Stages[:1]
	input.Required = []evidence.Field{
		evidence.SourceField,
		evidence.ScoreField,
		evidence.JudgmentField,
		evidence.SnippetField,
	}
	input.Stages[0].Judgments = evidence.Observed
	input.Stages[0].Hits[0].Judgment = &evidence.Judgment{Query: "q1", Source: reference(), Grade: 2, Rubric: "ordinal"}
	policy.AllowSnippet = func(id string) bool { return id == "p3" }
	// Act.
	record, err := evidence.Capture(context.Background(), read, input, policy)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := record.Snapshot()
	if err != nil || *snapshot.Stages[0].Hits[0].Snippet.Value != "private raw text" ||
		*snapshot.Stages[0].Hits[0].Judgment.Grade.Value != 2 {
		t.Fatal(err)
	}
	policy.AllowSnippet = nil
	if _, err = evidence.Capture(context.Background(), read, input, policy); !errors.Is(err, evidence.ErrPrivacy) {
		t.Fatal("required private snippet accepted", err)
	}
	input.Required = []evidence.Field{evidence.ScoreField}
	input.Stages[0].Scores = evidence.Unsupported
	input.Stages[0].Hits = nil
	if _, err = evidence.Capture(context.Background(), read, input, policy); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal("required unsupported empty score accepted", err)
	}
}

func TestDeniedIDsForeignRevisionAndMissingObservations(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	input.Stages[0].Hits[0].Document.ID = "denied"
	// Act.
	record, err := evidence.Capture(context.Background(), read, input, policy)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(record)
	if err != nil || bytes.Contains(data, []byte("denied")) {
		t.Fatal("denied export id leaked", err)
	}
	input.Stages[0].Hits[0].Sources[0].Revision = "r3"
	if _, err = evidence.Capture(context.Background(), read, input, policy); !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal("foreign source support admitted", err)
	}
	input.Stages[0].Hits = nil
	input.Stages[0].Status = evidence.MissingObservation
	input.Required = []evidence.Field{evidence.SourceField}
	if _, err = evidence.Capture(context.Background(), read, input, policy); !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal("missing observation became complete", err)
	}
}

func TestWireDecoderRejectsUnknownMissingAndIncompatibleFields(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	record, err := evidence.Capture(context.Background(), read, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	data, err := record.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err = json.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}
	// Act and Assert.
	decoded, err := evidence.Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	old, err := decoded.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	wire["auth"] = json.RawMessage(`"TOKEN"`)
	invalid, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(invalid, &decoded); err == nil {
		t.Fatal("unknown secret field accepted")
	}
	preserved, err := decoded.MarshalJSON()
	if err != nil || !bytes.Equal(old, preserved) {
		t.Fatal("failed decode replaced owned record")
	}
	delete(wire, "auth")
	delete(wire, "reason")
	invalid, err = json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = evidence.Decode(invalid); err == nil {
		t.Fatal("missing required zero-value field accepted")
	}
	if _, err = evidence.Decode(
		bytes.Replace(data, []byte(evidence.SchemaIdentity), []byte("incompatible"), 1),
	); !errors.Is(
		err,
		ragy.ErrUnsupported,
	) {
		t.Fatal(err)
	}
	duplicate := append([]byte(`{"schema":"incompatible",`), data[1:]...)
	if _, err = evidence.Decode(duplicate); !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal("duplicate field accepted", err)
	}
}

type spySink struct {
	calls   int
	err     error
	records []evidence.Record
}

func (s *spySink) Write(_ context.Context, record evidence.Record) error {
	s.calls++
	s.records = append(s.records, record)
	return s.err
}

func TestRecordingPoliciesExecuteOnceAndRetainRetrievalFact(t *testing.T) {
	for _, mode := range []evidence.Mode{evidence.Disabled, evidence.BestEffort, evidence.Required} {
		t.Run(string(mode), func(t *testing.T) {
			// Arrange.
			read, input, policy := fixture(t)
			sink := &spySink{err: errors.New("private credentials TOKEN")}
			executions, captures := 0, 0
			config := evidence.RecordingConfig[int]{Mode: mode, Sink: sink,
				Execute:     func(context.Context) (int, error) { executions++; return 7, nil },
				CloneResult: func(value int) (int, error) { return value, nil },
				Capture: func(ctx context.Context, binding access.Binding, _ int, _ error) (evidence.Record, error) {
					captures++
					return evidence.Capture(ctx, binding, input, policy)
				},
			}
			// Act.
			result, err := evidence.Run(context.Background(), read, config)
			// Assert.
			assertRecording(t, mode, executions, captures, sink, result, err)
		})
	}
}

func assertRecording(
	t *testing.T,
	mode evidence.Mode,
	executions, captures int,
	sink *spySink,
	result evidence.Execution[int],
	err error,
) {
	t.Helper()
	if executions != 1 || result.Result != 7 {
		t.Fatal("recording repeated/lost retrieval")
	}
	if mode == evidence.Disabled {
		if sink.calls != 0 || captures != 0 || result.Receipt.State != evidence.RecordingDisabled ||
			err != nil {
			t.Fatal("disabled recorder invoked capture/sink")
		}
		return
	}
	if sink.calls != 1 || captures != 1 || result.Receipt.State != evidence.RecordingFailed {
		t.Fatal("recording failure hidden")
	}
	if mode == evidence.Required &&
		(!errors.Is(err, evidence.ErrRecordingFailed) || strings.Contains(err.Error(), "TOKEN")) {
		t.Fatal("required failure lost or exposed sink error", err)
	}
	if mode == evidence.BestEffort && err != nil {
		t.Fatal("best effort changed retrieval outcome", err)
	}
	if _, snapshotErr := result.Receipt.Record.Snapshot(); snapshotErr != nil {
		t.Fatal("required retrieval fact was lost", snapshotErr)
	}
}
