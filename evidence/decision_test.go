package evidence_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/evidence"
)

func TestDecisionPrivacyAndUnavailableRevisions(t *testing.T) {
	read, input, _ := fixture(t)
	sufficient := false
	input.Decision = &evidence.DecisionInput{
		Queries: []evidence.DecisionQueryInput{
			{Index: 0, Text: "secret variant", Retrieved: true, Selected: true, Delivered: false, Uncertain: false},
		},
		Selected: []evidence.DecisionSelection{
			{
				Index:        0,
				Contributors: []evidence.DecisionContributor{{QueryIndex: 0, Rank: 1}},
				Delivered:    false,
				Uncertain:    false,
			},
		},
		Fusion:      evidence.StageObserved,
		Sufficiency: &sufficient,
		Stop:        "assessed",
		Revisions: evidence.HostRevisions{
			Model:  "secret model",
			Prompt: "secret prompt",
			Config: "secret config",
			Recipe: "secret recipe",
		},
	}
	policy := evidence.Policy{}
	record, err := evidence.Capture(context.Background(), read, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := record.Snapshot()
	if snapshot.Decision.State != evidence.Omitted {
		t.Fatalf("default captured decisions: %+v", snapshot.Decision)
	}
	policy.AllowDecisions = true
	record, err = evidence.Capture(context.Background(), read, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := record.MarshalJSON()
	snapshot, _ = record.Snapshot()
	if bytes.Contains(data, []byte("secret")) || snapshot.Decision.Revisions.Model.State != evidence.Omitted ||
		snapshot.Decision.Sufficiency == nil ||
		*snapshot.Decision.Sufficiency {
		t.Fatalf("privacy/decision mismatch: %s", data)
	}
	input.Decision.Revisions.Model = ""
	policy.AllowQuery = true
	policy.AllowIdentifier = func(kind evidence.IdentifierKind, _ string) bool { return kind == evidence.PromptIdentifier }
	record, err = evidence.Capture(context.Background(), read, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ = record.Snapshot()
	if snapshot.Decision.Queries[0].Text.State != evidence.Observed ||
		snapshot.Decision.Revisions.Model.State != evidence.Unavailable ||
		snapshot.Decision.Revisions.Prompt.State != evidence.Observed ||
		snapshot.Decision.Revisions.Config.State != evidence.Omitted {
		t.Fatal(snapshot.Decision)
	}
}
func TestDecisionIndependentSchemaCorpus(t *testing.T) {
	data, err := os.ReadFile("../docs/task17/fixtures/evidence-v2.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name   string          `json:"name"`
		Valid  bool            `json:"valid"`
		Record json.RawMessage `json:"record"`
	}
	if err = json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, item := range cases {
		t.Run(item.Name, func(t *testing.T) {
			_, decodeErr := evidence.Decode(item.Record)
			if (decodeErr == nil) != item.Valid {
				t.Fatalf("valid=%v err=%v", item.Valid, decodeErr)
			}
			if item.Name == "old" && !errors.Is(decodeErr, ragy.ErrUnsupported) {
				t.Fatal(decodeErr)
			}
		})
	}
}
func TestDecodeFiniteBytesDepthDuplicateFields(t *testing.T) {
	cases := [][]byte{
		bytes.Repeat([]byte(" "), evidence.MaxRecordBytes+1),
		[]byte(strings.Repeat("[", evidence.MaxRecordDepth+2) + "0" + strings.Repeat("]", evidence.MaxRecordDepth+2)),
		[]byte(`{"schema":"x","schema":"x"}`),
	}
	for _, data := range cases {
		if _, err := evidence.Decode(data); !errors.Is(err, ragy.ErrProtocol) {
			t.Fatalf("expected bounded rejection got %v", err)
		}
	}
}
