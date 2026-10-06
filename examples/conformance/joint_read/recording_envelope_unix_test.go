//go:build darwin || linux

package joint_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/tensor"
)

// These are consumer controls, not an extra core diagnostic vocabulary.
type tensorRunEnvelope struct {
	Schema             string          `json:"schema"`
	Configuration      string          `json:"configuration"`
	Scope              string          `json:"scope"`
	Publication        string          `json:"publication"`
	CandidateBudget    int             `json:"candidate_budget"`
	CandidateIDs       []string        `json:"candidate_ids"`
	CandidateDocuments []string        `json:"candidate_document_ids"`
	Record             evidence.Record `json:"record"`
}

const tensorRunSchema = "ragy.consumer.tensor-record"

func assertTensorRunEnvelope(
	t *testing.T,
	p recorderProfiles,
	stages []evidence.Stage[meta],
	controls tensor.RerankResult,
	record evidence.Record,
) {
	t.Helper()
	var documents []string
	for _, hit := range stages[0].Hits {
		documents = append(documents, hit.Document.ID)
	}
	envelope := tensorRunEnvelope{
		Schema:             tensorRunSchema,
		Configuration:      p.configuration,
		Scope:              p.tensorRead.Snapshot().Identity,
		Publication:        p.tensorRead.Publication().Reference(),
		CandidateBudget:    controls.CandidateBudget,
		CandidateIDs:       slices.Clone(controls.CandidateIDs),
		CandidateDocuments: documents,
		Record:             record,
	}
	wire, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := decodeTensorRunEnvelope(wire)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.CandidateBudget != controls.CandidateBudget ||
		!slices.Equal(decoded.CandidateIDs, controls.CandidateIDs) ||
		!slices.Equal(decoded.CandidateDocuments, documents) {
		t.Fatal("actual candidate controls lost")
	}
	// Producer mutation cannot rewrite the immutable serialized consumer run.
	producerID := controls.CandidateIDs[0]
	controls.CandidateIDs[0] = "mutated-producer"
	restored, err := decodeTensorRunEnvelope(wire)
	if err != nil || restored.CandidateIDs[0] != decoded.CandidateIDs[0] {
		t.Fatal("run envelope aliases producer", err)
	}
	controls.CandidateIDs[0] = producerID
	for _, mutate := range []func(map[string]json.RawMessage){
		func(value map[string]json.RawMessage) { delete(value, "candidate_budget") },
		func(value map[string]json.RawMessage) { value["candidate_budget"] = json.RawMessage("0") },
		func(value map[string]json.RawMessage) { value["publication"] = json.RawMessage(`"other"`) },
		func(value map[string]json.RawMessage) { value["configuration"] = json.RawMessage(`"other"`) },
		func(value map[string]json.RawMessage) { value["unknown"] = json.RawMessage("true") },
	} {
		var value map[string]json.RawMessage
		if err = json.Unmarshal(wire, &value); err != nil {
			t.Fatal(err)
		}
		mutate(value)
		bad, marshalErr := json.Marshal(value)
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		if _, err = decodeTensorRunEnvelope(bad); err == nil {
			t.Fatal("invalid run envelope accepted")
		}
	}
	assertTensorRunResultAssociation(t, wire)
	t.Logf("TASK12_TENSOR_RUN %s", wire)
}

func assertTensorRunResultAssociation(t *testing.T, wire []byte) {
	t.Helper()
	for _, mutate := range []func(map[string]any){
		func(value map[string]any) {
			tensorWireStages(value)[2].(map[string]any)["name"].(map[string]any)["value"] = "other"
		},
		func(value map[string]any) {
			tensorWireHit(tensorWireStages(value), 0)["id"].(map[string]any)["value"] = "outside-candidates"
		},
		func(value map[string]any) {
			tensorWireHit(tensorWireStages(value), 0)["score"].(map[string]any)["value"] = 42
		},
		func(value map[string]any) {
			tensorWireHit(tensorWireStages(value), 0)["rank"].(map[string]any)["value"] = 42
		},
		func(value map[string]any) {
			hits := tensorWireStages(value)[2].(map[string]any)["hits"].([]any)
			hits[1] = hits[0]
		},
		func(value map[string]any) {
			value["configuration"] = "unqualified"
			value["record"].(map[string]any)["recipe"].(map[string]any)["value"] = "unqualified"
		},
		func(value map[string]any) {
			ids := value["candidate_document_ids"].([]any)
			ids[1] = ids[0]
			hits := tensorWireStages(value)[0].(map[string]any)["hits"].([]any)
			hits[1].(map[string]any)["id"].(map[string]any)["value"] = ids[0]
		},
	} {
		// Arrange: a structurally valid record derived from the actual run.
		var value map[string]any
		if err := json.Unmarshal(wire, &value); err != nil {
			t.Fatal(err)
		}
		// Act: alter only final-result association.
		mutate(value)
		bad, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		// Assert: shape alone cannot admit an inconsistent MaxSim observation.
		if _, err = decodeTensorRunEnvelope(bad); err == nil {
			t.Fatal("inconsistent final result accepted")
		}
	}
}

func tensorWireHit(stages []any, index int) map[string]any {
	return stages[2].(map[string]any)["hits"].([]any)[index].(map[string]any)
}
func decodeTensorRunEnvelope(data []byte) (tensorRunEnvelope, error) {
	fields := []string{
		"schema",
		"configuration",
		"scope",
		"publication",
		"candidate_budget",
		"candidate_ids",
		"candidate_document_ids",
		"record",
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return tensorRunEnvelope{}, ragy.ErrProtocol
	}
	seen := make(map[string]bool)
	for decoder.More() {
		keyToken, keyErr := decoder.Token()
		key, ok := keyToken.(string)
		if keyErr != nil || !ok || seen[key] || !slices.Contains(fields, key) {
			return tensorRunEnvelope{}, ragy.ErrProtocol
		}
		seen[key] = true
		var raw json.RawMessage
		if err = decoder.Decode(&raw); err != nil {
			return tensorRunEnvelope{}, ragy.ErrProtocol
		}
	}
	if len(seen) != len(fields) {
		return tensorRunEnvelope{}, ragy.ErrProtocol
	}
	decoder = json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var value tensorRunEnvelope
	if err = decoder.Decode(&value); err != nil {
		return tensorRunEnvelope{}, ragy.ErrProtocol
	}
	var trailing any
	if !errors.Is(decoder.Decode(&trailing), io.EOF) {
		return tensorRunEnvelope{}, ragy.ErrProtocol
	}
	if err = validateTensorRunEnvelope(value); err != nil {
		return tensorRunEnvelope{}, err
	}
	return value, nil
}
func validateTensorRunEnvelope(value tensorRunEnvelope) error {
	if value.Schema != tensorRunSchema {
		return ragy.ErrUnsupported
	}
	digest, digestErr := hex.DecodeString(value.Configuration)
	if digestErr != nil || len(digest) != sha256.Size || hex.EncodeToString(digest) != value.Configuration {
		return ragy.ErrProtocol
	}
	if value.CandidateBudget <= 0 || len(value.CandidateIDs) == 0 || len(value.CandidateIDs) > value.CandidateBudget ||
		len(value.CandidateIDs) != len(value.CandidateDocuments) {
		return ragy.ErrProtocol
	}
	snapshot, err := value.Record.Snapshot()
	if err != nil || value.Configuration == "" || value.Scope == "" || value.Publication == "" {
		return ragy.ErrProtocol
	}
	for _, pair := range [][2]string{{value.Configuration, observedText(snapshot.Recipe)}, {value.Scope, observedText(snapshot.Scope)}, {value.Publication, observedText(snapshot.Publication)}} {
		if pair[0] != pair[1] {
			return ragy.ErrProtocol
		}
	}
	if len(snapshot.Stages) != 3 || len(snapshot.Stages[0].Hits) != len(value.CandidateIDs) {
		return ragy.ErrProtocol
	}
	if len(snapshot.Stages[1].Hits) != len(value.CandidateIDs) {
		return ragy.ErrProtocol
	}
	seen := make(map[string]bool)
	denseIDs := make(map[string]bool)
	for i, id := range value.CandidateIDs {
		denseID := value.CandidateDocuments[i]
		if id == "" || seen[id] || denseID == "" || denseIDs[denseID] ||
			denseID != observedText(snapshot.Stages[0].Hits[i].ID) ||
			id != observedText(snapshot.Stages[1].Hits[i].ID) {
			return ragy.ErrProtocol
		}
		seen[id] = true
		denseIDs[denseID] = true
	}
	return validateTensorRunStages(snapshot)
}

func validateTensorRunStages(snapshot evidence.Snapshot) error {
	names := []string{"dense-candidates", "tensor-candidate-observations", "maxsim"}
	for i, stage := range snapshot.Stages {
		if stage.Index != i || observedText(stage.Name) != names[i] ||
			stage.Status != evidence.StageObserved || stage.HitsState != evidence.Observed {
			return ragy.ErrProtocol
		}
	}
	byID := make(map[string]evidence.WireHit)
	matchedDense := make(map[int]bool)
	for _, hit := range snapshot.Stages[1].Hits {
		id := observedText(hit.ID)
		if hit.Score.State != evidence.ScoreNative || hit.Rank.State != evidence.Observed {
			return ragy.ErrProtocol
		}
		byID[id] = hit
		found := false
		for i, denseHit := range snapshot.Stages[0].Hits {
			if !matchedDense[i] && reflect.DeepEqual(hit.Sources, denseHit.Sources) {
				matchedDense[i], found = true, true
				break
			}
		}
		if !found {
			return ragy.ErrProtocol
		}
	}
	seen := make(map[string]bool)
	for _, hit := range snapshot.Stages[2].Hits {
		id := observedText(hit.ID)
		candidate, ok := byID[id]
		if !ok || seen[id] || !reflect.DeepEqual(candidate, hit) {
			return ragy.ErrProtocol
		}
		seen[id] = true
	}
	return nil
}
func observedText(value evidence.Text) string {
	if value.State != evidence.Observed || value.Value == nil {
		return ""
	}
	return *value.Value
}

func tensorWireStages(value map[string]any) []any {
	return value["record"].(map[string]any)["stages"].([]any)
}
