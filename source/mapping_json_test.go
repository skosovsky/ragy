package source_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func TestMappingJSONRoundTripAndAbsentPrecision(t *testing.T) {
	// Arrange.
	original := originalBeta(t)
	// Act.
	encoded, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	var decoded source.MappedText
	if decodeErr := json.Unmarshal(encoded, &decoded); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	var absent source.MappedText
	missing, err := json.Marshal(absent)
	if err != nil {
		t.Fatal(err)
	}
	// Assert.
	if decoded.Text() != "beta" || decoded.Fragments()[0].Location != original.Fragments()[0].Location {
		t.Fatal("mapping lost on wire")
	}
	if !strings.Contains(string(missing), `"state":"unobserved"`) || strings.Contains(string(missing), `"exact"`) {
		t.Fatal("absent mapping became exact")
	}
	exported := decoded.Fragments()
	exported[0].Supports[0].Reference.Revision = "r2"
	if decoded.Supports()[0].Reference.Revision != "r1" {
		t.Fatal("decoded supports not owned")
	}
}

func TestMappingJSONRejectsInvalidWireAndPreservesOldSnapshot(t *testing.T) {
	valid, err := json.Marshal(originalBeta(t))
	if err != nil {
		t.Fatal(err)
	}
	tests := []string{
		strings.Replace(string(valid), `"ragy.source-mapping"`, `"unknown"`, 1),
		strings.Replace(string(valid), `"state":"mapped"`, `"state":"unobserved"`, 1),
		strings.Replace(string(valid), `"precision":"exact"`, `"precision":"guessed"`, 1),
		strings.Replace(string(valid), `"origin":"original"`, `"origin":"derived"`, 1),
		strings.Replace(string(valid), `"text":"beta"`, `"text":"b"`, 1),
		strings.Replace(string(valid), `"rendered":{"start":0`, `"rendered":{"start":1`, 1),
		strings.Replace(string(valid), `"namespace":"docs"`, `"namespace":"docs","unexpected":true`, 1),
		string(valid) + ` {}`,
		`null`,
		`{"schema":"ragy.source-mapping","state":"unobserved","text":"private","fragments":[]}`,
	}
	for _, input := range tests {
		// Arrange.
		existing := originalBeta(t)
		// Act.
		decodeErr := json.Unmarshal([]byte(input), &existing)
		// Assert.
		if decodeErr == nil {
			t.Fatalf("invalid mapping accepted: %s", input)
		}
		if existing.Text() != "beta" || existing.Supports()[0].Reference.Revision != "r1" {
			t.Fatal("failed decode replaced retained snapshot")
		}
	}
}
