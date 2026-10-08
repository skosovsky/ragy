//go:build !integration && !e2e

package tooling_test

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

func TestHistoricalWireSchemas(t *testing.T) {
	// Arrange: preserve the exact positive/negative inputs of the retired validators.
	profiles := []struct{ task, name, schema, directory string }{
		{"task12", "coverage", "read_coverage", "v1"},
		{"task12", "evidence", "evidence", "v1"},
		{"task12", "lifecycle", "lifecycle", "v1"},
		{"task12", "locator", "locator", "v1"},
		{"task12", "tensor_run", "tensor-run", "v1"},
		{"task18", "lifecycle", "lifecycle", "lifecycle-v2"},
	}
	for _, profile := range profiles {
		t.Run(profile.task+"/"+profile.name, func(t *testing.T) {
			compiler := jsonschema.NewCompiler()
			compiler.AssertFormat()
			base := filepath.Join("testdata", "contracts", profile.directory)
			if profile.name == "tensor_run" {
				doc := decode(t, read(t, filepath.Join(base, "evidence.schema.json")))
				if err := compiler.AddResource(doc.(map[string]any)["$id"].(string), doc); err != nil {
					t.Fatal(err)
				}
			}
			schema, err := compiler.Compile(filepath.Join(base, profile.schema+".schema.json"))
			if err != nil {
				t.Fatal(err)
			}
			var cases []struct {
				Valid bool            `json:"valid"`
				Value json.RawMessage `json:"value"`
			}
			if err := json.Unmarshal(
				read(t, filepath.Join("testdata", "schemas", profile.task+"-verify_"+profile.name+"_schema.json")),
				&cases,
			); err != nil {
				t.Fatal(err)
			}
			checkSchemaCases(t, schema, cases)
		})
	}
}

func TestEvidenceV2SchemaCorpus(t *testing.T) {
	// Arrange.
	root := repoRoot(t)
	compiler := jsonschema.NewCompiler()
	schema, err := compiler.Compile(filepath.Join("testdata", "contracts", "evidence-v2", "evidence.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name   string          `json:"name"`
		Valid  bool            `json:"valid"`
		Record json.RawMessage `json:"record"`
	}
	if err := json.Unmarshal(
		read(t, filepath.Join(root, "evidence/testdata/evidence-v2.json")),
		&cases,
	); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			record := decode(t, c.Record).(map[string]any)
			// Act.
			accepted := schema.Validate(record) == nil && validDecision(record)
			// Assert.
			if accepted != c.Valid {
				t.Fatalf("accepted=%v, expected=%v", accepted, c.Valid)
			}
		})
	}
}

func validDecision(record map[string]any) bool {
	decision, ok := record["decision"].(map[string]any)
	if !ok || decision["state"] != "observed" {
		return true
	}
	queries, _ := decision["queries"].([]any)
	selected, _ := decision["selected"].([]any)
	for i, q := range queries {
		if number(q.(map[string]any)["index"]) != i {
			return false
		}
	}
	for i, s := range selected {
		row := s.(map[string]any)
		if number(row["index"]) != i {
			return false
		}
		contributors, _ := row["contributors"].([]any)
		for _, c := range contributors {
			ordinal := number(c.(map[string]any)["query_index"])
			if ordinal < 0 || ordinal >= len(queries) {
				return false
			}
			q := queries[ordinal].(map[string]any)
			if q["selected"] != true ||
				(row["delivered"] == true && row["uncertain"] != true && q["delivered"] != true) {
				return false
			}
		}
	}
	return true
}

func number(value any) int {
	n, ok := value.(json.Number)
	if !ok {
		return -1
	}
	i, err := n.Int64()
	if err != nil {
		return -1
	}
	return int(i)
}

func TestTensorRecordedAssociation(t *testing.T) {
	// Arrange: real frozen consumer envelope and independently recorded stages.
	base := filepath.Join("testdata", "recordings")
	fixture := decode(t, read(t, filepath.Join(base, "tensor-recorded-run.json"))).(map[string]any)
	record := fixture["record"].(map[string]any)
	// Act / Assert: candidate identity, rank and score survive recording exactly.
	assertJSONEqual(t, record, decode(t, read(t, filepath.Join(base, "tensor-record.json"))))
	if number(fixture["candidate_budget"]) != 100 {
		t.Fatal("candidate budget changed")
	}
	for key, field := range map[string]string{"configuration": "recipe", "scope": "scope", "publication": "publication"} {
		assertJSONEqual(t, record[field], map[string]any{"state": "observed", "value": fixture[key]})
	}
	stages := record["stages"].([]any)
	names := []string{"dense-candidates", "tensor-candidate-observations", "maxsim"}
	for i, stage := range stages {
		row := stage.(map[string]any)
		if row["status"] != "observed" || row["hits_state"] != "observed" ||
			row["name"].(map[string]any)["value"] != names[i] {
			t.Fatal("stage identity changed")
		}
	}
	hits := func(i int) []any { return stages[i].(map[string]any)["hits"].([]any) }
	for key, i := range map[string]int{"candidate_document_ids": 0, "candidate_ids": 1} {
		ids := []any{}
		for _, h := range hits(i) {
			ids = append(ids, h.(map[string]any)["id"].(map[string]any)["value"])
		}
		if len(ids) != 3 {
			t.Fatal("candidate count changed")
		}
		assertJSONEqual(t, ids, fixture[key])
	}
	byID := map[any]any{}
	for _, h := range hits(1) {
		byID[h.(map[string]any)["id"].(map[string]any)["value"]] = h
	}
	expected := []int{2, 1, -1}
	for i, h := range hits(2) {
		hit := h.(map[string]any)
		id := hit["id"].(map[string]any)["value"]
		assertJSONEqual(t, h, byID[id])
		delete(byID, id)
		if number(hit["score"].(map[string]any)["value"]) != expected[i] {
			t.Fatal("oracle score changed")
		}
	}
}

func assertJSONEqual(t *testing.T, a, b any) {
	t.Helper()
	left, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	right, err := json.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	if string(left) != string(right) {
		t.Fatal("JSON association differs")
	}
}

func checkSchemaCases(t *testing.T, schema *jsonschema.Schema, cases []struct {
	Valid bool            `json:"valid"`
	Value json.RawMessage `json:"value"`
}) {
	t.Helper()
	// Act / Assert: reject negative wire shapes, accept positive ones.
	for i, c := range cases {
		if err := schema.Validate(decode(t, c.Value)); (err == nil) != c.Valid {
			t.Errorf("case %d valid=%v: %v", i, c.Valid, err)
		}
	}
}

func decode(t *testing.T, data []byte) any {
	t.Helper()
	var value any
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		t.Fatal(err)
	}
	return value
}
