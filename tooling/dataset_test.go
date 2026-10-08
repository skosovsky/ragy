package tooling_test

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

func TestConformanceDataset(t *testing.T) {
	// Arrange: independently compile the corpus/split schema.
	base := filepath.Join(repoRoot(t), "examples/conformance/datasets/task19")
	schema, err := jsonschema.NewCompiler().Compile(filepath.Join(base, "schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	corpus := decode(t, read(t, filepath.Join(base, "corpus.json"))).(map[string]any)
	if err := schema.Validate(corpus); err != nil {
		t.Fatal(err)
	}
	// Act / Assert: schema, reference eligibility, split independence and frozen hashes.
	docs := indexDocuments(t, corpus["documents"].([]any))
	seen := map[string]bool{}
	for _, split := range []string{"dev", "holdout"} {
		value := decode(t, read(t, filepath.Join(base, split+".json"))).(map[string]any)
		if err := schema.Validate(value); err != nil {
			t.Fatal(err)
		}
		validateSplit(t, split, value, docs, seen)
	}
	var manifest struct {
		SHA256 map[string]string `json:"sha256"`
	}
	if err := json.Unmarshal(read(t, filepath.Join(base, "manifest.json")), &manifest); err != nil {
		t.Fatal(err)
	}
	// The retired validator hash remains historical metadata; check all four dataset artifacts.
	for _, name := range []string{"corpus.json", "dev.json", "schema.json", "holdout.json"} {
		digest := manifest.SHA256[name]
		if fmt.Sprintf("%x", sha256.Sum256(read(t, filepath.Join(base, name)))) != digest {
			t.Fatalf("digest changed: %s", name)
		}
	}
}

func indexDocuments(t *testing.T, rows []any) map[string]map[string]any {
	t.Helper()
	if len(rows) < 30 {
		t.Fatal("corpus reduced")
	}
	docs := map[string]map[string]any{}
	revisions := map[string]bool{}
	for _, d := range rows {
		row := d.(map[string]any)
		id := row["id"].(string)
		pair := fmt.Sprint(row["source_id"]) + "\x00" + fmt.Sprint(row["revision"])
		if docs[id] != nil || revisions[pair] {
			t.Fatal("duplicate document identity")
		}
		docs[id] = row
		revisions[pair] = true
	}
	for _, row := range docs {
		for _, r := range row["relations"].([]any) {
			if docs[r.(map[string]any)["target_id"].(string)] == nil {
				t.Fatal("dangling relation")
			}
		}
	}
	return docs
}

func validateSplit(
	t *testing.T,
	split string,
	value map[string]any,
	docs map[string]map[string]any,
	seen map[string]bool,
) {
	t.Helper()
	categories := map[string]int{}
	for _, q := range value["queries"].([]any) {
		row := q.(map[string]any)
		for _, key := range []string{"id", "case_id", "text"} {
			v := row[key].(string)
			k := key + "\x00" + v
			if seen[k] || (key != "text" && !strings.HasPrefix(v, split+"-")) {
				t.Fatalf("invalid query identity %s", v)
			}
			seen[k] = true
		}
		categories[row["category"].(string)]++
		validateGold(t, row, docs)
	}
	for _, category := range []string{"lookup", "multihop", "no_answer", "harmful_rewrite", "duplicates", "conflicting_stale", "limited_scope", "multilingual", "malicious_instructions"} {
		if categories[category] < 2 {
			t.Fatalf("category %s reduced", category)
		}
	}
}

func validateGold(t *testing.T, row map[string]any, docs map[string]map[string]any) {
	t.Helper()
	rels := row["qrels"].([]any)
	if row["answerable"] != (len(rels) > 0) {
		t.Fatal("answerability mismatch")
	}
	gold := map[string]bool{}
	for _, r := range rels {
		id := r.(map[string]any)["document_id"].(string)
		d := docs[id]
		if gold[id] || d == nil || d["current"] != true || (d["scope"] != "public" && d["scope"] != row["scope"]) {
			t.Fatalf("invalid gold %s", id)
		}
		gold[id] = true
	}
}

func TestRecordedCLIReceipts(t *testing.T) {
	// Arrange: historical responses, without model calls.
	base := filepath.Join(repoRoot(t), "docs/task12/results")
	for _, name := range []string{"text-cli-live-capture.json", "graph-cli-live-capture.json", "graph-cli-v2-capture.json", "text-cli-v2-capture.json"} {
		t.Run(name, func(t *testing.T) {
			capture := decode(t, read(t, filepath.Join(base, name))).(map[string]any)
			samples := capture["samples"].([]any)
			if prep, ok := capture["preparation"].(map[string]any); ok {
				if rows, ok := prep["extractions"].([]any); ok {
					samples = append(samples, rows...)
				}
			}
			// Act / Assert.
			for _, s := range samples {
				validateReceiptRow(t, s.(map[string]any))
			}
		})
	}
}

func validateReceiptRow(t *testing.T, row map[string]any) {
	t.Helper()
	calls, _ := row["cli_receipts"].([]any)
	if number(row["model_calls"]) != len(calls) {
		t.Fatal("model call count differs")
	}
	input, output := 0, 0
	for _, c := range calls {
		in, out := validateReceipt(t, c.(map[string]any))
		input += in
		output += out
	}
	if len(calls) > 0 && (number(row["input_tokens"]) != input || number(row["output_tokens"]) != output) {
		t.Fatal("row usage mismatch")
	}
	if _, sample := row["provider_price_state"]; sample && len(calls) > 0 &&
		(row["provider_price_state"] != "unavailable" || row["usage_known"] == true) {
		t.Fatal("unknown pricing lost")
	}
}

func validateReceipt(t *testing.T, call map[string]any) (int, int) {
	t.Helper()
	usage, ok := call["usage"].(map[string]any)
	if !ok || number(call["exit_code"]) != 0 || call["timed_out"] == true || call["tool_activity"] == true {
		t.Fatal("invalid receipt")
	}
	settlements, messages := 0, 0
	for _, e := range call["events"].([]any) {
		event := e.(map[string]any)
		switch event["type"] {
		case "turn.completed":
			settlements++
			u := event["usage"].(map[string]any)
			if number(u["input_tokens"]) != number(usage["input_tokens"]) ||
				number(u["output_tokens"]) != number(usage["output_tokens"]) {
				t.Fatal("settlement mismatch")
			}
		case "item.completed":
			if item, ok := event["item"].(map[string]any); ok && item["type"] == "agent_message" {
				messages++
			}
		}
	}
	if settlements != 1 || messages != 1 {
		t.Fatal("ambiguous receipt")
	}
	return number(usage["input_tokens"]), number(usage["output_tokens"])
}
