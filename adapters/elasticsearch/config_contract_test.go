package elasticsearch

import (
	"reflect"
	"strings"
	"testing"

	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

type changingTokenizer struct{ calls int }

func (t *changingTokenizer) Tokenize(string) []string {
	t.calls++
	if t.calls == 1 {
		return []string{"alias"}
	}
	return nil
}

func TestRetrieveOwnsSynonymsAndTokenizesOnce(t *testing.T) {
	// Arrange: the second tokenization would be empty, unlike the admitted first.
	schema := schemaWithContent(t)
	tokenizer := &changingTokenizer{}
	variants := []string{"needle"}
	synonyms := lexical.SynonymMap{"alias": variants}
	fields := []string{"content"}
	client := &fakeClient{}
	store, err := New(
		client,
		Config[contracttest.StructMeta]{
			Index:        "docs",
			SearchFields: fields,
			Schema:       schema,
			Synonyms:     synonyms,
			Tokenizer:    tokenizer,
		},
		structMetaCodec(t, schema),
	)
	if err != nil {
		t.Fatal(err)
	}
	expected := lexical.SynonymMap{"alias": []string{"needle"}}.Expand([]string{"alias"})
	// Act: mutate all caller-owned configuration before retrieval.
	variants[0] = "changed"
	synonyms["alias"] = []string{"replacement"}
	fields[0] = "unknown"
	result, err := retrieveStore(t.Context(), store, "input", retrieval.RetrieveOptions{TopK: 1})
	// Assert: exact first expansion goes to wire, with construction-time fields.
	if err != nil || result.Len() != 0 || tokenizer.calls != 1 {
		t.Fatal("single dispatch", result, err, tokenizer.calls)
	}
	match := client.body["query"].(map[string]any)["multi_match"].(map[string]any)
	if !reflect.DeepEqual(match["fields"], []string{"content"}) || match["query"] != strings.Join(expected, " ") {
		t.Fatal("borrowed config or re-tokenization", match)
	}
}
