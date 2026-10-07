package source_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func TestMappingJSONOrdinaryCodecPolicy(t *testing.T) {
	// Arrange: current structural codec deliberately differs from strict locator envelopes.
	encoded, err := json.Marshal(originalBeta(t))
	if err != nil {
		t.Fatal(err)
	}
	inputs := []string{
		strings.Replace(string(encoded), `"text":"beta"`, `"text":"ignored","text":"beta"`, 1),
		strings.Replace(string(encoded), `"text":"beta"`, `"Text":"beta"`, 1),
		`{"schema":"ragy.source-mapping","state":"unobserved"}`,
	}
	for i, input := range inputs {
		// Act.
		var decoded source.MappedText
		err = json.Unmarshal([]byte(input), &decoded)
		// Assert: last duplicate wins, aliases accepted, omitted zero values valid.
		want := "beta"
		if i == 2 {
			want = ""
		}
		if err != nil || decoded.Text() != want {
			t.Fatal(i, decoded.Text(), err)
		}
	}
	// Arrange/Act: an escaped isolated surrogate is repaired before structural checks.
	mapping, err := source.DerivedText("�", originalBeta(t).Supports())
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = json.Marshal(mapping)
	if err != nil {
		t.Fatal(err)
	}
	var decoded source.MappedText
	err = json.Unmarshal([]byte(strings.Replace(string(encoded), `"text":"�"`, `"text":"\ud800"`, 1)), &decoded)
	// Assert: this is structural support-only data, never authenticated original text.
	if err != nil || decoded.Text() != "�" || decoded.Fragments()[0].Precision != source.UnavailablePrecision {
		t.Fatal(decoded.Text(), err)
	}
}
