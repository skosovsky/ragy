package lexical

import "testing"

func TestSynonymMapExpandsTokens(t *testing.T) {
	t.Parallel()

	synonyms := SynonymMap{"car": {"automobile", "vehicle"}}
	out := synonyms.Expand([]string{"car", "drive"})
	if len(out) != 4 {
		t.Fatalf("Expand() = %#v, want 4 tokens", out)
	}
}
