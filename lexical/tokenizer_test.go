package lexical

import "testing"

func TestDefaultTokenizerSplitsWords(t *testing.T) {
	t.Parallel()

	tokens := DefaultTokenizer{}.Tokenize("Hello, BM25 world!")
	if len(tokens) != 3 || tokens[0] != "hello" || tokens[2] != "world" {
		t.Fatalf("Tokenize() = %#v", tokens)
	}
}
