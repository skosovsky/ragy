package lexical

import (
	"regexp"
	"strings"
	"unicode"
)

var punctPattern = regexp.MustCompile(`[^\p{L}\p{N}\s]+`)

// Tokenizer splits text into normalized search tokens.
type Tokenizer interface {
	Tokenize(text string) []string
}

// DefaultTokenizer lowercases, strips punctuation, and splits on whitespace.
type DefaultTokenizer struct{}

// Tokenize implements Tokenizer.
func (DefaultTokenizer) Tokenize(text string) []string {
	text = strings.ToLower(text)
	text = punctPattern.ReplaceAllString(text, " ")
	fields := strings.FieldsFunc(text, unicode.IsSpace)
	if len(fields) == 0 {
		return nil
	}
	return fields
}
