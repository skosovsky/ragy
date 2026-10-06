package codexcall

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"unicode"
)

const maxJSONDepth = 64

// UniqueJSON rejects ambiguous repeated fields before any typed decoder can
// silently apply last-value-wins. Field matching also catches case aliases
// accepted by Go's struct decoder. It checks every nested object.
func UniqueJSON(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := uniqueValue(decoder, 0); err != nil {
		return ErrExecution
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return ErrExecution
	}
	return nil
}
func uniqueValue(decoder *json.Decoder, depth int) error {
	if depth > maxJSONDepth {
		return ErrExecution
	}
	token, err := decoder.Token()
	if err != nil {
		return ErrExecution
	}
	delimiter, compound := token.(json.Delim)
	if !compound {
		return nil
	}
	switch delimiter {
	case '{':
		return uniqueObject(decoder, depth)
	case '[':
		for decoder.More() {
			if uniqueValue(decoder, depth+1) != nil {
				return ErrExecution
			}
		}
		end, closeErr := decoder.Token()
		if closeErr != nil || end != json.Delim(']') {
			return ErrExecution
		}
	default:
		return ErrExecution
	}
	return nil
}
func uniqueObject(decoder *json.Decoder, depth int) error {
	seen := make(map[string]bool)
	for decoder.More() {
		token, err := decoder.Token()
		key, isString := token.(string)
		if err != nil || !isString || seen[foldedKey(key)] {
			return ErrExecution
		}
		seen[foldedKey(key)] = true
		if uniqueValue(decoder, depth+1) != nil {
			return ErrExecution
		}
	}
	end, err := decoder.Token()
	if err != nil || end != json.Delim('}') {
		return ErrExecution
	}
	return nil
}

// Go's JSON struct matching uses Unicode SimpleFold, including aliases such as
// long-s. Canonicalize each fold cycle in linear key length, rather than scanning
// every previously seen field with quadratic EqualFold comparisons.
func foldedKey(key string) string {
	var canonical strings.Builder
	for _, character := range key {
		smallest := character
		for next := unicode.SimpleFold(character); next != character; next = unicode.SimpleFold(next) {
			if next < smallest {
				smallest = next
			}
		}
		canonical.WriteRune(smallest)
	}
	return canonical.String()
}
