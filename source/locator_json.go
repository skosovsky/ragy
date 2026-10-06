package source

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
)

// LocatorSchema identifies the persisted locator envelope, not a software release.
const LocatorSchema = "ragy.source-locator"

const maxLocatorEnvelopeDepth = 8

type locatorEnvelope struct {
	Schema   string  `json:"schema"`
	Location Locator `json:"location"`
}

// EncodeLocator validates and serializes a canonical value-only locator. Host
// extensions and authorization decisions remain outside this wire envelope.
func EncodeLocator(location Locator) ([]byte, error) {
	if err := location.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(locatorEnvelope{Schema: LocatorSchema, Location: location})
}

// DecodeLocator rejects unknown, duplicate and missing fields and incompatible
// schema identities. It validates geometry, not source authenticity or access.
// The host must still resolve the exact retained reference under its read binding.
func DecodeLocator(data []byte) (Locator, error) {
	var empty Locator
	if !utf8.Valid(data) {
		return empty, ragy.ErrProtocol
	}
	tokens := json.NewDecoder(bytes.NewReader(data))
	if err := uniqueLocatorValue(tokens, 0); err != nil {
		return empty, ragy.ErrProtocol
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var envelope locatorEnvelope
	if err := decoder.Decode(&envelope); err != nil {
		return empty, ragy.ErrProtocol
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return empty, ragy.ErrProtocol
	}
	if envelope.Schema != LocatorSchema {
		return empty, ragy.ErrUnsupported
	}
	encoded, err := EncodeLocator(envelope.Location)
	if err != nil {
		return empty, err
	}
	var supplied, canonical any
	if json.Unmarshal(data, &supplied) != nil || json.Unmarshal(encoded, &canonical) != nil ||
		!locatorShape(supplied, canonical) {
		return empty, ragy.ErrProtocol
	}
	return envelope.Location, nil
}

// Locator envelopes have no arrays and a fixed shallow object tree.
func uniqueLocatorValue(decoder *json.Decoder, depth int) error {
	if depth > maxLocatorEnvelopeDepth {
		return ragy.ErrProtocol
	}
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, structured := token.(json.Delim)
	if !structured {
		return nil
	}
	if delimiter != '{' {
		return ragy.ErrProtocol
	}
	seen := map[string]bool{}
	for decoder.More() {
		keyToken, keyErr := decoder.Token()
		if keyErr != nil {
			return keyErr
		}
		key, ok := keyToken.(string)
		if !ok || seen[key] {
			return ragy.ErrProtocol
		}
		seen[key] = true
		if err = uniqueLocatorValue(decoder, depth+1); err != nil {
			return err
		}
	}
	token, err = decoder.Token()
	if err != nil || token != json.Delim('}') {
		return ragy.ErrProtocol
	}
	return nil
}
func locatorShape(supplied, canonical any) bool {
	expected, object := canonical.(map[string]any)
	if !object {
		return supplied != nil
	}
	actual, ok := supplied.(map[string]any)
	if !ok || len(actual) != len(expected) {
		return false
	}
	for key, value := range expected {
		found, exists := actual[key]
		if !exists || !locatorShape(found, value) {
			return false
		}
	}
	return true
}
