package multimodal

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestPartValidateRejectsMixedPayload(t *testing.T) {
	part := Part{
		Kind:  PartText,
		Text:  "hello",
		Bytes: []byte("bad"),
	}
	if err := part.Validate(); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Part.Validate() error = %v", err)
	}
}

func TestInputValidateRequiresAtLeastOnePart(t *testing.T) {
	if err := (Input{}).Validate(); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Input.Validate() error = %v", err)
	}
}

func TestInputValidateAcceptsValidKinds(t *testing.T) {
	inputs := []Input{
		{Parts: []Part{{Kind: PartText, Text: "hello"}}},
		{Parts: []Part{{Kind: PartBytes, MIME: "image/png", Bytes: []byte{1, 2, 3}}}},
		{Parts: []Part{{Kind: PartURL, URL: "https://example.com/image.png"}}},
	}

	for _, input := range inputs {
		if err := input.Validate(); err != nil {
			t.Fatalf("Input.Validate() error = %v", err)
		}
	}
}

func TestPartLiteralInactiveFieldsAndTransportNeutralURL(t *testing.T) {
	for _, part := range []Part{
		{Kind: PartText, Text: string([]byte{0xff})},
		{Kind: PartText, Text: "text", MIME: " "},
		{Kind: PartText, Text: "text", URL: " "},
		{Kind: PartBytes, Bytes: []byte{1}, MIME: "image/png", Text: " "},
		{Kind: PartURL, URL: "https://example.test/a", Text: " "},
	} {
		// Arrange/Act/Assert: inactive whitespace is payload, not absence.
		if !errors.Is(part.Validate(), ragy.ErrInvalidArgument) {
			t.Fatal(part.Kind)
		}
	}
	part := Part{Kind: PartURL, URL: " custom://host/value "}
	before := part.URL
	if err := part.Validate(); err != nil || part.URL != before {
		t.Fatal(part, err)
	}
}
