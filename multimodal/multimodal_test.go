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
