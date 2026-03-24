// Package multimodal provides multimodal embedding contracts.
package multimodal

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	ragy "github.com/skosovsky/ragy"
)

// PartKind identifies a multimodal input part.
type PartKind string

const (
	PartText  PartKind = "text"
	PartBytes PartKind = "bytes"
	PartURL   PartKind = "url"
)

// Part is a single multimodal input segment.
type Part struct {
	Kind  PartKind
	MIME  string
	Text  string
	Bytes []byte
	URL   string
}

// Validate checks part invariants.
func (p Part) Validate() error {
	text := strings.TrimSpace(p.Text)
	mime := strings.TrimSpace(p.MIME)
	rawURL := strings.TrimSpace(p.URL)

	switch p.Kind {
	case PartText:
		return validateTextPart(text, rawURL, mime, len(p.Bytes))
	case PartBytes:
		return validateBytesPart(text, rawURL, mime, len(p.Bytes))
	case PartURL:
		return validateURLPart(text, rawURL, mime, len(p.Bytes))
	default:
		return fmt.Errorf("%w: unknown multimodal part kind %q", ragy.ErrInvalidArgument, p.Kind)
	}
}

// Input is a multimodal embedding input.
type Input struct {
	Parts []Part
}

// Validate checks input invariants.
func (i Input) Validate() error {
	if len(i.Parts) == 0 {
		return fmt.Errorf("%w: multimodal input parts", ragy.ErrInvalidArgument)
	}

	for _, part := range i.Parts {
		if err := part.Validate(); err != nil {
			return err
		}
	}

	return nil
}

func validateTextPart(text, rawURL, mime string, bytesLen int) error {
	if text == "" {
		return fmt.Errorf("%w: multimodal text part", ragy.ErrEmptyText)
	}
	if bytesLen > 0 || rawURL != "" || mime != "" {
		return fmt.Errorf("%w: text part must only set Text", ragy.ErrInvalidArgument)
	}
	return nil
}

func validateBytesPart(text, rawURL, mime string, bytesLen int) error {
	if bytesLen == 0 {
		return fmt.Errorf("%w: multimodal bytes part", ragy.ErrInvalidArgument)
	}
	if mime == "" {
		return fmt.Errorf("%w: bytes part mime", ragy.ErrInvalidArgument)
	}
	if text != "" || rawURL != "" {
		return fmt.Errorf("%w: bytes part must only set Bytes and MIME", ragy.ErrInvalidArgument)
	}
	return nil
}

func validateURLPart(text, rawURL, mime string, bytesLen int) error {
	if rawURL == "" {
		return fmt.Errorf("%w: multimodal url part", ragy.ErrInvalidArgument)
	}
	parsed, err := url.Parse(rawURL)
	if err != nil || !parsed.IsAbs() {
		return fmt.Errorf("%w: multimodal url part", ragy.ErrInvalidArgument)
	}
	if text != "" || bytesLen > 0 || mime != "" {
		return fmt.Errorf("%w: url part must only set URL", ragy.ErrInvalidArgument)
	}
	return nil
}

// Embedder produces multimodal embeddings.
type Embedder interface {
	Embed(ctx context.Context, inputs []Input) ([][]float32, error)
}
