package source

import (
	"fmt"
	"strings"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
)

// Precision distinguishes exact source byte mapping from support-only evidence.
type Precision string

const (
	ExactPrecision       Precision = "exact"
	UnavailablePrecision Precision = "unavailable"
)

// ContentOrigin distinguishes source text from derived context/description.
type ContentOrigin string

const (
	OriginalContent ContentOrigin = "original"
	DerivedContent  ContentOrigin = "derived"
)

// MappedFragment relates an interval in rendered text to retained evidence.
// Location is active only for exact mapping; Supports always retains all source
// references even when a transformation cannot supply precise byte mapping.
type MappedFragment struct {
	Rendered  ByteSpan      `json:"rendered"`
	Precision Precision     `json:"precision"`
	Origin    ContentOrigin `json:"origin"`
	Location  Locator       `json:"location"`
	Supports  []Locator     `json:"supports"`
}

// MappedText owns text and all supporting locators. Its private snapshot prevents
// enrichment/truncation consumers from changing retained source coordinates.
//

type MappedText struct {
	text      string
	fragments []MappedFragment
}

// OriginalText extracts one exact span from the retained source representation.
func OriginalText(location Locator, retainedText string) (MappedText, error) {
	if !utf8.ValidString(retainedText) {
		return MappedText{}, invalidLocation()
	}
	return originalValidText(location, retainedText)
}

// OriginalTexts returns an owned ordered batch or no payload on any invalid locator
// or span. UTF-8 is checked once for the supplied retained snapshot. Like OriginalText,
// this validates structure; host authorization and source authenticity remain required.
func OriginalTexts(locations []Locator, retainedText string) ([]MappedText, error) {
	if len(locations) == 0 {
		return nil, nil
	}
	if !utf8.ValidString(retainedText) {
		return nil, invalidLocation()
	}
	parts := make([]MappedText, len(locations))
	for i, location := range locations {
		part, err := originalValidText(location, retainedText)
		if err != nil {
			return nil, err
		}
		parts[i] = part
	}
	return parts, nil
}

func originalValidText(location Locator, retainedText string) (MappedText, error) {
	if err := location.Validate(); err != nil {
		return MappedText{}, err
	}
	if location.Kind != TextLocation {
		return MappedText{}, invalidLocation()
	}
	if err := location.Span.validateValidText(retainedText); err != nil {
		return MappedText{}, err
	}
	text := retainedText[location.Span.Start:location.Span.End]
	return MappedText{text: text, fragments: []MappedFragment{{
		Rendered: ByteSpan{Start: 0, End: len(text)}, Precision: ExactPrecision, Origin: OriginalContent,
		Location: location, Supports: []Locator{location},
	}}}, nil
}

// DerivedText attaches a support-only description without claiming it is verbatim
// original source text. Supports cannot be empty or contain invalid locators.
func DerivedText(text string, supports []Locator) (MappedText, error) {
	return supportedText(text, DerivedContent, supports)
}

// SupportedOriginalText retains an original logical cell/region quote without
// pretending it has an exact byte span. Its source supports remain explicit.
func SupportedOriginalText(text string, supports []Locator) (MappedText, error) {
	return supportedText(text, OriginalContent, supports)
}

func supportedText(text string, origin ContentOrigin, supports []Locator) (MappedText, error) {
	if text == "" || !utf8.ValidString(text) {
		return MappedText{}, invalidLocation()
	}
	if err := validateSupports(supports); err != nil {
		return MappedText{}, err
	}
	var inactive Locator
	return MappedText{
		text: text,
		fragments: []MappedFragment{
			{
				Rendered:  ByteSpan{Start: 0, End: len(text)},
				Precision: UnavailablePrecision,
				Origin:    origin,
				Location:  inactive,
				Supports:  uniqueSupports(supports),
			},
		},
	}, nil
}

// Text returns the rendered text, whose byte offsets are separate from source offsets.
func (m MappedText) Text() string { return m.text }

// Fragments returns an owned copy of every mapping/support slice.
func (m MappedText) Fragments() []MappedFragment {
	out := make([]MappedFragment, len(m.fragments))
	for i, fragment := range m.fragments {
		out[i] = fragment
		out[i].Supports = append([]Locator(nil), fragment.Supports...)
	}
	return out
}

// Supports returns unique source locations in encounter order, without ranks.
func (m MappedText) Supports() []Locator {
	var supports []Locator
	for _, fragment := range m.fragments {
		supports = append(supports, fragment.Supports...)
	}
	return uniqueSupports(supports)
}

// Validate checks complete, ordered coverage and the exact/derived distinction.
// Zero MappedText means absent mapping and must not be treated as exact evidence.
func (m MappedText) Validate() error {
	if m.text == "" || !utf8.ValidString(m.text) || len(m.fragments) == 0 {
		return invalidLocation()
	}
	previous := 0
	for _, fragment := range m.fragments {
		if fragment.Rendered.Start != previous {
			return invalidLocation()
		}
		if err := fragment.Rendered.validateValidText(m.text); err != nil {
			return err
		}
		if err := validateFragment(fragment); err != nil {
			return err
		}
		previous = fragment.Rendered.End
	}
	if previous != len(m.text) {
		return invalidLocation()
	}
	return nil
}

func validateFragment(fragment MappedFragment) error {
	if err := validateSupports(fragment.Supports); err != nil {
		return err
	}
	switch fragment.Precision {
	case ExactPrecision:
		if fragment.Origin != OriginalContent || fragment.Location.Kind != TextLocation {
			return invalidLocation()
		}
		if err := fragment.Location.Validate(); err != nil {
			return err
		}
		if fragment.Location.Span.End-fragment.Location.Span.Start != fragment.Rendered.End-fragment.Rendered.Start {
			return invalidLocation()
		}
		found := false
		for _, support := range fragment.Supports {
			if support == fragment.Location {
				found = true
			}
		}
		if !found {
			return invalidLocation()
		}
	case UnavailablePrecision:
		var inactive Locator
		if fragment.Location != inactive || (fragment.Origin != OriginalContent && fragment.Origin != DerivedContent) {
			return invalidLocation()
		}
	default:
		return invalidLocation()
	}
	return nil
}

// Slice truncates at rendered UTF-8 byte boundaries. Exact source spans are
// recalculated; derived content remains support-only with unavailable precision.
func (m MappedText) Slice(span ByteSpan) (MappedText, error) {
	if err := m.Validate(); err != nil {
		return MappedText{}, err
	}
	if err := span.ValidateText(m.text); err != nil {
		return MappedText{}, err
	}
	out := MappedText{text: m.text[span.Start:span.End], fragments: nil}
	for _, fragment := range m.fragments {
		start, end := max(span.Start, fragment.Rendered.Start), min(span.End, fragment.Rendered.End)
		if start >= end {
			continue
		}
		owned := fragment
		owned.Rendered = ByteSpan{Start: start - span.Start, End: end - span.Start}
		owned.Supports = append([]Locator(nil), fragment.Supports...)
		if fragment.Precision == ExactPrecision {
			owned.Location.Span = ByteSpan{
				Start: fragment.Location.Span.Start + start - fragment.Rendered.Start,
				End:   fragment.Location.Span.Start + end - fragment.Rendered.Start,
			}
			// Preserve the full original support as well as the narrowed exact citation.
			owned.Supports = uniqueSupports(append(owned.Supports, owned.Location))
		}
		out.fragments = append(out.fragments, owned)
	}
	return out, out.Validate()
}

// JoinMapped joins multiple retained/derived fragments. Separator bytes are
// explicitly derived support-only context, never attributed to one source span.
func JoinMapped(separator string, parts ...MappedText) (MappedText, error) {
	if len(parts) == 0 || !utf8.ValidString(separator) {
		return MappedText{}, invalidLocation()
	}
	for _, part := range parts {
		if err := part.Validate(); err != nil {
			return MappedText{}, err
		}
	}
	var builder strings.Builder
	out := MappedText{text: "", fragments: nil}
	for i, part := range parts {
		if i > 0 && separator != "" {
			supports := uniqueSupports(append(parts[i-1].Supports(), part.Supports()...))
			derived, err := DerivedText(separator, supports)
			if err != nil {
				return MappedText{}, err
			}
			appendMapped(&builder, &out, derived)
		}
		appendMapped(&builder, &out, part)
	}
	out.text = builder.String()
	return out, out.Validate()
}

func appendMapped(builder *strings.Builder, out *MappedText, part MappedText) {
	offset := builder.Len()
	builder.WriteString(part.text)
	for _, fragment := range part.Fragments() {
		fragment.Rendered.Start += offset
		fragment.Rendered.End += offset
		out.fragments = append(out.fragments, fragment)
	}
}

func validateSupports(supports []Locator) error {
	if len(supports) == 0 {
		return fmt.Errorf("%w: missing source supports", ragy.ErrInvalidArgument)
	}
	for _, support := range supports {
		if err := support.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func uniqueSupports(supports []Locator) []Locator {
	seen := make(map[Locator]struct{}, len(supports))
	out := make([]Locator, 0, len(supports))
	for _, support := range supports {
		if _, exists := seen[support]; exists {
			continue
		}
		seen[support] = struct{}{}
		out = append(out, support)
	}
	return out
}
