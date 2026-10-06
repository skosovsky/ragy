package layout

import (
	"context"
	"unicode/utf8"

	"github.com/skosovsky/ragy/source"
)

// OCRState distinguishes unobserved capability from explicit engine outcomes.
type OCRState string

const (
	OCRUnobserved  OCRState = ""
	OCRRecognized  OCRState = "recognized"
	OCRUnreadable  OCRState = "unreadable"
	OCRUnsupported OCRState = "unsupported"
)

// OCR diagnostic codes are stable non-content outcomes.
const (
	DiagnosticOCRUnprocessed = "ocr_unprocessed"
	DiagnosticOCRUnreadable  = "ocr_region_unreadable"
	DiagnosticOCRUnsupported = "ocr_unsupported"
	DiagnosticOCRDerived     = "ocr_text_derived"
)

// OCRObservation is a typed host/adapter result for one exact original image.
// Transformation identifies the OCR configuration; text is always derived evidence.
type OCRObservation struct {
	Source         source.Locator `json:"source"`
	State          OCRState       `json:"state"`
	Transformation string         `json:"transformation"`
	Text           string         `json:"text"`
}

// Validate rejects ambiguous states and never fabricates original byte offsets.
func (o OCRObservation) Validate() error {
	if o.State == OCRUnobserved {
		var absent source.Locator
		if o.Source != absent || o.Transformation != "" || o.Text != "" {
			return invalidLayout()
		}
		return nil
	}
	if err := o.Source.Validate(); err != nil {
		return err
	}
	if o.Source.Kind != source.ImageLocation || o.Transformation == "" || !utf8.ValidString(o.Transformation) ||
		!utf8.ValidString(o.Text) {
		return invalidLayout()
	}
	switch o.State {
	case OCRRecognized:
		if o.Text == "" {
			return invalidLayout()
		}
	case OCRUnreadable, OCRUnsupported:
		if o.Text != "" {
			return invalidLayout()
		}
	case OCRUnobserved:
		return invalidLayout()
	default:
		return invalidLayout()
	}
	return nil
}

// Mapping returns support-only derived OCR evidence, or unobserved mapping if
// no text was recognized. OCR accuracy is not certified by this method.
func (o OCRObservation) Mapping() (source.MappedText, error) {
	if err := o.Validate(); err != nil {
		return source.MappedText{}, err
	}
	if o.State != OCRRecognized {
		return source.MappedText{}, nil
	}
	return source.DerivedText(o.Text, []source.Locator{o.Source})
}

// ApplyOCR applies explicit observations to an owned layout snapshot, preserving
// partial coverage and original page text/geometry. Any error returns no document.
func ApplyOCR(ctx context.Context, document Document, observations []OCRObservation) (Document, error) {
	if err := ctx.Err(); err != nil {
		return Document{}, err
	}
	if err := document.Validate(); err != nil {
		return Document{}, err
	}
	indexed, err := indexOCR(observations)
	if err != nil {
		return Document{}, err
	}
	out := cloneDocument(document)
	applied := make(map[source.Locator]struct{}, len(indexed))
	for pageIndex := range out.Pages {
		if err := ctx.Err(); err != nil {
			return Document{}, err
		}
		page := &out.Pages[pageIndex]
		for imageIndex := range page.Images {
			image := &page.Images[imageIndex]
			observation, exists := indexed[image.Location]
			if !exists {
				continue
			}
			image.OCR = observation
			page.Diagnostics = ocrDiagnostics(page.Diagnostics, image.Location, observation.State)
			page.Coverage = Partial
			out.Coverage = Partial
			applied[image.Location] = struct{}{}
		}
	}
	if len(applied) != len(indexed) {
		return Document{}, invalidLayout()
	}
	if err := out.Validate(); err != nil {
		return Document{}, err
	}
	if err := ctx.Err(); err != nil {
		return Document{}, err
	}
	return out, nil
}

func indexOCR(observations []OCRObservation) (map[source.Locator]OCRObservation, error) {
	out := make(map[source.Locator]OCRObservation, len(observations))
	for _, observation := range observations {
		if err := observation.Validate(); err != nil {
			return nil, err
		}
		if observation.State == OCRUnobserved {
			return nil, invalidLayout()
		}
		if _, exists := out[observation.Source]; exists {
			return nil, invalidLayout()
		}
		out[observation.Source] = observation
	}
	return out, nil
}

func ocrDiagnostics(existing []Diagnostic, location source.Locator, state OCRState) []Diagnostic {
	out := make([]Diagnostic, 0, len(existing)+1)
	for _, diagnostic := range existing {
		if diagnostic.Element == location.Reference.Artifact &&
			(diagnostic.Code == DiagnosticOCRUnprocessed || diagnostic.Code == DiagnosticOCRUnreadable || diagnostic.Code == DiagnosticOCRUnsupported || diagnostic.Code == DiagnosticOCRDerived) {
			continue
		}
		out = append(out, diagnostic)
	}
	code := DiagnosticOCRDerived
	switch state {
	case OCRUnreadable:
		code = DiagnosticOCRUnreadable
	case OCRUnsupported:
		code = DiagnosticOCRUnsupported
	case OCRUnobserved, OCRRecognized:
	}
	return append(out, Diagnostic{Code: code, Element: location.Reference.Artifact})
}

func cloneDocument(document Document) Document {
	document.Diagnostics = append([]Diagnostic(nil), document.Diagnostics...)
	document.Pages = append([]Page(nil), document.Pages...)
	for index := range document.Pages {
		page := &document.Pages[index]
		page.Words = append([]Word(nil), page.Words...)
		page.Cells = append([]Cell(nil), page.Cells...)
		page.Images = append([]Image(nil), page.Images...)
		page.Diagnostics = append([]Diagnostic(nil), page.Diagnostics...)
	}
	return document
}
