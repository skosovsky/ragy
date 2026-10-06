// Package layout defines parser-independent retained text and geometry envelopes.
// Parsers consume already authorized input; authentication/storage/OCR engines
// and domain metadata remain host responsibilities.
package layout

import (
	"context"
	"fmt"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

// Schema identifies the normalized layout wire contract, not a software release.
const Schema = "ragy.layout"

// Coverage is observed parser coverage, propagated without implicit promotion.
type Coverage string

const (
	Complete    Coverage = "complete"
	Partial     Coverage = "partial"
	Unsupported Coverage = "unsupported"
)

// Diagnostic is a non-content parser diagnostic code, optionally on one element.
type Diagnostic struct {
	Code    string `json:"code"`
	Element string `json:"element"`
}

// Input contains already authorized bytes in the given retained representation.
// An ingestion/source host supplies and owns this association; Parse cannot
// authenticate a revision or make a raw URL/path lookup scoped.
type Input struct {
	Reference source.Reference
	Data      []byte
}

// Parser returns a validated owned layout snapshot or no payload on error.
type Parser interface {
	Parse(context.Context, Input) (Document, error)
}

// Word addresses a UTF-8 interval in the Page.Text representation, with geometry.
type Word struct {
	ID     string           `json:"id"`
	Span   source.ByteSpan  `json:"span"`
	Region source.Rectangle `json:"region"`
}

// Cell identifies one original logical table cell, independent of word ordering.
type Cell struct {
	Location source.Locator   `json:"location"`
	Region   source.Rectangle `json:"region"`
	Text     string           `json:"text"`
}

// Image contains an original image region, with no invented description/OCR text.
type Image struct {
	Location source.Locator `json:"location"`
	OCR      OCRObservation `json:"ocr"`
}

// Page separates physical geometry from retained normalized text representation.
type Page struct {
	Reference   source.Reference    `json:"reference"`
	Geometry    source.PageGeometry `json:"geometry"`
	Text        string              `json:"text"`
	Words       []Word              `json:"words"`
	Cells       []Cell              `json:"cells"`
	Images      []Image             `json:"images"`
	Coverage    Coverage            `json:"coverage"`
	Diagnostics []Diagnostic        `json:"diagnostics"`
}

// Document is the normalized layout envelope. PageCount includes unparsed pages;
// Pages never silently imply full document coverage if the parser stopped early.
type Document struct {
	Schema      string           `json:"schema"`
	Reference   source.Reference `json:"reference"`
	PageCount   int              `json:"page_count"`
	Coverage    Coverage         `json:"coverage"`
	Diagnostics []Diagnostic     `json:"diagnostics"`
	Pages       []Page           `json:"pages"`
}

// Validate checks representation identities, coverage, geometry and text spans.
func (d Document) Validate() error {
	if d.Schema != Schema || d.PageCount <= 0 || len(d.Pages) > d.PageCount || !validCoverage(d.Coverage) {
		return invalidLayout()
	}
	if err := d.Reference.Validate(); err != nil {
		return err
	}
	if err := validateDiagnostics(d.Diagnostics); err != nil {
		return err
	}
	complete := len(d.Pages) == d.PageCount
	previous := -1
	for _, page := range d.Pages {
		if page.Geometry.PhysicalIndex <= previous || page.Geometry.PhysicalIndex >= d.PageCount ||
			!sameSource(page.Reference, d.Reference) || page.Reference.Representation == d.Reference.Representation {
			return invalidLayout()
		}
		if err := page.Validate(); err != nil {
			return err
		}
		if page.Coverage != Complete {
			complete = false
		}
		previous = page.Geometry.PhysicalIndex
	}
	if d.Coverage == Complete && (!complete || incompleteDiagnostics(d.Diagnostics)) {
		return invalidLayout()
	}
	if d.Coverage != Complete && len(d.Diagnostics) == 0 && !pagesDiagnosed(d.Pages) {
		return invalidLayout()
	}
	return nil
}

// Validate checks one page without fetching source payload or resolving latest.
func (p Page) Validate() error {
	if err := p.Reference.Validate(); err != nil {
		return err
	}
	if err := p.Geometry.Validate(); err != nil {
		return err
	}
	if !validCoverage(p.Coverage) || !utf8.ValidString(p.Text) {
		return invalidLayout()
	}
	if err := validateDiagnostics(p.Diagnostics); err != nil {
		return err
	}
	if p.Coverage == Complete && incompleteDiagnostics(p.Diagnostics) {
		return invalidLayout()
	}
	if p.Coverage != Complete && len(p.Diagnostics) == 0 {
		return invalidLayout()
	}
	if err := p.validateWords(); err != nil {
		return err
	}
	if err := p.validateCells(); err != nil {
		return err
	}
	return p.validateImages()
}

func (p Page) validateWords() error {
	seen := make(map[string]struct{}, len(p.Words))
	previous := 0
	for _, word := range p.Words {
		if word.ID == "" || !utf8.ValidString(word.ID) || word.Span.Start < previous {
			return invalidLayout()
		}
		if _, exists := seen[word.ID]; exists {
			return invalidLayout()
		}
		if err := word.Span.ValidateText(p.Text); err != nil {
			return err
		}
		if err := word.Region.Validate(p.Geometry); err != nil {
			return err
		}
		seen[word.ID] = struct{}{}
		previous = word.Span.End
	}
	return nil
}

func (p Page) validateCells() error {
	seen := make(map[cellIdentity]struct{}, len(p.Cells))
	for _, cell := range p.Cells {
		if err := cell.Location.Validate(); err != nil {
			return err
		}
		if cell.Location.Kind != source.CellLocation || cell.Location.Page != p.Geometry ||
			!sameSource(cell.Location.Reference, p.Reference) ||
			!utf8.ValidString(cell.Text) {
			return invalidLayout()
		}
		if err := cell.Region.Validate(p.Geometry); err != nil {
			return err
		}
		key := cellIdentity{table: cell.Location.Cell.Table, element: cell.Location.Cell.Element}
		if _, exists := seen[key]; exists {
			return invalidLayout()
		}
		seen[key] = struct{}{}
	}
	return nil
}

func (p Page) validateImages() error {
	seen := make(map[source.Locator]struct{}, len(p.Images))
	for _, image := range p.Images {
		if err := image.Location.Validate(); err != nil {
			return err
		}
		if image.Location.Kind != source.ImageLocation || image.Location.Page != p.Geometry ||
			!sameSource(image.Location.Reference, p.Reference) {
			return invalidLayout()
		}
		if _, exists := seen[image.Location]; exists {
			return invalidLayout()
		}
		if err := image.OCR.Validate(); err != nil {
			return err
		}
		if image.OCR.State != OCRUnobserved && image.OCR.Source != image.Location {
			return invalidLayout()
		}
		if image.OCR.State != OCRUnobserved && p.Coverage == Complete {
			return invalidLayout()
		}
		seen[image.Location] = struct{}{}
	}
	return nil
}

func sameSource(first, second source.Reference) bool {
	return first.Namespace == second.Namespace && first.Source == second.Source && first.Revision == second.Revision &&
		first.Transformation == second.Transformation && first.AccessFingerprint == second.AccessFingerprint
}
func validCoverage(coverage Coverage) bool {
	return coverage == Complete || coverage == Partial || coverage == Unsupported
}
func validateDiagnostics(diagnostics []Diagnostic) error {
	for _, diagnostic := range diagnostics {
		if diagnostic.Code == "" || !utf8.ValidString(diagnostic.Code) || !utf8.ValidString(diagnostic.Element) {
			return invalidLayout()
		}
	}
	return nil
}
func pagesDiagnosed(pages []Page) bool {
	for _, page := range pages {
		if page.Coverage != Complete && len(page.Diagnostics) > 0 {
			return true
		}
	}
	return false
}
func invalidLayout() error {
	return fmt.Errorf("%w: normalized layout envelope", ragy.ErrInvalidArgument)
}

func incompleteDiagnostics(diagnostics []Diagnostic) bool {
	for _, diagnostic := range diagnostics {
		switch diagnostic.Code {
		case "page_limit",
			DiagnosticOCRUnprocessed,
			DiagnosticOCRUnreadable,
			"unsupported_geometry",
			"unsupported_grid",
			DiagnosticOCRUnsupported,
			DiagnosticOCRDerived:
			return true
		}
	}
	return false
}

type cellIdentity struct {
	table   string
	element string
}
