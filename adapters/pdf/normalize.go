package pdf

import (
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/source"
)

type engineOutput struct {
	PageCount   int                 `json:"page_count"`
	Diagnostics []layout.Diagnostic `json:"diagnostics"`
	Pages       []enginePage        `json:"pages"`
	Error       string              `json:"error"`
}
type enginePage struct {
	Geometry    source.PageGeometry `json:"geometry"`
	Text        string              `json:"text"`
	Words       []layout.Word       `json:"words"`
	Cells       []engineCell        `json:"cells"`
	Images      []engineImage       `json:"images"`
	Coverage    layout.Coverage     `json:"coverage"`
	Diagnostics []layout.Diagnostic `json:"diagnostics"`
}
type engineCell struct {
	Table      string           `json:"table"`
	Element    string           `json:"element"`
	Row        int              `json:"row"`
	Column     int              `json:"column"`
	RowSpan    int              `json:"row_span"`
	ColumnSpan int              `json:"column_span"`
	Text       string           `json:"text"`
	Region     source.Rectangle `json:"region"`
}
type engineImage struct {
	ID     string           `json:"id"`
	Region source.Rectangle `json:"region"`
}

func normalize(output engineOutput, reference source.Reference) (layout.Document, error) {
	switch output.Error {
	case "":
	case "unsupported_geometry":
		return layout.Document{}, ragy.ErrUnsupported
	case "invalid_pdf", "limit_exceeded":
		return layout.Document{}, ragy.ErrInvalidArgument
	default:
		return layout.Document{}, ragy.ErrProtocol
	}
	document := layout.Document{
		Schema:      layout.Schema,
		Reference:   reference,
		PageCount:   output.PageCount,
		Coverage:    layout.Complete,
		Diagnostics: append([]layout.Diagnostic(nil), output.Diagnostics...),
		Pages:       make([]layout.Page, 0, len(output.Pages)),
	}
	if len(output.Pages) != output.PageCount {
		document.Coverage = layout.Partial
	}
	for _, parsed := range output.Pages {
		page := normalizePage(parsed, reference)
		if page.Coverage != layout.Complete {
			document.Coverage = layout.Partial
		}
		document.Pages = append(document.Pages, page)
	}
	return document, nil
}

func normalizePage(parsed enginePage, reference source.Reference) layout.Page {
	pageID := fmt.Sprintf("p%d", parsed.Geometry.PhysicalIndex)
	pageRef := represented(reference, pageID+"/text", "normalized-page-text")
	page := layout.Page{
		Reference:   pageRef,
		Geometry:    parsed.Geometry,
		Text:        parsed.Text,
		Words:       append([]layout.Word(nil), parsed.Words...),
		Cells:       make([]layout.Cell, 0, len(parsed.Cells)),
		Images:      make([]layout.Image, 0, len(parsed.Images)),
		Coverage:    parsed.Coverage,
		Diagnostics: append([]layout.Diagnostic(nil), parsed.Diagnostics...),
	}
	for _, cell := range parsed.Cells {
		location := cellLocator(
			represented(reference, pageID+"/"+cell.Table+"/"+cell.Element, "original-cell-text"),
			parsed.Geometry,
			cell,
		)
		page.Cells = append(page.Cells, layout.Cell{Location: location, Region: cell.Region, Text: cell.Text})
	}
	for _, image := range parsed.Images {
		location := imageLocator(
			represented(reference, pageID+"/"+image.ID, "original-image"),
			parsed.Geometry,
			image.Region,
		)
		var unobserved layout.OCRObservation
		page.Images = append(page.Images, layout.Image{Location: location, OCR: unobserved})
		for index := range page.Diagnostics {
			if page.Diagnostics[index].Element == image.ID {
				page.Diagnostics[index].Element = location.Reference.Artifact
			}
		}
	}
	return page
}

func represented(reference source.Reference, artifact, representation string) source.Reference {
	reference.Artifact, reference.Representation = artifact, representation
	return reference
}
func cellLocator(reference source.Reference, page source.PageGeometry, cell engineCell) source.Locator {
	var span source.ByteSpan
	var region source.Rectangle
	return source.Locator{
		Reference: reference,
		Kind:      source.CellLocation,
		Span:      span,
		Page:      page,
		Region:    region,
		Cell: source.TableCell{
			Table:      cell.Table,
			Element:    cell.Element,
			Row:        cell.Row,
			Column:     cell.Column,
			RowSpan:    cell.RowSpan,
			ColumnSpan: cell.ColumnSpan,
		},
	}
}
func imageLocator(reference source.Reference, page source.PageGeometry, region source.Rectangle) source.Locator {
	var span source.ByteSpan
	var cell source.TableCell
	return source.Locator{
		Reference: reference,
		Kind:      source.ImageLocation,
		Span:      span,
		Page:      page,
		Region:    region,
		Cell:      cell,
	}
}
