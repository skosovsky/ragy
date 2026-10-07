package layout_test

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/source"
)

func TestProjectionGatesImageCallbackAndRejectsFalseOriginalText(t *testing.T) {
	for _, kind := range []string{"before", "during", "wrong support", "original mapping"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange.
			fixture := newResolveFixture(t)
			document := ocrDocument()
			calls := 0
			if kind == "before" {
				fixture.host.revoked = true
			}
			options := layout.ProjectionOptions{
				Read:      fixture.read,
				ImageText: projectionCallback(document, fixture.host, kind, &calls),
			}
			// Act.
			projected, err := layout.Project(context.Background(), document, options)
			// Assert.
			if err == nil || len(projected) != 0 {
				t.Fatal("invalid/denied projection leaked partial source evidence")
			}
			if kind == "before" && calls != 0 {
				t.Fatal("external image consumer ran after revocation")
			}
		})
	}
}

func TestProjectionWorksWithoutImageModelAndPreservesCoverageDiagnostics(t *testing.T) {
	// Arrange.
	document := ocrDocument()
	// Act.
	projected, err := layout.Project(
		context.Background(),
		document,
		layout.ProjectionOptions{Read: access.Unrestricted()},
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if len(projected) != 1 || projected[0].Coverage != layout.Partial ||
		projected[0].Diagnostics[0].Code != layout.DiagnosticOCRUnprocessed {
		t.Fatal("model-free projection lost partial state or fabricated image text")
	}
	projected[0].Diagnostics[0].Code = "changed"
	if document.Pages[0].Diagnostics[0].Code != layout.DiagnosticOCRUnprocessed {
		t.Fatal("projection diagnostics alias input")
	}
}

func projectionCallback(
	document layout.Document,
	host *layoutHost,
	kind string,
	calls *int,
) func(context.Context, access.Binding, layout.Image) (source.MappedText, error) {
	return func(_ context.Context, _ access.Binding, image layout.Image) (source.MappedText, error) {
		*calls++
		if kind == "during" {
			host.revoked = true
		}
		if kind == "original mapping" {
			location := source.Locator{
				Reference: document.Pages[0].Reference,
				Kind:      source.TextLocation,
				Span:      source.ByteSpan{Start: 6, End: 10},
			}
			return source.OriginalText(location, document.Pages[0].Text)
		}
		support := image.Location
		if kind == "wrong support" {
			support.Reference.Revision = "r2"
		}
		return source.DerivedText("description", []source.Locator{support})
	}
}

func TestEmptyImageTextExplicitlyOverridesRecognizedOCR(t *testing.T) {
	// Arrange: real recognized OCR exists but host selects the explicit empty policy.
	document := ocrDocument()
	document, err := layout.ApplyOCR(t.Context(), document, []layout.OCRObservation{{
		Source: document.Pages[0].Images[0].Location, State: layout.OCRRecognized,
		Transformation: "ocr", Text: "recognized",
	}})
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := layout.Project(t.Context(), document, layout.ProjectionOptions{Read: access.Unrestricted()})
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	// Act.
	output, err := layout.Project(t.Context(), document, layout.ProjectionOptions{
		Read: access.Unrestricted(),
		ImageText: func(context.Context, access.Binding, layout.Image) (source.MappedText, error) {
			calls++
			return source.MappedText{}, nil
		},
	})
	// Assert: only image-derived text disappears; page evidence/partial coverage remain.
	if err != nil || calls != 1 || len(output) != len(baseline)-1 {
		t.Fatal(len(output), len(baseline), calls, err)
	}
	for _, projected := range output {
		if projected.Text.Text() == "recognized" || projected.Coverage != layout.Partial {
			t.Fatal(projected)
		}
	}
}
