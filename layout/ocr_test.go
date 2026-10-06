package layout_test

import (
	"context"
	"errors"
	"testing"

	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/source"
)

func ocrDocument() layout.Document {
	document := validDocument()
	page := &document.Pages[0]
	reference := page.Reference
	reference.Artifact, reference.Representation = "p0/img0", "original-image"
	location := source.Locator{
		Reference: reference,
		Kind:      source.ImageLocation,
		Page:      page.Geometry,
		Region:    source.Rectangle{Left: 100, Top: 200, Right: 300, Bottom: 400},
	}
	page.Images = []layout.Image{{Location: location}}
	page.Coverage = layout.Partial
	page.Diagnostics = []layout.Diagnostic{{Code: "ocr_unprocessed", Element: reference.Artifact}}
	document.Coverage = layout.Partial
	return document
}

func TestOCRObservationsPreserveOriginalAndPartialCoverage(t *testing.T) {
	for _, state := range []layout.OCRState{layout.OCRRecognized, layout.OCRUnreadable, layout.OCRUnsupported} {
		t.Run(string(state), func(t *testing.T) {
			// Arrange.
			original := ocrDocument()
			observation := layout.OCRObservation{
				Source:         original.Pages[0].Images[0].Location,
				State:          state,
				Transformation: "host-ocr-fingerprint",
			}
			if state == layout.OCRRecognized {
				observation.Text = "Revenue"
			}
			// Act.
			updated, err := layout.ApplyOCR(context.Background(), original, []layout.OCRObservation{observation})
			// Assert.
			if err != nil {
				t.Fatal(err)
			}
			assertOCRSnapshots(t, original, updated)
			observed := updated.Pages[0].Images[0].OCR
			mapping, err := observed.Mapping()
			if err != nil {
				t.Fatal(err)
			}
			assertOCRMapping(t, state, mapping)
			if len(updated.Pages[0].Diagnostics) != 1 {
				t.Fatal("old OCR diagnostic not replaced")
			}
			if state == layout.OCRUnreadable && updated.Pages[0].Diagnostics[0].Code != "ocr_region_unreadable" {
				t.Fatal("unreadable diagnostic lost")
			}
		})
	}
}

func TestOCRRejectsInvalidStatesIdentitiesDuplicatesAndCancellation(t *testing.T) {
	for _, kind := range []string{"duplicate", "wrong revision", "unknown image", "missing transformation", "unreadable text", "recognized empty", "unknown state", "cancelled"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange.
			document := ocrDocument()
			observation := layout.OCRObservation{
				Source:         document.Pages[0].Images[0].Location,
				State:          layout.OCRUnreadable,
				Transformation: "ocr",
			}
			observations := []layout.OCRObservation{observation}
			ctx := context.Background()
			switch kind {
			case "duplicate":
				observations = append(observations, observation)
			case "wrong revision":
				observations[0].Source.Reference.Revision = "r2"
			case "unknown image":
				observations[0].Source.Reference.Artifact = "other"
			case "missing transformation":
				observations[0].Transformation = ""
			case "unreadable text":
				observations[0].Text = "fake"
			case "recognized empty":
				observations[0].State = layout.OCRRecognized
			case "unknown state":
				observations[0].State = "guessed"
			case "cancelled":
				cancelled, cancel := context.WithCancel(ctx)
				cancel()
				ctx = cancelled
			}
			// Act.
			updated, err := layout.ApplyOCR(ctx, document, observations)
			// Assert.
			if err == nil || len(updated.Pages) != 0 {
				t.Fatal("invalid observation returned a partial document")
			}
			if kind == "cancelled" && !errors.Is(err, context.Canceled) {
				t.Fatal("cancellation ignored")
			}
		})
	}
}

func assertOCRMapping(t *testing.T, state layout.OCRState, mapping source.MappedText) {
	t.Helper()
	if state != layout.OCRRecognized {
		if mapping.Text() != "" {
			t.Fatal("unreadable/unsupported OCR fabricated text")
		}
		return
	}
	if mapping.Text() != "Revenue" || mapping.Fragments()[0].Origin != source.DerivedContent ||
		mapping.Fragments()[0].Precision != source.UnavailablePrecision {
		t.Fatal("OCR text claimed original byte precision")
	}
}

func assertOCRSnapshots(t *testing.T, original, updated layout.Document) {
	t.Helper()
	if updated.Coverage != layout.Partial || updated.Pages[0].Coverage != layout.Partial ||
		updated.Pages[0].Text != original.Pages[0].Text {
		t.Fatal("OCR promoted coverage or replaced original page text")
	}
	if original.Pages[0].Images[0].OCR.State != layout.OCRUnobserved ||
		original.Pages[0].Diagnostics[0].Code != "ocr_unprocessed" {
		t.Fatal("OCR mutated original snapshot")
	}
}
