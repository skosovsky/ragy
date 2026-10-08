package pdf_test

import (
	"context"
	"errors"
	"os"
	"runtime"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	pdfadapter "github.com/skosovsky/ragy/adapters/pdf"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/source"
)

func fixtureReference() source.Reference {
	return source.Reference{
		Namespace:         "manuals",
		Source:            "manual",
		Revision:          "r1",
		Transformation:    "pdf/layout",
		AccessFingerprint: "public-acl",
		Artifact:          "pdf",
		Representation:    "pdf-binary",
	}
}
func parserConfig(python string) pdfadapter.Config {
	return pdfadapter.Config{Python: python, Transformation: "pdf/layout", Limits: pdfadapter.Limits{
		InputBytes:  1 << 20,
		OutputBytes: 1 << 20,
		Pages:       10,
		Words:       100,
		Cells:       100,
		Images:      10,
		Timeout:     5 * time.Second,
	}}
}

func TestParserRejectsInvalidInputAndCancellationBeforeExternalExecution(t *testing.T) {
	// Arrange: non-existent executable proves rejected input does not dispatch.
	parser, err := pdfadapter.New(parserConfig("/nonexistent/ragy-python"))
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"empty", "wrong representation", "wrong transformation", "oversize", "cancelled"} {
		t.Run(kind, func(t *testing.T) {
			input := layout.Input{Reference: fixtureReference(), Data: []byte("pdf bytes")}
			ctx := context.Background()
			switch kind {
			case "empty":
				input.Data = nil
			case "wrong representation":
				input.Reference.Representation = "normalized-page-text"
			case "wrong transformation":
				input.Reference.Transformation = "other"
			case "oversize":
				input.Data = make([]byte, 1<<20+1)
			case "cancelled":
				cancelled, cancel := context.WithCancel(ctx)
				cancel()
				ctx = cancelled
			}
			// Act.
			document, err := parser.Parse(ctx, input)
			// Assert.
			if err == nil || errors.Is(err, ragy.ErrUnavailable) || len(document.Pages) != 0 {
				t.Fatal("invalid input dispatched or returned payload")
			}
		})
	}
}

func TestParserPreservesEarlierDeadlineAndDoesNotExportEngineErrors(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test process uses POSIX sh; actual Python adapter remains portable")
	}
	// Arrange: a process with no child worker waits until caller cancellation.
	executable := t.TempDir() + "/parser"
	script := []byte("#!/bin/sh\nprintf 'private-source-sentinel' >&2\nwhile :; do :; done\n")
	if err := os.WriteFile(executable, script, 0700); err != nil {
		t.Fatal(err)
	}
	parser, err := pdfadapter.New(parserConfig(executable))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	// Act.
	document, err := parser.Parse(
		ctx,
		layout.Input{Reference: fixtureReference(), Data: []byte("already-authorized bytes")},
	)
	// Assert.
	if !errors.Is(err, context.DeadlineExceeded) || len(document.Pages) != 0 || document.Reference.Source != "" {
		t.Fatal("deadline was replaced or payload exported")
	}
	if err.Error() != context.DeadlineExceeded.Error() {
		t.Fatal("external stderr leaked through error")
	}
}
