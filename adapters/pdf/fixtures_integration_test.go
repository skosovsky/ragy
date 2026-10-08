//go:build integration || e2e

package pdf_test

import (
	"os"
	"testing"
)

func integrationPython(t *testing.T) string {
	t.Helper()
	python := os.Getenv("RAGY_PDF_PYTHON")
	if python == "" {
		t.Fatal("PDF tests require RAGY_PDF_PYTHON for the retained parser")
	}
	return python
}

func loadFixture(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/manual.pdf")
	if err != nil {
		t.Fatal(err)
	}
	return data
}
