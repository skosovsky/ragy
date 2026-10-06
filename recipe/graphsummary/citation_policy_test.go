package graphsummary_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/source"
)

func TestSelectedCitationsAreNotCompleteDerivationDependencies(t *testing.T) {
	// Arrange: model sees s1 and s2, selects only the complete membership in s1.
	f := newFixture(t)
	extra := f.request.Communities[1].Snippets[0]
	extra.Members = slices.Clone(f.request.Communities[0].Members)
	f.request.Communities[0].Snippets = append(f.request.Communities[0].Snippets, extra)
	denied := ""
	checked := map[string]int{}
	admit := func(_ context.Context, _ access.Binding, location source.Locator) error {
		id := location.Reference.Source
		checked[id]++
		if id == denied {
			return ragy.ErrUnavailable
		}
		return nil
	}
	f.config.AdmitSource = admit
	result, _, err := run(t.Context(), t, f, false)
	if err != nil || len(result.Communities) != 1 || checked["s1"] == 0 || checked["s2"] == 0 {
		t.Fatal(result, checked, err)
	}
	summary := result.Communities[0]
	if len(summary.Supports()) != 1 || summary.Supports()[0].Reference.Source != "s1" {
		t.Fatal(summary.Supports())
	}
	// Act: later unselected revocation does not alter the same still-valid binding.
	denied = "s2"
	checked = map[string]int{}
	delivered, err := summary.Resolve(t.Context(), f.request.Read, admit)
	// Assert: fresh Resolve checks selected evidence only; delivered bytes are owned.
	if err != nil || delivered.Text() == "" || checked["s2"] != 0 {
		t.Fatal(checked, err)
	}
	ownedText := delivered.Text()
	denied = "s1"
	suppressed, err := summary.Resolve(t.Context(), f.request.Read, admit)
	if !errors.Is(err, ragy.ErrUnavailable) || suppressed.Text() != "" || delivered.Text() != ownedText {
		t.Fatal("selected revocation must suppress new delivery without retracting prior data", err)
	}
}
