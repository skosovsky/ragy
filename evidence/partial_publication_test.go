package evidence_test

import (
	"testing"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
)

func TestPartialPublicationExportNeverPromotesOrMasksInvalidEmpty(t *testing.T) {
	// Arrange: producer claims complete with retained original source evidence.
	read, input, policy := fixture(t)
	pub, err := access.PinPartialPublication(
		read.Publication().Reference(),
		read.Publication().Targets(),
		[]string{"tensor"},
	)
	if err != nil {
		t.Fatal(err)
	}
	partialRead, err := access.UnrestrictedAt(pub)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	record, err := evidence.Capture(t.Context(), partialRead, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := record.Snapshot()
	// Assert: partial binding is authoritative even when producer supplies complete coverage.
	if err != nil || snapshot.Outcome != evidence.Partial || !snapshot.Coverage.IsPartial() {
		t.Fatal("partial pin promoted", err)
	}
	input.Outcome = evidence.CompleteEmpty
	if _, err = evidence.Capture(t.Context(), partialRead, input, policy); err == nil {
		t.Fatal("partial binding concealed malformed complete-empty input")
	}
	input.Stages = nil
	if _, err = evidence.Capture(t.Context(), partialRead, input, policy); err != nil {
		t.Fatal("valid empty partial observation refused", err)
	}
}
