package evidence_test

import (
	"errors"
	"strconv"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/evidence"
)

func TestEvidenceRankExactOrdinalBoundary(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("all representable int ranks on this platform fit exact float64")
	}
	for _, rank := range []int64{0, 1, (1 << 53) - 1, 1 << 53, (1 << 53) + 1} {
		// Arrange.
		read, input, policy := fixture(t)
		input.Stages[0].Hits[0].Document.Rank = int(rank)
		// Act.
		record, err := evidence.Capture(t.Context(), read, input, policy)
		// Assert: huge external numeric IDs are not evidence ordinals.
		if rank > (1<<53)-1 {
			if !errors.Is(err, ragy.ErrInvalidArgument) {
				t.Fatal(rank, err)
			}
			if _, snapshotErr := record.Snapshot(); snapshotErr == nil {
				t.Fatal("failed rank exposed payload")
			}
			continue
		}
		snapshot, snapshotErr := record.Snapshot()
		if err != nil || snapshotErr != nil {
			t.Fatal(rank, err, snapshotErr)
		}
		value := snapshot.Stages[0].Hits[0].Rank.Value
		if rank == 0 {
			if value != nil {
				t.Fatal("zero invented rank", value)
			}
		} else if value == nil || *value != float64(rank) {
			t.Fatal(rank, value)
		}
	}
}

func TestEvidenceDecodeRejectsOutsideExactRankDomain(t *testing.T) {
	// Arrange: canonical record with observed rank one.
	read, input, policy := fixture(t)
	record, err := evidence.Capture(t.Context(), read, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	data, err := record.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	old := `"rank":{"state":"observed","value":1}`
	changed := strings.Replace(string(data), old, `"rank":{"state":"observed","value":9007199254740992}`, 1)
	if changed == string(data) {
		t.Fatal("rank fixture missing")
	}
	// Act/Assert: raw producer cannot bypass Capture's ordinal admission.
	if _, err = evidence.Decode([]byte(changed)); !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal(err)
	}
}
