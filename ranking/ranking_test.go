package ranking

import (
	"context"
	"errors"
	"testing"

	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

type cancelOnErrCallContext struct {
	context.Context

	cancelOnCall int
	calls        int
}

func newCancelOnErrCallContext(cancelOnCall int) *cancelOnErrCallContext {
	return &cancelOnErrCallContext{
		Context:      context.Background(),
		cancelOnCall: cancelOnCall,
	}
}

func nilContext() context.Context {
	return nil
}

func (c *cancelOnErrCallContext) Err() error {
	c.calls++
	if c.calls >= c.cancelOnCall {
		return context.Canceled
	}

	return nil
}

func assertCanceledMerge(
	ctx context.Context,
	t *testing.T,
	lists ...[]retrieval.Document[contracttest.Meta],
) {
	t.Helper()

	merger, err := NewReciprocalRankFusion[contracttest.Meta](60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := merger.Merge(ctx, lists...)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Merge() error = %v, want context.Canceled", err)
	}
	if out != nil {
		t.Fatalf("Merge() out = %#v, want nil", out)
	}
}

func TestRRFMergeNormalizesScore(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.Meta](60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := merger.Merge(
		context.Background(),
		[]retrieval.Document[contracttest.Meta]{
			{ID: "a", Content: "A", Score: 0.1},
			{ID: "b", Content: "B", Score: 0.1},
		},
		[]retrieval.Document[contracttest.Meta]{
			{ID: "b", Content: "B", Score: 0.1},
			{ID: "a", Content: "A", Score: 0.1},
		},
	)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}

	if len(out) != 2 {
		t.Fatalf("len(out) = %d, want 2", len(out))
	}

	if out[0].Score <= 0 || out[0].Score > 1 {
		t.Fatalf("out[0].Score = %f, want in (0,1]", out[0].Score)
	}
}

func TestRRFMergeTreatsNilContextAsBackground(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.Meta](60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	lists := [][]retrieval.Document[contracttest.Meta]{
		{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
		{{ID: "b", Content: "B"}, {ID: "a", Content: "A"}},
	}

	var (
		nilOut []retrieval.Document[contracttest.Meta]
		nilErr error
	)
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Merge(nil, ...) panicked: %v", r)
			}
		}()
		nilOut, nilErr = merger.Merge(nilContext(), lists...)
	}()
	if nilErr != nil {
		t.Fatalf("Merge(nil, ...) error = %v, want nil", nilErr)
	}

	bgOut, bgErr := merger.Merge(context.Background(), lists...)
	if bgErr != nil {
		t.Fatalf("Merge(background) error = %v", bgErr)
	}

	if len(nilOut) != len(bgOut) {
		t.Fatalf("len(Merge(nil)) = %d, want %d", len(nilOut), len(bgOut))
	}
	for i := range bgOut {
		if nilOut[i].ID != bgOut[i].ID ||
			nilOut[i].Content != bgOut[i].Content ||
			nilOut[i].Score != bgOut[i].Score {
			t.Fatalf("Merge(nil) result[%d] = %#v, want %#v", i, nilOut[i], bgOut[i])
		}

		if !samePayload(nilOut[i], bgOut[i]) {
			t.Fatalf("Merge(nil) payload[%d] = %#v, want %#v", i, nilOut[i], bgOut[i])
		}
	}
}

func TestRRFRejectsDocumentsWithoutID(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.Meta](60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	if _, err := merger.Merge(
		context.Background(),
		[]retrieval.Document[contracttest.Meta]{{Content: "broken"}},
	); err == nil {
		t.Fatal("Merge() error = nil, want error")
	}
}

func TestRRFRejectsConflictingContentForSameID(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.Meta](60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	_, err = merger.Merge(
		context.Background(),
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A"}},
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "B"}},
	)
	if err == nil {
		t.Fatal("Merge() error = nil, want error")
	}
}

func TestRRFRejectsConflictingMetaForSameID(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.Meta](60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	_, err = merger.Merge(
		context.Background(),
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A", Meta: contracttest.Meta{"tenant": "acme"}}},
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A", Meta: contracttest.Meta{"tenant": "globex"}}},
	)
	if err == nil {
		t.Fatal("Merge() error = nil, want error")
	}
}

func TestRRFTreatsNilAndEmptyMetaAsDistinct(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.Meta](60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	_, err = merger.Merge(
		context.Background(),
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A", Meta: nil}},
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A", Meta: contracttest.Meta{}}},
	)
	if err == nil {
		t.Fatal("Merge() error = nil, want conflicting payload error")
	}
}

func TestRRFMergesMatchingMeta(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.Meta](60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := merger.Merge(
		context.Background(),
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A", Meta: contracttest.Meta{"age": int64(7)}}},
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A", Meta: contracttest.Meta{"age": int64(7)}}},
	)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}

	value, ok := out[0].Meta["age"].(int64)
	if !ok || value != 7 {
		t.Fatalf("merged age = %#v, want int64(7)", out[0].Meta["age"])
	}
}

func TestRRFMergeFailsFastOnPreCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	assertCanceledMerge(
		ctx,
		t,
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A"}},
		[]retrieval.Document[contracttest.Meta]{{ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastOnMidMergeCancellation(t *testing.T) {
	assertCanceledMerge(
		newCancelOnErrCallContext(5),
		t,
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastDuringMaxMergedScore(t *testing.T) {
	assertCanceledMerge(
		newCancelOnErrCallContext(8),
		t,
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastDuringBuildMergedDocuments(t *testing.T) {
	assertCanceledMerge(
		newCancelOnErrCallContext(10),
		t,
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastAfterSort(t *testing.T) {
	assertCanceledMerge(
		newCancelOnErrCallContext(13),
		t,
		[]retrieval.Document[contracttest.Meta]{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}
