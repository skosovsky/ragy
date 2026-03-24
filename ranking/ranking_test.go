package ranking

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
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
	lists ...[]ragy.Document,
) {
	t.Helper()

	merger, err := NewReciprocalRankFusion(60)
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

func TestRRFMergeNormalizesRelevance(t *testing.T) {
	merger, err := NewReciprocalRankFusion(60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := merger.Merge(context.Background(),
		[]ragy.Document{{ID: "a", Content: "A", Relevance: 0.1}, {ID: "b", Content: "B", Relevance: 0.1}},
		[]ragy.Document{{ID: "b", Content: "B", Relevance: 0.1}, {ID: "a", Content: "A", Relevance: 0.1}},
	)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}

	if len(out) != 2 {
		t.Fatalf("len(out) = %d, want 2", len(out))
	}

	if out[0].Relevance <= 0 || out[0].Relevance > 1 {
		t.Fatalf("out[0].Relevance = %f, want in (0,1]", out[0].Relevance)
	}
}

func TestRRFMergeTreatsNilContextAsBackground(t *testing.T) {
	merger, err := NewReciprocalRankFusion(60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	lists := [][]ragy.Document{
		{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
		{{ID: "b", Content: "B"}, {ID: "a", Content: "A"}},
	}

	var (
		nilOut []ragy.Document
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
			nilOut[i].Relevance != bgOut[i].Relevance {
			t.Fatalf("Merge(nil) result[%d] = %#v, want %#v", i, nilOut[i], bgOut[i])
		}

		same, sameErr := samePayload(nilOut[i], bgOut[i])
		if sameErr != nil {
			t.Fatalf("samePayload() error = %v", sameErr)
		}
		if !same {
			t.Fatalf("Merge(nil) payload[%d] = %#v, want %#v", i, nilOut[i], bgOut[i])
		}
	}
}

func TestRRFRejectsDocumentsWithoutID(t *testing.T) {
	merger, err := NewReciprocalRankFusion(60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	if _, err := merger.Merge(context.Background(), []ragy.Document{{Content: "broken"}}); err == nil {
		t.Fatal("Merge() error = nil, want error")
	}
}

func TestRRFRejectsConflictingContentForSameID(t *testing.T) {
	merger, err := NewReciprocalRankFusion(60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	_, err = merger.Merge(
		context.Background(),
		[]ragy.Document{{ID: "a", Content: "A"}},
		[]ragy.Document{{ID: "a", Content: "B"}},
	)
	if err == nil {
		t.Fatal("Merge() error = nil, want error")
	}
}

func TestRRFRejectsConflictingAttributesForSameID(t *testing.T) {
	merger, err := NewReciprocalRankFusion(60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	_, err = merger.Merge(
		context.Background(),
		[]ragy.Document{{ID: "a", Content: "A", Attributes: ragy.Attributes{"tenant": "acme"}}},
		[]ragy.Document{{ID: "a", Content: "A", Attributes: ragy.Attributes{"tenant": "globex"}}},
	)
	if err == nil {
		t.Fatal("Merge() error = nil, want error")
	}
}

func TestRRFTreatsNilAndEmptyAttributesAsEquivalent(t *testing.T) {
	merger, err := NewReciprocalRankFusion(60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := merger.Merge(
		context.Background(),
		[]ragy.Document{{ID: "a", Content: "A", Attributes: nil}},
		[]ragy.Document{{ID: "a", Content: "A", Attributes: ragy.Attributes{}}},
	)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	if len(out) != 1 || out[0].ID != "a" {
		t.Fatalf("Merge() = %#v, want merged document a", out)
	}
}

func TestRRFReturnsCanonicalizedAttributes(t *testing.T) {
	merger, err := NewReciprocalRankFusion(60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := merger.Merge(
		context.Background(),
		[]ragy.Document{{ID: "a", Content: "A", Attributes: ragy.Attributes{"age": int(7)}}},
		[]ragy.Document{{ID: "a", Content: "A", Attributes: ragy.Attributes{"age": json.Number("7")}}},
	)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}

	value, ok := out[0].Attributes["age"].(int64)
	if !ok || value != 7 {
		t.Fatalf("merged age = %#v, want int64(7)", out[0].Attributes["age"])
	}
}

func TestRRFRejectsUnsupportedAttributeValueTypesWithoutPanic(t *testing.T) {
	merger, err := NewReciprocalRankFusion(60)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	_, err = merger.Merge(
		context.Background(),
		[]ragy.Document{{ID: "a", Content: "A", Attributes: ragy.Attributes{"tags": []string{"x"}}}},
		[]ragy.Document{{ID: "a", Content: "A", Attributes: ragy.Attributes{"tags": []string{"x"}}}},
	)
	if err == nil {
		t.Fatal("Merge() error = nil, want unsupported attribute error")
	}
}

func TestRRFMergeFailsFastOnPreCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	assertCanceledMerge(
		ctx,
		t,
		[]ragy.Document{{ID: "a", Content: "A"}},
		[]ragy.Document{{ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastOnMidMergeCancellation(t *testing.T) {
	assertCanceledMerge(
		newCancelOnErrCallContext(5),
		t,
		[]ragy.Document{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastDuringMaxMergedScore(t *testing.T) {
	assertCanceledMerge(
		newCancelOnErrCallContext(8),
		t,
		[]ragy.Document{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastDuringBuildMergedDocuments(t *testing.T) {
	assertCanceledMerge(
		newCancelOnErrCallContext(10),
		t,
		[]ragy.Document{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastAfterSort(t *testing.T) {
	assertCanceledMerge(
		newCancelOnErrCallContext(13),
		t,
		[]ragy.Document{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}
