package ranking

import (
	"context"
	"errors"
	"reflect"
	"testing"

	ragy "github.com/skosovsky/ragy"
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

func resultSet(docs ...retrieval.Document[contracttest.StructMeta]) retrieval.ResultSet[contracttest.StructMeta] {
	return retrieval.NewResultSet(docs, retrieval.DocumentIDResolver[contracttest.StructMeta]{})
}

func mergeDocLists(
	ctx context.Context,
	merger Merger[contracttest.StructMeta],
	lists ...[]retrieval.Document[contracttest.StructMeta],
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	sets := make([]retrieval.ResultSet[contracttest.StructMeta], len(lists))
	for i, list := range lists {
		sets[i] = resultSet(list...)
	}
	return merger.Merge(ctx, sets...)
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

func assertCanceledMergePreserves(
	ctx context.Context,
	t *testing.T,
	wantMinLen int,
	lists ...[]retrieval.Document[contracttest.StructMeta],
) {
	t.Helper()

	merger, err := NewReciprocalRankFusion[contracttest.StructMeta](60, nil)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := mergeDocLists(ctx, merger, lists...)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Merge() error = %v, want context.Canceled", err)
	}
	if out == nil {
		t.Fatal("Merge() out = nil, want non-nil ResultSet")
	}
	if out.Len() < wantMinLen {
		t.Fatalf("Merge() Len() = %d, want at least %d preserved docs", out.Len(), wantMinLen)
	}
}

func TestRRFMergeNormalizesScore(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.StructMeta](60, nil)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := mergeDocLists(context.Background(), merger,
		[]retrieval.Document[contracttest.StructMeta]{
			{ID: "a", Content: "A", Score: 0.1},
			{ID: "b", Content: "B", Score: 0.1},
		},
		[]retrieval.Document[contracttest.StructMeta]{
			{ID: "b", Content: "B", Score: 0.1},
			{ID: "a", Content: "A", Score: 0.1},
		},
	)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}

	if out.Len() != 2 {
		t.Fatalf("out.Len() = %d, want 2", out.Len())
	}

	if out.Documents()[0].Score <= 0 || out.Documents()[0].Score > 1 {
		t.Fatalf("out.Documents()[0].Score = %f, want in (0,1]", out.Documents()[0].Score)
	}
}

func TestRRFMergeTreatsNilContextAsBackground(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.StructMeta](60, nil)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	lists := [][]retrieval.Document[contracttest.StructMeta]{
		{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
		{{ID: "b", Content: "B"}, {ID: "a", Content: "A"}},
	}

	var (
		nilOut retrieval.ResultSet[contracttest.StructMeta]
		nilErr error
	)
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Merge(nil, ...) panicked: %v", r)
			}
		}()
		nilOut, nilErr = mergeDocLists(nilContext(), merger, lists...)
	}()
	if nilErr != nil {
		t.Fatalf("Merge(nil, ...) error = %v, want nil", nilErr)
	}

	bgOut, bgErr := mergeDocLists(context.Background(), merger, lists...)
	if bgErr != nil {
		t.Fatalf("Merge(background) error = %v", bgErr)
	}

	if nilOut.Len() != bgOut.Len() {
		t.Fatalf("Len(Merge(nil)) = %d, want %d", nilOut.Len(), bgOut.Len())
	}
	bgDocs := bgOut.Documents()
	nilDocs := nilOut.Documents()
	for i := range bgDocs {
		if nilDocs[i].ID != bgDocs[i].ID ||
			nilDocs[i].Content != bgDocs[i].Content ||
			nilDocs[i].Score != bgDocs[i].Score {
			t.Fatalf("Merge(nil) result[%d] = %#v, want %#v", i, nilDocs[i], bgDocs[i])
		}

		if !rrfSamePayload(nilDocs[i], bgDocs[i]) {
			t.Fatalf("Merge(nil) payload[%d] = %#v, want %#v", i, nilDocs[i], bgDocs[i])
		}
	}
}

func TestRRFRejectsDocumentsWithoutID(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.StructMeta](60, nil)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	if _, err := mergeDocLists(context.Background(), merger,
		[]retrieval.Document[contracttest.StructMeta]{{Content: "broken"}},
	); err == nil {
		t.Fatal("Merge() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Merge() error = %v, want protocol", err)
	}
}

func TestRRFRejectsConflictingContentForSameID(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.StructMeta](60, nil)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	_, err = mergeDocLists(context.Background(), merger,
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A"}},
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "B"}},
	)
	if err == nil {
		t.Fatal("Merge() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Merge() error = %v, want invalid argument", err)
	}
}

func TestRRFRejectsConflictingMetaForSameID(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.StructMeta](60, nil)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	_, err = mergeDocLists(
		context.Background(),
		merger,
		[]retrieval.Document[contracttest.StructMeta]{
			{ID: "a", Content: "A", Meta: contracttest.StructMeta{Tenant: "acme"}},
		},
		[]retrieval.Document[contracttest.StructMeta]{
			{ID: "a", Content: "A", Meta: contracttest.StructMeta{Tenant: "globex"}},
		},
	)
	if err == nil {
		t.Fatal("Merge() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Merge() error = %v, want invalid argument", err)
	}
}

func TestRRFTreatsNilAndEmptyMetaAsDistinct(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.StructMeta](60, nil)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	_, err = mergeDocLists(context.Background(), merger,
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A", Meta: contracttest.StructMeta{}}},
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A", Meta: contracttest.StructMeta{}}},
	)
	if err != nil {
		t.Fatalf("Merge() error = %v, want nil for matching empty meta", err)
	}
}

func TestRRFMergesMatchingMeta(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.StructMeta](60, nil)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := mergeDocLists(context.Background(), merger,
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A", Meta: contracttest.StructMeta{Age: 7}}},
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A", Meta: contracttest.StructMeta{Age: 7}}},
	)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}

	if out.Documents()[0].Meta.Age != 7 {
		t.Fatalf("merged age = %d, want 7", out.Documents()[0].Meta.Age)
	}
}

func TestRRFMergeUsesMergeKey(t *testing.T) {
	merger, err := NewReciprocalRankFusion[contracttest.StructMeta](60, tenantResolver{})
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	out, err := mergeDocLists(context.Background(), merger,
		[]retrieval.Document[contracttest.StructMeta]{
			{ID: "a", Content: "A", Score: 0.1, Meta: contracttest.StructMeta{Tenant: "acme"}},
		},
		[]retrieval.Document[contracttest.StructMeta]{
			{ID: "b", Content: "A", Score: 0.1, Meta: contracttest.StructMeta{Tenant: "acme"}},
		},
	)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	if out.Len() != 1 {
		t.Fatalf("out.Len() = %d, want 1 merged tenant", out.Len())
	}
}

type tenantResolver struct{}

func (tenantResolver) Resolve(doc retrieval.Document[contracttest.StructMeta]) retrieval.Identity {
	return retrieval.Identity{MergeKey: doc.Meta.Tenant, DocumentID: doc.ID}
}

func TestRRFMergeFailsFastOnPreCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	assertCanceledMergePreserves(
		ctx,
		t,
		1,
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A"}},
		[]retrieval.Document[contracttest.StructMeta]{{ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastOnMidMergeCancellation(t *testing.T) {
	assertCanceledMergePreserves(
		newCancelOnErrCallContext(5),
		t,
		1,
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastDuringMaxMergedScore(t *testing.T) {
	assertCanceledMergePreserves(
		newCancelOnErrCallContext(4),
		t,
		1,
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastDuringBuildMergedDocuments(t *testing.T) {
	assertCanceledMergePreserves(
		newCancelOnErrCallContext(6),
		t,
		2,
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func TestRRFMergeFailsFastAfterSort(t *testing.T) {
	assertCanceledMergePreserves(
		newCancelOnErrCallContext(7),
		t,
		2,
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}},
	)
}

func rrfSamePayload(left, right retrieval.Document[contracttest.StructMeta]) bool {
	return left.Content == right.Content && reflect.DeepEqual(left.Meta, right.Meta)
}
