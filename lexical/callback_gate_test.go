package lexical_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

type boundaryCodec struct {
	retrieval.MetadataCodec[snapshotMeta]

	trigger func() error
	calls   int
	at      int
}

func (c *boundaryCodec) Encode(meta snapshotMeta) (filter.RawAttributes, error) {
	c.calls++
	attrs, err := c.MetadataCodec.Encode(meta)
	if c.calls == c.at {
		return attrs, c.trigger()
	}
	return attrs, err
}

func newBoundaryRead(t *testing.T, revoked *bool) (filter.Schema, access.Binding) {
	t.Helper()
	schema, base := newSnapshotRead(t)
	mandatory, err := base.Prepare(
		context.Background(),
		schema,
		filter.Condition{},
		access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.Scoped(
		access.ScopedConfig{
			Schema:      schema,
			Mandatory:   mandatory,
			Publication: base.Publication(),
			Snapshot:    base.Snapshot(),
			Now:         func() time.Time { return time.Unix(100, 0) },
			Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
				if *revoked {
					return ragy.ErrUnavailable
				}
				return nil
			}),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return schema, read
}

func TestSnapshotMetadataCallbacksStopOnFirstProtection(t *testing.T) {
	for _, phase := range []string{"capture", "index", "retrieve"} {
		for _, revoke := range []bool{false, true} {
			for _, fail := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/revoke=%t/fail=%t", phase, revoke, fail), func(t *testing.T) {
					runSnapshotBoundary(t, phase, revoke, fail)
				})
			}
		}
	}
}

func runSnapshotBoundary(t *testing.T, phase string, revoke, fail bool) {
	t.Helper()
	// Arrange: two authorized candidates, deterministic first codec cancellation/revocation.
	revoked := false
	schema, read := newBoundaryRead(t, &revoked)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	codec := &boundaryCodec{MetadataCodec: retrieval.NewJSONCodec[snapshotMeta](schema), at: 1}
	codec.trigger = func() error {
		if revoke {
			revoked = true
		} else {
			cancel()
		}
		if fail {
			return fmt.Errorf("%w: private ordinary codec error", ragy.ErrProtocol)
		}
		return nil
	}
	fields := []string{"content"}
	if phase == "index" {
		fields = append(fields, "tenant")
		codec.at = 3
	}
	if phase == "retrieve" {
		codec.at = 0
	}
	clones := 0
	snapshot, err := lexical.NewBM25Snapshot(
		ctx,
		schema,
		lexical.Config[snapshotMeta]{SearchFields: fields, Codec: codec},
		read,
		[]retrieval.Document[snapshotMeta]{
			{ID: "a", Content: "searchable", Meta: snapshotMeta{Tenant: "a"}},
			{ID: "b", Content: "searchable", Meta: snapshotMeta{Tenant: "a"}},
		},
		func(meta snapshotMeta) (snapshotMeta, error) { clones++; return meta, nil },
	)
	if phase != "retrieve" {
		// Assert: capture dispatches one codec; metadata-field build dispatches its first
		// codec after two capture encodes, then suppresses the entire new snapshot.
		expectedClones := 0
		if phase == "index" {
			expectedClones = 2
		}
		assertMetadataProtection(t, err, revoke, fail)
		if snapshot != nil || codec.calls != codec.at || clones != expectedClones {
			t.Fatal(snapshot, codec.calls, clones)
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	codec.calls, codec.at, clones = 0, 1, 0
	// Act.
	result, err := snapshot.Retrieve(
		ctx,
		retrieval.Query[struct{}]{Read: read, Text: "searchable", Options: retrieval.RetrieveOptions{TopK: 2}},
	)
	// Assert: no next candidate Encode/output clone/delivery.
	assertMetadataProtection(t, err, revoke, fail)
	if result.Len() != 0 || codec.calls != 1 || clones != 0 {
		t.Fatal(result.Documents(), codec.calls, clones)
	}
}

func assertMetadataProtection(t *testing.T, err error, revoked, fail bool) {
	t.Helper()
	if fail && !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal("callback cause lost", err)
	}
	cause := context.Canceled
	if revoked {
		cause = ragy.ErrUnavailable
	}
	if !access.IsProtectionFailure(err) || !errors.Is(err, cause) {
		t.Fatal("lost protected read cause", err)
	}
}

func TestSnapshotNormalRankingAfterScopedBuild(t *testing.T) {
	// Arrange: metadata-field indexing is gated only for this construction context.
	schema, read := newSnapshotRead(t)
	ctx, cancel := context.WithCancel(t.Context())
	snapshot, err := lexical.NewBM25Snapshot(
		ctx,
		schema,
		lexical.Config[snapshotMeta]{SearchFields: []string{"content", "tenant"}},
		read,
		[]retrieval.Document[snapshotMeta]{
			{ID: "b", Content: "searchable", Meta: snapshotMeta{Tenant: "a"}},
			{ID: "a", Content: "searchable", Meta: snapshotMeta{Tenant: "a"}},
		},
		func(meta snapshotMeta) (snapshotMeta, error) { return meta, nil },
	)
	cancel()
	if err != nil {
		t.Fatal(err)
	}
	// Act: a new read context still works; the construction codec gate was not retained.
	result, err := snapshot.Retrieve(
		t.Context(),
		retrieval.Query[struct{}]{Read: read, Text: "searchable", Options: retrieval.RetrieveOptions{TopK: 2}},
	)
	// Assert: deterministic equal-score ID ordering, owned snapshot remains usable.
	if err != nil || result.Len() != 2 || result.Documents()[0].ID != "a" || result.Documents()[1].ID != "b" ||
		result.Documents()[0].Score != result.Documents()[1].Score {
		t.Fatal(result.Documents(), err)
	}
}

func TestSnapshotCodecProtectionDropsEarlierCandidateBeforeCloning(t *testing.T) {
	// Arrange: the second codec explicitly denies without mutating the binding.
	schema, read := newSnapshotRead(t)
	codec := &boundaryCodec{MetadataCodec: retrieval.NewJSONCodec[snapshotMeta](schema)}
	clones := 0
	snapshot, err := lexical.NewBM25Snapshot(
		t.Context(),
		schema,
		lexical.Config[snapshotMeta]{SearchFields: []string{"content"}, Codec: codec},
		read,
		[]retrieval.Document[snapshotMeta]{
			{ID: "a", Content: "searchable", Meta: snapshotMeta{Tenant: "a"}},
			{ID: "b", Content: "searchable", Meta: snapshotMeta{Tenant: "a"}},
		},
		func(meta snapshotMeta) (snapshotMeta, error) { clones++; return meta, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	codec.calls, codec.at, clones = 0, 2, 0
	codec.trigger = func() error { return access.Protect(ragy.ErrUnavailable) }
	// Act.
	result, err := snapshot.Retrieve(
		t.Context(),
		retrieval.Query[struct{}]{Read: read, Text: "searchable", Options: retrieval.RetrieveOptions{TopK: 2}},
	)
	// Assert: already accepted candidate cannot proceed to an output clone.
	if result.Len() != 0 || !access.IsProtectionFailure(err) || !errors.Is(err, ragy.ErrUnavailable) ||
		codec.calls != 2 ||
		clones != 0 {
		t.Fatal(result.Documents(), err, codec.calls, clones)
	}
}
