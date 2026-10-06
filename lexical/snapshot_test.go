package lexical_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

type snapshotMeta struct {
	Tenant string `json:"tenant"`
}

func TestReadonlySnapshotGatesProjectionAndBindingReplacement(t *testing.T) {
	// Arrange: caller supplies a mixed corpus; denied payload must not enter cloning.
	schema, read := newSnapshotRead(t)
	copied := 0
	snapshot, err := lexical.NewBM25Snapshot(context.Background(), schema,
		lexical.Config[snapshotMeta]{SearchFields: []string{"content"}}, read,
		[]retrieval.Document[snapshotMeta]{
			{ID: "allowed", Content: "searchable", Meta: snapshotMeta{Tenant: "a"}},
			{ID: "denied", Content: "searchable", Meta: snapshotMeta{Tenant: "b"}},
		}, func(meta snapshotMeta) (snapshotMeta, error) {
			if meta.Tenant != "a" {
				t.Fatal("denied payload cloned")
			}
			copied++
			return meta, nil
		},
	)
	if err != nil || copied != 1 {
		t.Fatal("snapshot admission failed", err)
	}
	// Act/Assert.
	query := retrieval.Query[struct{}]{Read: read, Text: "searchable", Options: retrieval.RetrieveOptions{TopK: 10}}
	result, err := snapshot.Retrieve(context.Background(), query)
	if err != nil || result.Len() != 1 || result.Documents()[0].ID != "allowed" {
		t.Fatal("fixed corpus scoring failed", err)
	}
	replaced, err := access.UnrestrictedAt(read.Publication())
	if err != nil {
		t.Fatal(err)
	}
	query.Read = replaced
	result, err = snapshot.Retrieve(context.Background(), query)
	if !errors.Is(err, ragy.ErrUnavailable) || result.Len() != 0 {
		t.Fatal("snapshot binding replaced", err)
	}
}

func newSnapshotRead(t *testing.T) (filter.Schema, access.Binding) {
	t.Helper()
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	publication, err := access.PinPublication(
		"fixed",
		[]access.TargetRevision{
			{
				Target:            "lexical",
				Namespace:         "n",
				Source:            "policy",
				Revision:          "r1",
				Transformation:    "text",
				AccessFingerprint: "acl",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	read, err := access.Scoped(access.ScopedConfig{
		Schema:      schema,
		Mandatory:   mandatory,
		Publication: publication,
		Snapshot:    access.Snapshot{Identity: "scope", PolicyEpoch: 1, IssuedAt: now, ExpiresAt: now.Add(time.Minute)},
		Now:         func() time.Time { return now },
		Authority:   access.AuthorityFunc(func(context.Context, access.Snapshot) error { return nil }),
	})
	if err != nil {
		t.Fatal(err)
	}
	return schema, read
}

func TestSnapshotResultMetadataCannotMutatePinnedCorpus(t *testing.T) {
	// Arrange: metadata contains a pointer and must be cloned by the host.
	schema, read := newSnapshotRead(t)
	snapshot, err := lexical.NewBM25Snapshot(context.Background(), schema,
		lexical.Config[*snapshotMeta]{SearchFields: []string{"content"}}, read,
		[]retrieval.Document[*snapshotMeta]{{ID: "allowed", Content: "searchable", Meta: &snapshotMeta{Tenant: "a"}}},
		func(meta *snapshotMeta) (*snapshotMeta, error) { copyMeta := *meta; return &copyMeta, nil })
	if err != nil {
		t.Fatal(err)
	}
	query := retrieval.Query[struct{}]{Read: read, Text: "searchable", Options: retrieval.RetrieveOptions{TopK: 10}}
	first, err := snapshot.Retrieve(context.Background(), query)
	if err != nil || first.Len() != 1 {
		t.Fatal("first read failed", err)
	}
	// Act: mutate the delivered metadata, then query the same immutable snapshot.
	first.Documents()[0].Meta.Tenant = "b"
	second, err := snapshot.Retrieve(context.Background(), query)
	// Assert: pinned metadata and filtering remain intact.
	if err != nil || second.Len() != 1 || second.Documents()[0].Meta.Tenant != "a" {
		t.Fatal("result metadata changed pinned corpus", err)
	}
}

func TestSnapshotChecksFreshnessAfterFailingClone(t *testing.T) {
	for _, phase := range []string{"capture", "delivery"} {
		t.Run(phase, func(t *testing.T) {
			// Arrange: a host callback cancels the read and also returns an ordinary error.
			schema, read := newSnapshotRead(t)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			calls := 0
			snapshot, err := lexical.NewBM25Snapshot(
				ctx,
				schema,
				lexical.Config[snapshotMeta]{SearchFields: []string{"content"}},
				read,
				[]retrieval.Document[snapshotMeta]{
					{ID: "allowed", Content: "searchable", Meta: snapshotMeta{Tenant: "a"}},
				},
				func(meta snapshotMeta) (snapshotMeta, error) {
					calls++
					if phase == "capture" || calls == 2 {
						cancel()
						return meta, errors.New("private clone failure")
					}
					return meta, nil
				},
			)
			// Act/Assert: freshness failure wins even when the same callback failed.
			if phase == "capture" {
				if snapshot != nil {
					t.Fatal("rejected capture returned snapshot")
				}
				assertCloneProtection(t, err)
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			result, err := snapshot.Retrieve(
				ctx,
				retrieval.Query[struct{}]{Read: read, Text: "searchable", Options: retrieval.RetrieveOptions{TopK: 1}},
			)
			if result.Len() != 0 {
				t.Fatal("rejected delivery returned documents")
			}
			assertCloneProtection(t, err)
		})
	}
}

func assertCloneProtection(t *testing.T, err error) {
	t.Helper()
	if !errors.Is(err, context.Canceled) || !access.IsProtectionFailure(err) {
		t.Fatal("clone callback bypassed freshness gate", err)
	}
}
