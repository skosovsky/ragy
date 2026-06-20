package chunking

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
)

type sourceMeta struct {
	Tenant string
}

type docMeta struct {
	SourceID  string
	SourceURI string
	Index     int
	Total     int
	Tenant    string
}

func TestProjectDocumentsBuildsStorageReadyDocuments(t *testing.T) {
	t.Parallel()

	chunks := []Chunk[struct{}]{
		{ID: "src_0", SourceID: "src", Index: 0, Total: 2, Content: "alpha"},
		{ID: "src_1", SourceID: "src", Index: 1, Total: 2, Content: "beta"},
	}
	docs, err := ProjectDocuments(chunks, ProjectionConfig[sourceMeta, struct{}, docMeta]{
		Source: SourceDescriptor[sourceMeta]{
			ID:  "src",
			URI: "file://source",
			Meta: sourceMeta{
				Tenant: "acme",
			},
		},
		MetadataProjector: MetadataProjectorFunc[sourceMeta, struct{}, docMeta](
			func(source SourceDescriptor[sourceMeta], _ Chunk[struct{}], id ChunkIdentity) (docMeta, error) {
				return docMeta{
					SourceID:  id.SourceID,
					SourceURI: id.SourceURI,
					Index:     id.Index,
					Total:     id.Total,
					Tenant:    source.Meta.Tenant,
				}, nil
			},
		),
	})
	if err != nil {
		t.Fatalf("ProjectDocuments(): %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("len(docs) = %d, want 2", len(docs))
	}
	if docs[0].ID != "src_0" || docs[0].Meta.SourceURI != "file://source" || docs[0].Meta.Total != 2 {
		t.Fatalf("doc[0] = %#v, want projected identity and meta", docs[0])
	}
	if docs[0].ScoreState != retrieval.ScoreAbsent || docs[0].Rank != 1 {
		t.Fatalf("ScoreState/Rank = %v/%d, want ScoreAbsent/1", docs[0].ScoreState, docs[0].Rank)
	}
}

func TestProjectDocumentsDerivesMissingChunkID(t *testing.T) {
	t.Parallel()

	chunks := []Chunk[struct{}]{
		{SourceID: "src", Index: 0, Total: 1, Content: "alpha"},
	}
	docs, err := ProjectDocuments(chunks, ProjectionConfig[sourceMeta, struct{}, docMeta]{
		Source: SourceDescriptor[sourceMeta]{
			ID: "src",
		},
		MetadataProjector: MetadataProjectorFunc[sourceMeta, struct{}, docMeta](
			func(_ SourceDescriptor[sourceMeta], _ Chunk[struct{}], id ChunkIdentity) (docMeta, error) {
				return docMeta{
					SourceID: id.SourceID,
					Index:    id.Index,
					Total:    id.Total,
				}, nil
			},
		),
	})
	if err != nil {
		t.Fatalf("ProjectDocuments(): %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("len(docs) = %d, want 1", len(docs))
	}
	if docs[0].ID != "src_0" || docs[0].Meta.SourceID != "src" {
		t.Fatalf("doc = %#v, want derived chunk identity", docs[0])
	}
}

func TestProjectDocumentsRequiresMetadataProjector(t *testing.T) {
	t.Parallel()

	_, err := ProjectDocuments(
		[]Chunk[struct{}]{{ID: "src_0", SourceID: "src", Index: 0, Total: 1, Content: "alpha"}},
		ProjectionConfig[sourceMeta, struct{}, docMeta]{
			Source: SourceDescriptor[sourceMeta]{ID: "src"},
		},
	)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ProjectDocuments() error = %v, want invalid argument", err)
	}
}
