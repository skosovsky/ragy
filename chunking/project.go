package chunking

import (
	"fmt"
	"strings"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
)

// SourceDescriptor carries source-level identity and metadata into chunk projection.
type SourceDescriptor[TSourceMeta any] struct {
	ID        string
	URI       string
	StorageID string
	Category  string
	Meta      TSourceMeta
}

// Validate checks source descriptor invariants.
func (s SourceDescriptor[TSourceMeta]) Validate() error {
	if strings.TrimSpace(s.ID) == "" {
		return fmt.Errorf("%w: source descriptor id", ragy.ErrMissingSourceID)
	}
	return nil
}

// ChunkIdentity is the storage-ready identity for a chunk document.
type ChunkIdentity struct {
	DocumentID string
	SourceID   string
	SourceURI  string
	StorageID  string
	MergeKey   string
	Index      int
	Total      int
}

// ChunkIdentityPolicy derives stable chunk identity from a source descriptor and chunk.
type ChunkIdentityPolicy[TSourceMeta, TChunkMeta any] interface {
	Identity(source SourceDescriptor[TSourceMeta], chunk Chunk[TChunkMeta]) (ChunkIdentity, error)
}

// ChunkIdentityPolicyFunc adapts a function into ChunkIdentityPolicy.
type ChunkIdentityPolicyFunc[TSourceMeta, TChunkMeta any] func(
	SourceDescriptor[TSourceMeta],
	Chunk[TChunkMeta],
) (ChunkIdentity, error)

// Identity implements ChunkIdentityPolicy.
func (f ChunkIdentityPolicyFunc[TSourceMeta, TChunkMeta]) Identity(
	source SourceDescriptor[TSourceMeta],
	chunk Chunk[TChunkMeta],
) (ChunkIdentity, error) {
	return f(source, chunk)
}

// DefaultChunkIdentityPolicy uses source id, chunk index, and chunk total.
type DefaultChunkIdentityPolicy[TSourceMeta, TChunkMeta any] struct{}

// Identity implements ChunkIdentityPolicy.
func (DefaultChunkIdentityPolicy[TSourceMeta, TChunkMeta]) Identity(
	source SourceDescriptor[TSourceMeta],
	chunk Chunk[TChunkMeta],
) (ChunkIdentity, error) {
	if err := source.Validate(); err != nil {
		return ChunkIdentity{}, err
	}
	if strings.TrimSpace(chunk.Content) == "" {
		return ChunkIdentity{}, fmt.Errorf("%w: chunk content", ragy.ErrEmptyText)
	}
	if chunk.Index < 0 {
		return ChunkIdentity{}, fmt.Errorf("%w: chunk index", ragy.ErrInvalidArgument)
	}
	if chunk.Total < 0 {
		return ChunkIdentity{}, fmt.Errorf("%w: chunk total", ragy.ErrInvalidArgument)
	}
	if chunk.Total > 0 && chunk.Index >= chunk.Total {
		return ChunkIdentity{}, fmt.Errorf("%w: chunk index out of total", ragy.ErrInvalidArgument)
	}
	chunkSourceID := chunk.SourceID
	if strings.TrimSpace(chunkSourceID) == "" {
		chunkSourceID = source.ID
	}
	documentID := chunk.ID
	if strings.TrimSpace(documentID) == "" {
		documentID = fmt.Sprintf("%s_%d", chunkSourceID, chunk.Index)
	}
	mergeKey := documentID
	if source.URI != "" {
		mergeKey = fmt.Sprintf("%s#chunk=%d", source.URI, chunk.Index)
	}
	storageID := source.StorageID
	if storageID == "" {
		storageID = documentID
	}
	return ChunkIdentity{
		DocumentID: documentID,
		SourceID:   chunkSourceID,
		SourceURI:  source.URI,
		StorageID:  storageID,
		MergeKey:   mergeKey,
		Index:      chunk.Index,
		Total:      chunk.Total,
	}, nil
}

// MetadataProjector projects source and chunk metadata into retrieval document metadata.
type MetadataProjector[TSourceMeta, TChunkMeta, TDocMeta any] interface {
	Project(source SourceDescriptor[TSourceMeta], chunk Chunk[TChunkMeta], id ChunkIdentity) (TDocMeta, error)
}

// MetadataProjectorFunc adapts a function into MetadataProjector.
type MetadataProjectorFunc[TSourceMeta, TChunkMeta, TDocMeta any] func(
	SourceDescriptor[TSourceMeta],
	Chunk[TChunkMeta],
	ChunkIdentity,
) (TDocMeta, error)

// Project implements MetadataProjector.
func (f MetadataProjectorFunc[TSourceMeta, TChunkMeta, TDocMeta]) Project(
	source SourceDescriptor[TSourceMeta],
	chunk Chunk[TChunkMeta],
	id ChunkIdentity,
) (TDocMeta, error) {
	return f(source, chunk, id)
}

// ProjectionConfig configures storage-ready document projection.
type ProjectionConfig[TSourceMeta, TChunkMeta, TDocMeta any] struct {
	Source            SourceDescriptor[TSourceMeta]
	IdentityPolicy    ChunkIdentityPolicy[TSourceMeta, TChunkMeta]
	MetadataProjector MetadataProjector[TSourceMeta, TChunkMeta, TDocMeta]
}

// ProjectDocuments converts chunks into storage-ready retrieval documents.
func ProjectDocuments[TSourceMeta, TChunkMeta, TDocMeta any](
	chunks []Chunk[TChunkMeta],
	cfg ProjectionConfig[TSourceMeta, TChunkMeta, TDocMeta],
) ([]retrieval.Document[TDocMeta], error) {
	if err := cfg.Source.Validate(); err != nil {
		return nil, err
	}
	policy := cfg.IdentityPolicy
	if policy == nil {
		policy = DefaultChunkIdentityPolicy[TSourceMeta, TChunkMeta]{}
	}
	if cfg.MetadataProjector == nil {
		return nil, fmt.Errorf("%w: metadata projector", ragy.ErrInvalidArgument)
	}

	docs := make([]retrieval.Document[TDocMeta], 0, len(chunks))
	for _, chunk := range chunks {
		id, err := policy.Identity(cfg.Source, chunk)
		if err != nil {
			return docs, err
		}
		meta, err := cfg.MetadataProjector.Project(cfg.Source, chunk, id)
		if err != nil {
			return docs, err
		}
		doc := retrieval.Document[TDocMeta]{
			ID:         id.DocumentID,
			Content:    chunk.Content,
			ScoreState: retrieval.ScoreAbsent,
			Rank:       id.Index + 1,
			Meta:       meta,
		}
		if err := retrieval.ValidateDocument(doc); err != nil {
			return docs, err
		}
		docs = append(docs, doc)
	}
	return docs, nil
}
