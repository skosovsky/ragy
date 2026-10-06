package lexical

import (
	"context"
	"errors"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/readfailure"
	"github.com/skosovsky/ragy/retrieval"
)

// BM25Snapshot is readonly scoring over one owned corpus and captured read binding.
// It exposes no Index/Upsert operation and cannot silently replace its pinned corpus.
type BM25Snapshot[TMeta any] struct {
	index     *BM25Index[TMeta]
	cloneMeta func(TMeta) (TMeta, error)
}

// NewBM25Snapshot owns metadata using the host clone contract. Binding freshness
// gates every callback; all subsequent queries require the captured fingerprint.
func NewBM25Snapshot[TMeta any](
	ctx context.Context, schema filter.Schema, config Config[TMeta], read access.Binding,
	documents []retrieval.Document[TMeta], cloneMeta func(TMeta) (TMeta, error),
) (*BM25Snapshot[TMeta], error) {
	if err := read.Check(ctx); err != nil {
		return nil, err
	}
	if read.Publication().IsCurrent() || cloneMeta == nil {
		return nil, ragy.ErrInvalidArgument
	}
	if len(read.Publication().Targets()) == 0 && len(documents) != 0 {
		return nil, ragy.ErrProtocol
	}
	fingerprint, err := read.Fingerprint()
	if err != nil {
		return nil, err
	}
	mandatory, err := read.Prepare(
		ctx,
		schema,
		filter.Condition{},
		access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true},
	)
	if err != nil {
		return nil, err
	}
	if config.Codec == nil {
		config.Codec = retrieval.NewJSONCodec[TMeta](schema)
	}
	captured, err := captureSnapshotDocuments(ctx, config.Codec, read, documents, mandatory, cloneMeta)
	if err != nil {
		return nil, err
	}
	index, err := NewBM25Index(schema, config, nil, nil)
	if err != nil {
		return nil, err
	}
	// The builder can encode metadata search fields. Its temporary codec carries
	// this capture's gates, then is removed before publishing the readonly index.
	index.codec = snapshotCodec[TMeta]{ctx: ctx, read: read, codec: config.Codec}
	err = index.Index(captured)
	index.codec = config.Codec
	if gateErr := read.Check(ctx); gateErr != nil {
		return nil, readfailure.Join(gateErr, err)
	}
	if err != nil {
		return nil, err
	}
	if err = read.Check(ctx); err != nil {
		return nil, err
	}
	index.snapshotReadFingerprint = fingerprint
	return &BM25Snapshot[TMeta]{index: index, cloneMeta: cloneMeta}, nil
}

func (s *BM25Snapshot[TMeta]) Retrieve(
	ctx context.Context,
	request retrieval.Query[struct{}],
) (retrieval.ResultSet[TMeta], error) {
	if s == nil || s.index == nil {
		return retrieval.NewResultSet[TMeta](nil, nil), access.Protect(ragy.ErrInvalidArgument)
	}
	result, err := s.index.retrieve(ctx, request)
	if access.IsProtectionFailure(err) {
		return retrieval.NewResultSet[TMeta](nil, s.index.resolver), access.Protect(err)
	}
	documents := result.Documents()
	for i := range documents {
		if gateErr := request.Read.Check(ctx); gateErr != nil {
			return retrieval.NewResultSet[TMeta](nil, s.index.resolver), readfailure.Join(gateErr, err)
		}
		meta, cloneErr := s.cloneMeta(documents[i].Meta)
		if gateErr := request.Read.Check(ctx); gateErr != nil {
			return retrieval.NewResultSet[TMeta](
					nil,
					s.index.resolver,
				), readfailure.Join(
					gateErr,
					errors.Join(err, cloneErr),
				)
		}
		if cloneErr != nil {
			return retrieval.NewResultSet[TMeta](nil, s.index.resolver), cloneErr
		}
		documents[i].Meta = meta
	}
	owned := retrieval.NewResultSet(documents, s.index.resolver)
	err = readfailure.Check(ctx, request.Read, err)
	if access.IsProtectionFailure(err) {
		return retrieval.NewResultSet[TMeta](nil, s.index.resolver), err
	}
	return owned, err
}
func (s *BM25Snapshot[TMeta]) Schema() filter.Schema {
	if s == nil || s.index == nil {
		return filter.Schema{}
	}
	return s.index.Schema()
}
func (s *BM25Snapshot[TMeta]) ReadCapabilities() access.Capabilities {
	if s == nil || s.index == nil {
		return access.Capabilities{ScopeProfile: false, PinnedPublication: false, RequirePinnedPublication: true}
	}
	return s.index.ReadCapabilities()
}
func (*BM25Snapshot[TMeta]) LexicalBackend() {}

func captureSnapshotDocuments[TMeta any](
	ctx context.Context, codec retrieval.MetadataCodec[TMeta], read access.Binding,
	documents []retrieval.Document[TMeta], mandatory filter.Condition, cloneMeta func(TMeta) (TMeta, error),
) ([]retrieval.Document[TMeta], error) {
	captured := make([]retrieval.Document[TMeta], 0, len(documents))
	for _, document := range documents {
		if err := read.Check(ctx); err != nil {
			return nil, err
		}
		allowed, matchErr := retrieval.MatchDocument(codec, document, mandatory)
		if gateErr := read.Check(ctx); gateErr != nil {
			return nil, readfailure.Join(gateErr, matchErr)
		}
		if matchErr != nil {
			return nil, matchErr
		}
		if !allowed {
			continue
		}
		if err := read.Check(ctx); err != nil {
			return nil, err
		}
		meta, cloneErr := cloneMeta(document.Meta)
		if err := read.Check(ctx); err != nil {
			return nil, readfailure.Join(err, cloneErr)
		}
		if cloneErr != nil {
			return nil, cloneErr
		}
		document.Meta = meta
		document.ScoreHistory = slices.Clone(document.ScoreHistory)
		document.SourceSupports = slices.Clone(document.SourceSupports)
		captured = append(captured, document)
	}
	return captured, nil
}

func (s *BM25Snapshot[TMeta]) AdmitPublication(publication access.Publication) error {
	if s == nil || s.index == nil {
		return access.Protect(ragy.ErrInvalidArgument)
	}
	return s.index.AdmitPublication(publication)
}
