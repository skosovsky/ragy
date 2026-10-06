package documents

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// HydrationConfig specializes source materialization for retrieval documents.
// Source admission is shared with other typed payloads; storage/retention stay host-owned.
type HydrationConfig[TAccess, TMeta any] struct {
	Target     string
	Schema     filter.Schema
	Catalog    source.Catalog[TAccess]
	Loader     source.Loader[retrieval.Document[TMeta]]
	Attributes func(TAccess) (filter.RawAttributes, error)
	CloneMeta  retrieval.MetadataCloner[TMeta]
}

// Hydrator applies document validation and metadata/history copying over source.Reader.
type Hydrator[TAccess, TMeta any] struct {
	reader *source.Reader[TAccess, retrieval.Document[TMeta]]
}

func NewHydrator[TAccess, TMeta any](config HydrationConfig[TAccess, TMeta]) (*Hydrator[TAccess, TMeta], error) {
	if config.CloneMeta == nil {
		return nil, ragy.ErrInvalidArgument
	}
	reader, err := source.NewReader(source.ReadConfig[TAccess, retrieval.Document[TMeta]]{
		Target:     config.Target,
		Schema:     config.Schema,
		Catalog:    config.Catalog,
		Loader:     config.Loader,
		Attributes: config.Attributes,
		ValidatePayload: func(reference source.Reference, document retrieval.Document[TMeta]) error {
			if document.ID != reference.Artifact {
				return ragy.ErrProtocol
			}
			return retrieval.ValidateDocument(document)
		},
		ClonePayload: func(document retrieval.Document[TMeta]) (retrieval.Document[TMeta], error) {
			meta, cloneErr := config.CloneMeta(document.Meta)
			if cloneErr != nil {
				return retrieval.Document[TMeta]{}, cloneErr
			}
			document.Meta = meta
			document.ScoreHistory = append([]retrieval.ScoreObservation(nil), document.ScoreHistory...)
			document.SourceSupports = append([]source.Locator(nil), document.SourceSupports...)
			return document, nil
		},
	})
	if err != nil {
		return nil, err
	}
	return &Hydrator[TAccess, TMeta]{reader: reader}, nil
}

// Lookup admits the entire batch before loading any document and returns owned
// snapshots. No payload is delivered on revision, permission or freshness failure.
func (h *Hydrator[TAccess, TMeta]) Lookup(
	ctx context.Context,
	request source.LookupRequest,
) ([]source.Materialized[retrieval.Document[TMeta]], error) {
	if h == nil {
		return nil, access.NonSkippable(ragy.ErrInvalidArgument)
	}
	return h.reader.Lookup(ctx, request)
}
