//go:build darwin || linux

package persistent

import (
	"cmp"
	"context"
	"encoding/json"
	"path/filepath"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/durablefs"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// Intent supplies a compatible dense vector, separately from token-matrix queries.
type Intent struct{ Embedding dense.Embedding }

type QueryCapabilities struct {
	Space          dense.Space
	ScoreSemantics retrieval.ScoreSemantics
	MaxScanRecords int
	Exact          bool
}

func (a *Adapter[TMeta]) Schema() filter.Schema { return a.config.Schema }
func (*Adapter[TMeta]) ReadCapabilities() access.Capabilities {
	return access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true}
}
func (a *Adapter[TMeta]) QueryCapabilities() QueryCapabilities {
	return QueryCapabilities{
		Space:          a.config.Space,
		ScoreSemantics: a.scoreSemantics(),
		MaxScanRecords: a.config.MaxScanRecords,
		Exact:          true,
	}
}

func (a *Adapter[TMeta]) Retrieve(
	ctx context.Context,
	request retrieval.Query[Intent],
) (retrieval.ResultSet[TMeta], error) {
	empty := retrieval.NewResultSet[TMeta](nil, nil)
	result, err := a.retrieve(ctx, request)
	if err != nil {
		return retrieval.DeliverRead(ctx, request.Read, empty, access.NonSkippable(err), nil)
	}
	return retrieval.DeliverRead(ctx, request.Read, result, nil, nil)
}

func (a *Adapter[TMeta]) retrieve(
	ctx context.Context,
	request retrieval.Query[Intent],
) (retrieval.ResultSet[TMeta], error) {
	if a == nil {
		return nil, ragy.ErrInvalidArgument
	}
	request = retrieval.CopyRequestOptions(request)
	request.Intent.Embedding.Vector = slices.Clone(request.Intent.Embedding.Vector)
	if err := request.Read.Check(ctx); err != nil {
		return nil, err
	}

	prepared, err := retrieval.PrepareRead(ctx, request, a)
	if err != nil {
		return nil, err
	}
	if err = a.validateQuery(prepared); err != nil {
		return nil, err
	}
	lock, err := durablefs.Lock(ctx, filepath.Join(a.root, "target.lock"), false)
	if err != nil {
		return nil, err
	}
	defer func() { _ = lock.Close() }()
	entries, err := a.selectCatalogs(ctx, prepared.Read)
	if err != nil {
		return nil, err
	}
	admitted, err := a.admitRecords(ctx, prepared, entries)
	if err != nil {
		return nil, err
	}
	docs := make([]retrieval.Document[TMeta], 0, len(admitted))
	for _, record := range admitted {
		doc, loadErr := a.loadDocument(ctx, prepared, record.entry, record.record)
		if loadErr != nil {
			return nil, loadErr
		}
		docs = append(docs, doc)
	}
	slices.SortStableFunc(docs, func(first, second retrieval.Document[TMeta]) int {
		if order := cmp.Compare(second.Score, first.Score); order != 0 {
			return order
		}
		return cmp.Compare(first.ID, second.ID)
	})
	if len(docs) > prepared.Options.TopK {
		docs = docs[:prepared.Options.TopK]
	}
	for i := range docs {
		docs[i].Rank = i + 1
	}
	return retrieval.NewResultSet(docs, nil), nil
}

func (a *Adapter[TMeta]) validateQuery(request retrieval.Query[Intent]) error {
	if request.Options.TopK <= 0 || request.Options.TopK > a.config.MaxScanRecords {
		return ragy.ErrInvalidArgument
	}
	if request.Options.Threshold != nil || len(request.Options.Vector) != 0 || request.Options.Graph != nil {
		return ragy.ErrUnsupported
	}
	if err := request.Options.Validate(); err != nil {
		return err
	}
	if request.Intent.Embedding.Space != a.config.Space {
		return ragy.ErrInvalidArgument
	}
	return request.Intent.Embedding.Validate()
}

type admittedRecord struct {
	entry  catalog
	record descriptor
}

func (a *Adapter[TMeta]) admitRecords(
	ctx context.Context,
	request retrieval.Query[Intent],
	entries []catalog,
) ([]admittedRecord, error) {
	var admitted []admittedRecord
	for _, entry := range entries {
		for _, record := range entry.Records {
			if err := request.Read.Check(ctx); err != nil {
				return nil, err
			}
			allowed, err := filter.MatchCondition(
				request.Options.Filters,
				func(field string) (any, bool) { value, exists := record.Attributes[field]; return value, exists },
			)
			if err != nil {
				return nil, err
			}
			if !allowed {
				continue
			}
			if len(admitted) == a.config.MaxScanRecords {
				return nil, ragy.ErrInvalidArgument
			}
			admitted = append(admitted, admittedRecord{entry: entry, record: record})
		}
	}
	return admitted, nil
}

func (a *Adapter[TMeta]) loadDocument(
	ctx context.Context,
	request retrieval.Query[Intent],
	entry catalog,
	record descriptor,
) (retrieval.Document[TMeta], error) {
	if err := request.Read.Check(ctx); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	data, err := durablefs.QueryPayload(
		ctx,
		a.config.PayloadReader,
		lifecycle.PayloadRead{
			Reference: record.Reference,
			Path:      filepath.Join(a.path(entry.Manifest), record.Digest+".json"),
			MaxBytes:  a.config.MaxPayloadBytes,
		},
	)
	if err != nil {
		return retrieval.Document[TMeta]{}, durablefs.QueryPayloadError(err)
	}
	var stored payload
	if digest(data) != record.Digest || decodeStrict(data, &stored) != nil || stored.Schema != payloadSchema ||
		stored.Reference != record.Reference {
		return retrieval.Document[TMeta]{}, ragy.ErrProtocol
	}
	score, err := dense.Similarity(
		ctx,
		request.Intent.Embedding,
		dense.Embedding{Space: entry.Space, Vector: stored.Vector},
	)
	if err != nil {
		return retrieval.Document[TMeta]{}, ragy.ErrProtocol
	}
	if err = request.Read.Check(ctx); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	meta, err := a.config.Codec.Decode(record.Attributes)
	if err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	if err = request.Read.Check(ctx); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	meta, err = a.config.CloneMeta(meta)
	if err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	if err = request.Read.Check(ctx); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	var location source.Locator
	location.Reference, location.Kind = record.Reference, source.DocumentLocation
	if validateRecordMapping(record.Reference, stored.SourceMapping, stored.Content) != nil {
		return retrieval.Document[TMeta]{}, ragy.ErrProtocol
	}
	id, err := location.Identity()
	if err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	return retrieval.Document[TMeta]{
		ID:             id,
		Content:        stored.Content,
		SourceMapping:  stored.SourceMapping,
		SourceSupports: []source.Locator{location},
		Meta:           meta,
		Score:          score,
		ScoreState:     retrieval.ScorePresent,
		ScoreSemantics: a.scoreSemantics(),
	}, nil
}

func (a *Adapter[TMeta]) scoreSemantics() retrieval.ScoreSemantics {
	data, _ := json.Marshal(a.config.Space)
	return retrieval.ScoreSemantics(dense.ScoreSemantics(a.config.Space.Metric) + ":" + digest(data))
}

func (a *Adapter[TMeta]) selectCatalogs(ctx context.Context, read access.Binding) ([]catalog, error) {
	if err := read.Check(ctx); err != nil {
		return nil, err
	}
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return nil, err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return nil, ragy.ErrProtocol
	}
	var selected []catalog
	for _, pinned := range read.Publication().Targets() {
		if pinned.Target != a.config.Target {
			continue
		}
		if pinned.Namespace != a.config.Namespace {
			return nil, ragy.ErrUnavailable
		}
		manifest, err := selectedManifest(snapshot, pinned)
		if err != nil {
			return nil, err
		}
		if err = read.Check(ctx); err != nil {
			return nil, err
		}
		entry, err := a.readCatalog(ctx, manifest.ID)
		if err != nil {
			return nil, ragy.ErrUnavailable
		}
		if entry.Identity != manifest.Identity || entry.PayloadFingerprint != manifest.Payload ||
			!catalogInventory(entry, manifest, a.config.Target) {
			return nil, ragy.ErrProtocol
		}
		selected = append(selected, entry)
	}
	return selected, nil
}

func selectedManifest(snapshot lifecycle.Snapshot, pinned access.TargetRevision) (lifecycle.Manifest, error) {
	var selected lifecycle.Manifest
	for _, manifest := range snapshot.Manifests {
		id := manifest.Identity
		if id.Namespace != pinned.Namespace || id.Source != pinned.Source || id.Revision != pinned.Revision ||
			id.Transformation != pinned.Transformation ||
			id.Access != pinned.AccessFingerprint ||
			manifest.PublishedAt.IsZero() ||
			manifest.Tombstone || manifest.Retired {
			continue
		}
		for _, target := range manifest.Targets {
			if target.Name == pinned.Target && target.State == lifecycle.TargetReady {
				if selected.ID != "" {
					return lifecycle.Manifest{}, ragy.ErrUnavailable
				}
				selected = manifest
			}
		}
	}
	if selected.ID == "" {
		return lifecycle.Manifest{}, ragy.ErrUnavailable
	}
	return selected, nil
}

func catalogInventory(entry catalog, manifest lifecycle.Manifest, target string) bool {
	for _, inventory := range manifest.Targets {
		if inventory.Name == target {
			return lifecycle.SameArtifactInventory(entry.Artifacts, inventory.Artifacts)
		}
	}
	return false
}

func (a *Adapter[TMeta]) AdmitPublication(publication access.Publication) error {
	if a == nil {
		return access.Protect(ragy.ErrInvalidArgument)
	}
	return publication.AdmitTarget(a.config.Target)
}
