package managed

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// Retrieve requires a captured pinned inventory; staging is never selected by a
// live fallback. Mandatory metadata intersection precedes payload clone/projection.
func (a *Adapter[TMeta]) Retrieve(
	ctx context.Context,
	request retrieval.Query[struct{}],
) (retrieval.ResultSet[TMeta], error) {
	if a == nil {
		return retrieval.DeliverRead(
			ctx,
			request.Read,
			retrieval.NewResultSet[TMeta](nil, nil),
			access.NonSkippable(ragy.ErrInvalidArgument),
			nil,
		)
	}
	empty := retrieval.NewResultSet[TMeta](nil, a.config.BM25.Resolver)
	result, err := a.retrieve(ctx, request)
	if err != nil {
		return retrieval.DeliverRead(ctx, request.Read, empty, access.NonSkippable(err), a.config.BM25.Resolver)
	}
	return retrieval.DeliverRead(ctx, request.Read, result, nil, a.config.BM25.Resolver)
}

func (a *Adapter[TMeta]) retrieve(
	ctx context.Context,
	request retrieval.Query[struct{}],
) (retrieval.ResultSet[TMeta], error) {
	prepared, err := retrieval.PrepareRead(ctx, request, a)
	if err != nil {
		return nil, err
	}
	if prepared.Read.Publication().IsCurrent() {
		return nil, ragy.ErrUnsupported
	}
	if err = prepared.Options.Validate(); err != nil {
		return nil, err
	}
	records, err := a.selectedRecords(ctx, prepared.Read.Publication())
	if err != nil {
		return nil, err
	}
	var docs []retrieval.Document[TMeta]
	for _, record := range records {
		if err = prepared.Read.Check(ctx); err != nil {
			return nil, err
		}
		allowed, matchErr := retrieval.MatchDocument(a.config.BM25.Codec, record.Document, prepared.Options.Filters)
		if matchErr != nil {
			return nil, matchErr
		}
		if !allowed {
			continue
		}
		if err = prepared.Read.Check(ctx); err != nil {
			return nil, err
		}
		cloned, cloneErr := a.cloneRecord(record)
		if cloneErr != nil {
			return nil, cloneErr
		}
		if err = prepared.Read.Check(ctx); err != nil {
			return nil, err
		}
		docs = append(docs, cloned.Document)
	}
	index, err := lexical.NewBM25Snapshot(ctx, a.config.Schema, a.config.BM25, prepared.Read, docs, a.config.CloneMeta)
	if err != nil {
		return nil, err
	}
	return index.Retrieve(ctx, prepared)
}
func (a *Adapter[TMeta]) selectedRecords(ctx context.Context, publication access.Publication) ([]Record[TMeta], error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return nil, err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return nil, ragy.ErrProtocol
	}
	var records []Record[TMeta]
	for _, target := range publication.Targets() {
		if target.Target != a.config.Target {
			continue
		}
		if target.Namespace != a.config.Namespace {
			return nil, ragy.ErrUnavailable
		}
		key := revisionKey{
			namespace:      target.Namespace,
			source:         target.Source,
			revision:       target.Revision,
			transformation: target.Transformation,
			access:         target.AccessFingerprint,
		}
		version, exists := a.versions[key]
		if !exists || !confirmedVersion(snapshot, target, version) {
			return nil, ragy.ErrUnavailable
		}
		records = append(records, version.records...)
	}
	return records, nil
}

// A caller-supplied pinned tuple is insufficient: the exact inventory must have
// reached publication in the durable ledger. Historical published snapshots are
// retained until cleanup, even after the active publication changes.
func confirmedVersion[TMeta any](
	snapshot lifecycle.Snapshot,
	target access.TargetRevision,
	version staged[TMeta],
) bool {
	for _, manifest := range snapshot.Manifests {
		if manifest.ID != version.manifest.ID || manifest.Tombstone || manifest.PublishedAt.IsZero() ||
			keyForIdentity(manifest.Identity) != (revisionKey{
				namespace: target.Namespace, source: target.Source, revision: target.Revision,
				transformation: target.Transformation, access: target.AccessFingerprint,
			}) {
			continue
		}
		for _, inventory := range manifest.Targets {
			if inventory.Name != target.Target || inventory.State != lifecycle.TargetReady ||
				inventory.Revision != target.Revision || len(inventory.Artifacts) != len(version.records) {
				continue
			}
			return matchesRecords(inventory.Artifacts, version.records)
		}
	}
	return false
}

func matchesRecords[TMeta any](artifacts []lifecycle.Artifact, records []Record[TMeta]) bool {
	remaining := make(map[source.Reference]struct{}, len(artifacts))
	for _, artifact := range artifacts {
		remaining[artifact.Reference] = struct{}{}
	}
	for _, record := range records {
		if _, exists := remaining[record.Reference]; !exists {
			return false
		}
		delete(remaining, record.Reference)
	}
	return len(remaining) == 0
}

// AdmitPublication enforces partial branch exclusion before target reads.
func (a *Adapter[TMeta]) AdmitPublication(publication access.Publication) error {
	if a == nil {
		return access.Protect(ragy.ErrInvalidArgument)
	}
	return publication.AdmitTarget(a.config.Target)
}
