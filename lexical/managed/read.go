package managed

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/internal/readfailure"
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
	err = readfailure.Check(ctx, request.Read, err)
	if err != nil {
		return empty, access.NonSkippable(err)
	}
	return result, nil
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
	records, generation, inventory, err := a.selectedRecords(ctx, prepared.Read.Publication())
	if err != nil {
		return nil, err
	}
	binding, err := prepared.Read.Fingerprint()
	if err != nil {
		return nil, err
	}
	predicate, err := prepared.Options.Filters.Fingerprint()
	if err != nil {
		return nil, err
	}
	key := snapshotKey{binding: binding, predicate: predicate, generation: generation, inventory: inventory}
	if err = prepared.Read.Check(ctx); err != nil {
		return nil, err
	}
	index := a.cached(key)
	if index == nil {
		index, err = a.buildSnapshot(ctx, prepared, records)
		if err != nil {
			return nil, err
		}
		index = a.cacheSnapshot(key, index)
	}
	if err = prepared.Read.Check(ctx); err != nil {
		return nil, err
	}
	result, err := index.Retrieve(ctx, prepared)
	if err != nil {
		return nil, err
	}
	// Cleanup and ledger changes must be checked even when an index was cached.
	if _, _, _, err = a.selectedRecords(ctx, prepared.Read.Publication()); err != nil {
		return nil, err
	}
	if err = prepared.Read.Check(ctx); err != nil {
		return nil, err
	}
	return result, nil
}

func (a *Adapter[TMeta]) buildSnapshot(
	ctx context.Context,
	prepared retrieval.Query[struct{}],
	records []Record[TMeta],
) (*lexical.BM25Snapshot[TMeta], error) {
	var err error
	var docs []retrieval.Document[TMeta]
	for _, record := range records {
		if err = prepared.Read.Check(ctx); err != nil {
			return nil, err
		}
		allowed, matchErr := retrieval.MatchDocument(a.config.BM25.Codec, record.Document, prepared.Options.Filters)
		if gateErr := prepared.Read.Check(ctx); gateErr != nil {
			return nil, readfailure.Join(gateErr, matchErr)
		}
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
		if err = prepared.Read.Check(ctx); err != nil {
			return nil, readfailure.Join(err, cloneErr)
		}
		if cloneErr != nil {
			return nil, cloneErr
		}
		docs = append(docs, cloned.Document)
	}
	index, err := lexical.NewBM25Snapshot(ctx, a.config.Schema, a.config.BM25, prepared.Read, docs, a.config.CloneMeta)
	if err != nil {
		return nil, err
	}
	return index, nil
}

func (a *Adapter[TMeta]) selectedRecords(
	ctx context.Context,
	publication access.Publication,
) ([]Record[TMeta], uint64, string, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return nil, 0, "", err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return nil, 0, "", ragy.ErrProtocol
	}
	var records []Record[TMeta]
	var manifests []lifecycle.Manifest
	for _, target := range publication.Targets() {
		if target.Target != a.config.Target {
			continue
		}
		if target.Namespace != a.config.Namespace {
			return nil, 0, "", ragy.ErrUnavailable
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
			return nil, 0, "", ragy.ErrUnavailable
		}
		records = append(records, version.records...)
		manifests = append(manifests, version.manifest)
	}
	data, err := json.Marshal(manifests)
	if err != nil {
		return nil, 0, "", ragy.ErrProtocol
	}
	digest := sha256.Sum256(data)
	return records, a.generation, hex.EncodeToString(digest[:]), nil
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
		if manifest.Retired || manifest.ID != version.manifest.ID || manifest.Payload != version.manifest.Payload ||
			!lifecycle.SameTargetInventory(
				manifest,
				version.manifest,
				target.Target,
			) || manifest.Tombstone || manifest.PublishedAt.IsZero() ||
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
