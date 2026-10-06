// Package managed supplies a lifecycle-aware, in-memory BM25 target.
package managed

import (
	"context"
	"slices"
	"sync"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type Record[TMeta any] struct {
	Reference source.Reference
	Document  retrieval.Document[TMeta]
}

type Config[TMeta any] struct {
	Namespace string
	Target    string
	Store     lifecycle.Store
	Schema    filter.Schema
	BM25      lexical.Config[TMeta]
	CloneMeta func(TMeta) (TMeta, error)
	// MaxCachedSnapshots is a required positive bound on resident scoped BM25 indexes.
	MaxCachedSnapshots int
}

type revisionKey struct{ namespace, source, revision, transformation, access string }
type staged[TMeta any] struct {
	manifest lifecycle.Manifest
	records  []Record[TMeta]
}

// Adapter retains exact snapshots until explicit cleanup. It does not promise
// persistence across process restart; missing retained snapshots fail explicitly.
type Adapter[TMeta any] struct {
	mu         sync.RWMutex
	config     Config[TMeta]
	versions   map[revisionKey]staged[TMeta]
	generation uint64
	cacheClock uint64
	cache      map[snapshotKey]cachedSnapshot[TMeta]
}

func New[TMeta any](config Config[TMeta]) (*Adapter[TMeta], error) {
	if config.Namespace == "" || config.Target == "" || config.Store == nil || config.CloneMeta == nil ||
		config.MaxCachedSnapshots <= 0 {
		return nil, ragy.ErrInvalidArgument
	}
	config.BM25.SearchFields = slices.Clone(config.BM25.SearchFields)
	if config.BM25.Codec == nil {
		config.BM25.Codec = retrieval.NewJSONCodec[TMeta](config.Schema)
	}
	if _, err := lexical.NewBM25Index(config.Schema, config.BM25, nil, nil); err != nil {
		return nil, err
	}
	return &Adapter[TMeta]{
		mu:         sync.RWMutex{},
		generation: 0,
		cacheClock: 0,
		config:     config,
		versions:   make(map[revisionKey]staged[TMeta]),
		cache:      make(map[snapshotKey]cachedSnapshot[TMeta]),
	}, nil
}
func (a *Adapter[TMeta]) Schema() filter.Schema { return a.config.Schema }
func (*Adapter[TMeta]) ReadCapabilities() access.Capabilities {
	return access.Capabilities{RequirePinnedPublication: true, ScopeProfile: true, PinnedPublication: true}
}
func (*Adapter[TMeta]) LexicalBackend() {}

// Stage owns and validates the full exact inventory, then installs it atomically.
func (a *Adapter[TMeta]) Stage(
	ctx context.Context,
	request lifecycle.StageRequest,
	records []Record[TMeta],
) (lifecycle.StageResult, error) {
	request.Manifest = request.Manifest.Clone()
	captured, err := a.captureStage(ctx, request, records)
	if err != nil {
		return lifecycle.StageResult{}, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err = a.checkStagePublication(ctx, request); err != nil {
		return lifecycle.StageResult{}, err
	}
	identity := request.Manifest.Identity
	key := keyForIdentity(identity)
	if previous, exists := a.versions[key]; exists && previous.manifest.ID != request.Manifest.ID {
		return lifecycle.StageResult{}, lifecycle.ErrConflict
	}
	a.versions[key] = staged[TMeta]{manifest: request.Manifest, records: captured}
	a.invalidateCacheLocked()
	if err = ctx.Err(); err != nil {
		return lifecycle.StageResult{}, err
	}
	return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: identity.Revision}, nil
}
func (a *Adapter[TMeta]) Inspect(ctx context.Context, request lifecycle.StageRequest) (lifecycle.StageResult, error) {
	if err := ctx.Err(); err != nil {
		return lifecycle.StageResult{}, err
	}
	if a == nil || request.Target != a.config.Target || request.Manifest.Identity.Namespace != a.config.Namespace {
		return lifecycle.StageResult{}, ragy.ErrInvalidArgument
	}
	if request.Manifest.Tombstone {
		return lifecycle.StageResult{}, ragy.ErrInvalidArgument
	}
	if err := request.Manifest.Validate(); err != nil {
		return lifecycle.StageResult{}, err
	}
	if stageInventory(request) == nil {
		return lifecycle.StageResult{}, ragy.ErrProtocol
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	previous, exists := a.versions[keyForIdentity(request.Manifest.Identity)]
	if !exists {
		return lifecycle.StageResult{State: lifecycle.TargetPending, Revision: ""}, nil
	}
	if previous.manifest.ID != request.Manifest.ID {
		return lifecycle.StageResult{}, lifecycle.ErrConflict
	}
	if previous.manifest.Identity != request.Manifest.Identity ||
		previous.manifest.Payload != request.Manifest.Payload ||
		!lifecycle.SameTargetInventory(previous.manifest, request.Manifest, request.Target) {
		return lifecycle.StageResult{}, ragy.ErrProtocol
	}
	if err := ctx.Err(); err != nil {
		return lifecycle.StageResult{}, err
	}
	return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: request.Manifest.Identity.Revision}, nil
}

func (a *Adapter[TMeta]) captureStage(
	ctx context.Context,
	request lifecycle.StageRequest,
	records []Record[TMeta],
) ([]Record[TMeta], error) {
	if a == nil || request.Target != a.config.Target || request.Manifest.Identity.Namespace != a.config.Namespace ||
		request.Manifest.Tombstone {
		return nil, ragy.ErrInvalidArgument
	}
	if err := request.Manifest.Validate(); err != nil {
		return nil, err
	}
	planned := stageInventory(request)
	if planned == nil || len(planned) != len(records) {
		return nil, ragy.ErrProtocol
	}
	captured := make([]Record[TMeta], 0, len(records))
	for _, record := range records {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, exists := planned[record.Reference]; !exists {
			return nil, ragy.ErrProtocol
		}
		delete(planned, record.Reference)
		if record.Document.ID != record.Reference.Artifact || record.Document.ScoreState != retrieval.ScoreAbsent {
			return nil, ragy.ErrProtocol
		}
		owned, err := a.cloneRecord(record)
		if err != nil {
			return nil, err
		}
		captured = append(captured, owned)
	}
	index, err := lexical.NewBM25Index(a.config.Schema, a.config.BM25, nil, nil)
	if err != nil {
		return nil, err
	}
	docs := make([]retrieval.Document[TMeta], 0, len(captured))
	for _, record := range captured {
		docs = append(docs, record.Document)
	}
	if err = index.Index(docs); err != nil {
		return nil, err
	}
	return captured, nil
}
func (a *Adapter[TMeta]) cloneRecord(record Record[TMeta]) (Record[TMeta], error) {
	if err := record.Reference.Validate(); err != nil {
		return Record[TMeta]{}, err
	}
	if err := retrieval.ValidateDocument(record.Document); err != nil {
		return Record[TMeta]{}, err
	}
	for _, location := range record.Document.SourceLocations() {
		ref := location.Reference
		if ref.Namespace != record.Reference.Namespace || ref.Source != record.Reference.Source ||
			ref.Revision != record.Reference.Revision || ref.AccessFingerprint != record.Reference.AccessFingerprint {
			return Record[TMeta]{}, ragy.ErrInvalidArgument
		}
	}
	meta, err := a.config.CloneMeta(record.Document.Meta)
	if err != nil {
		return Record[TMeta]{}, err
	}
	var location source.Locator
	location.Reference, location.Kind = record.Reference, source.DocumentLocation
	id, err := location.Identity()
	if err != nil {
		return Record[TMeta]{}, err
	}
	record.Document.Meta = meta
	record.Document.ScoreHistory = slices.Clone(record.Document.ScoreHistory)
	record.Document.SourceSupports = append(record.Document.SourceLocations(), location)
	record.Document.ID = id
	return record, nil
}
func (a *Adapter[TMeta]) checkStagePublication(ctx context.Context, request lifecycle.StageRequest) error {
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return ragy.ErrProtocol
	}
	if !registeredStage(snapshot, request) {
		return ragy.ErrProtocol
	}
	for _, publication := range snapshot.Publications {
		if publication.Source == request.Manifest.Identity.Source {
			if publication.Manifest != request.Manifest.ExpectedPublication {
				return lifecycle.ErrConflict
			}
			return ctx.Err()
		}
	}
	if request.Manifest.ExpectedPublication != "" {
		return lifecycle.ErrConflict
	}
	return ctx.Err()
}
func keyForIdentity(identity lifecycle.Identity) revisionKey {
	return revisionKey{
		namespace:      identity.Namespace,
		source:         identity.Source,
		revision:       identity.Revision,
		transformation: identity.Transformation,
		access:         identity.Access,
	}
}

func stageInventory(request lifecycle.StageRequest) map[source.Reference]struct{} {
	for _, target := range request.Manifest.Targets {
		if target.Name != request.Target {
			continue
		}
		planned := make(map[source.Reference]struct{}, len(target.Artifacts))
		for _, artifact := range target.Artifacts {
			planned[artifact.Reference] = struct{}{}
		}
		return planned
	}
	return nil
}

func registeredStage(snapshot lifecycle.Snapshot, request lifecycle.StageRequest) bool {
	for _, manifest := range snapshot.Manifests {
		if manifest.ID != request.Manifest.ID {
			continue
		}
		if manifest.Identity != request.Manifest.Identity || manifest.Key != request.Manifest.Key ||
			manifest.Payload != request.Manifest.Payload || manifest.ExpectedPublication != request.Manifest.ExpectedPublication ||
			manifest.Tombstone != request.Manifest.Tombstone || manifest.Partial != request.Manifest.Partial {
			return false
		}
		return registeredInventory(manifest, request)
	}
	return false
}

func registeredInventory(manifest lifecycle.Manifest, request lifecycle.StageRequest) bool {
	for _, target := range manifest.Targets {
		if target.Name == request.Target && target.State == lifecycle.TargetUnknown {
			return lifecycle.SameTargetInventory(manifest, request.Manifest, request.Target)
		}
	}
	return false
}
