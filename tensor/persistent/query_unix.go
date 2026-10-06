//go:build darwin || linux

package persistent

import (
	"context"
	"encoding/json"
	"path/filepath"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/durablefs"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

func (a *Adapter[TMeta]) Schema() filter.Schema { return a.config.Schema }
func (*Adapter[TMeta]) ReadCapabilities() access.Capabilities {
	return access.Capabilities{RequirePinnedPublication: true, ScopeProfile: true, PinnedPublication: true}
}

func (a *Adapter[TMeta]) QueryCapabilities() tensor.QueryCapabilities {
	return tensor.QueryCapabilities{
		Space:                 a.config.Space,
		ScoreSemantics:        string(a.scoreSemantics()),
		CandidateLimit:        a.config.MaxRecords,
		ExactWithinCandidates: true,
		Exhaustive:            false,
	}
}

// Retrieve integrates bounded native MaxSim into the ordinary retrieval contract.
func (a *Adapter[TMeta]) Retrieve(
	ctx context.Context,
	request retrieval.Query[tensorquery.Intent],
) (retrieval.ResultSet[TMeta], error) {
	result, err := a.Query(ctx, request)
	return result.Documents, err
}

func (a *Adapter[TMeta]) Query(
	ctx context.Context,
	request retrieval.Query[tensorquery.Intent],
) (tensorquery.Result[TMeta], error) {
	empty := retrieval.NewResultSet[TMeta](nil, nil)
	result, err := a.query(ctx, request)
	if err != nil {
		return tensorquery.Result[TMeta]{
			Documents: empty,
			Evidence:  tensor.RerankResult{Ranking: nil, CandidateIDs: nil, CandidateBudget: 0},
		}, access.NonSkippable(
			err,
		)
	}
	if err = request.Read.Check(ctx); err != nil {
		return tensorquery.Result[TMeta]{
			Documents: empty,
			Evidence:  tensor.RerankResult{Ranking: nil, CandidateIDs: nil, CandidateBudget: 0},
		}, err
	}
	return result, nil
}

func (a *Adapter[TMeta]) query(
	ctx context.Context,
	request retrieval.Query[tensorquery.Intent],
) (tensorquery.Result[TMeta], error) {
	if a == nil {
		return tensorquery.Result[TMeta]{}, ragy.ErrInvalidArgument
	}
	request = tensorquery.CopyRequest(request)
	if err := request.Read.Check(ctx); err != nil {
		return tensorquery.Result[TMeta]{}, err
	}

	prepared, err := retrieval.PrepareRead(ctx, request, a)
	if err != nil {
		return tensorquery.Result[TMeta]{}, err
	}
	if err = a.validateQuery(prepared); err != nil {
		return tensorquery.Result[TMeta]{}, err
	}

	lock, err := durablefs.Lock(ctx, filepath.Join(a.root, "target.lock"), false)
	if err != nil {
		return tensorquery.Result[TMeta]{}, err
	}
	defer func() { _ = lock.Close() }()
	selected, err := a.selectCatalogs(ctx, prepared.Read)
	if err != nil {
		return tensorquery.Result[TMeta]{}, err
	}
	candidates, docs, err := a.loadCandidates(ctx, prepared, selected)
	if err != nil {
		return tensorquery.Result[TMeta]{}, err
	}
	evidence, err := tensor.Rerank(
		ctx,
		prepared.Intent.Embedding,
		candidates,
		tensor.RerankOptions{CandidateBudget: prepared.Intent.CandidateBudget, TopK: prepared.Options.TopK},
	)
	if err != nil {
		return tensorquery.Result[TMeta]{}, err
	}
	ranked := make([]retrieval.Document[TMeta], 0, len(evidence.Ranking))
	for i := range evidence.Ranking {
		evidence.Ranking[i].Semantics = string(a.scoreSemantics())
		score := evidence.Ranking[i]
		doc := docs[score.ID]
		doc.Score, doc.Rank, doc.ScoreState, doc.ScoreSemantics = score.Score, score.Rank, retrieval.ScorePresent, a.scoreSemantics()
		ranked = append(ranked, doc)
	}
	return tensorquery.Result[TMeta]{Documents: retrieval.NewResultSet(ranked, nil), Evidence: evidence}, nil
}

func (a *Adapter[TMeta]) validateQuery(request retrieval.Query[tensorquery.Intent]) error {
	intent := request.Intent
	if intent.CandidateBudget <= 0 || intent.CandidateBudget > a.config.MaxRecords ||
		len(intent.Candidates) > intent.CandidateBudget ||
		request.Options.TopK <= 0 ||
		request.Options.TopK > intent.CandidateBudget {
		return ragy.ErrInvalidArgument
	}
	if request.Options.Threshold != nil || len(request.Options.Vector) != 0 || request.Options.Graph != nil {
		return ragy.ErrUnsupported
	}
	if err := request.Options.Validate(); err != nil {
		return err
	}
	if intent.Embedding.Space != a.config.Space {
		return ragy.ErrInvalidArgument
	}
	if err := intent.Embedding.Validate(); err != nil {
		return err
	}
	seen := make(map[source.Reference]struct{}, len(intent.Candidates))
	for _, ref := range intent.Candidates {
		if err := ref.Validate(); err != nil {
			return err
		}
		if ref.Namespace != a.config.Namespace {
			return ragy.ErrInvalidArgument
		}
		if _, exists := seen[ref]; exists {
			return ragy.ErrInvalidArgument
		}
		seen[ref] = struct{}{}
	}
	return nil
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
			manifest.Tombstone {
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

func (a *Adapter[TMeta]) loadCandidates(
	ctx context.Context,
	request retrieval.Query[tensorquery.Intent],
	entries []catalog,
) ([]tensor.Candidate, map[string]retrieval.Document[TMeta], error) {
	wanted := make(map[source.Reference]struct{}, len(request.Intent.Candidates))
	for _, ref := range request.Intent.Candidates {
		wanted[ref] = struct{}{}
	}
	docs := make(map[string]retrieval.Document[TMeta])
	var candidates []tensor.Candidate
	for _, entry := range entries {
		for _, record := range entry.Records {
			if _, exists := wanted[record.Reference]; !exists {
				continue
			}
			if err := request.Read.Check(ctx); err != nil {
				return nil, nil, err
			}
			allowed, err := filter.MatchCondition(
				request.Options.Filters,
				func(field string) (any, bool) { value, exists := record.Attributes[field]; return value, exists },
			)
			if err != nil {
				return nil, nil, err
			}
			if !allowed {
				continue
			}
			candidate, doc, err := a.loadCandidate(ctx, request.Read, entry, record)
			if err != nil {
				return nil, nil, err
			}
			candidates = append(candidates, candidate)
			docs[doc.ID] = doc
		}
	}
	return candidates, docs, nil
}

func (a *Adapter[TMeta]) loadCandidate(
	ctx context.Context,
	read access.Binding,
	entry catalog,
	record descriptor,
) (tensor.Candidate, retrieval.Document[TMeta], error) {
	if err := read.Check(ctx); err != nil {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, err
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
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, durablefs.QueryPayloadError(err)
	}
	var stored payload
	if digest(data) != record.Digest || decodeStrict(data, &stored) != nil || stored.Schema != payloadSchema ||
		stored.Reference != record.Reference {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, ragy.ErrProtocol
	}
	if err = (tensor.Embedding{Space: entry.Space, Tokens: stored.Tokens}).Validate(); err != nil {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, ragy.ErrProtocol
	}
	if err = read.Check(ctx); err != nil {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, err
	}
	meta, err := a.config.Codec.Decode(record.Attributes)
	if err != nil {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, err
	}
	if err = read.Check(ctx); err != nil {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, err
	}
	meta, err = a.config.CloneMeta(meta)
	if err != nil {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, err
	}
	if err = read.Check(ctx); err != nil {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, err
	}
	var location source.Locator
	location.Reference, location.Kind = record.Reference, source.DocumentLocation
	if validateRecordMapping(record.Reference, stored.SourceMapping, stored.Content) != nil {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, ragy.ErrProtocol
	}
	id, err := location.Identity()
	if err != nil {
		return tensor.Candidate{}, retrieval.Document[TMeta]{}, err
	}
	return tensor.Candidate{
		ID:        id,
		Embedding: tensor.Embedding{Space: entry.Space, Tokens: stored.Tokens},
	}, retrieval.Document[TMeta]{
		ID:             id,
		Content:        stored.Content,
		SourceMapping:  stored.SourceMapping,
		Meta:           meta,
		SourceSupports: []source.Locator{location},
	}, nil
}

func (a *Adapter[TMeta]) scoreSemantics() retrieval.ScoreSemantics {
	data, _ := json.Marshal(a.config.Space)
	return retrieval.ScoreSemantics(tensor.MaxSimSemantics + ":" + digest(data))
}

// AdmitPublication enforces partial branch exclusion before target reads.
func (a *Adapter[TMeta]) AdmitPublication(publication access.Publication) error {
	if a == nil {
		return access.Protect(ragy.ErrInvalidArgument)
	}
	return publication.AdmitTarget(a.config.Target)
}
