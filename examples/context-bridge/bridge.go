package bridge

import (
	"context"
	"encoding/json"
	"unicode/utf8"

	"github.com/skosovsky/contexty"
	"github.com/skosovsky/memy"

	"github.com/skosovsky/ragy/retrieval"
)

// Run captures an epoch, recalls canonical payload, renders, encodes and publishes under a lineage fence.
// Empty batches return a durable empty-context snapshot without invoking Publish.
func (b Bridge[P, R, A, M, U]) Run(ctx context.Context) (Published[U], error) {
	out, preparationErr := b.prepare(ctx)
	if preparationErr != nil {
		return Published[U]{}, failure("prepare", preparationErr)
	}
	if err := ctx.Err(); err != nil {
		return Published[U]{}, failure("publication", err)
	}
	refs := make([]memy.RevisionRef, 0, len(out.Evidence.References))
	for _, ref := range out.Evidence.References {
		refs = append(refs, memy.RevisionRef{RecordID: ref.RecordID, Revision: ref.Revision})
	}
	if len(refs) == 0 {
		current, err := b.Engine.Fence(ctx, b.Authority, b.Scope, b.Purpose)
		if err != nil {
			return Published[U]{}, failure("empty-gate", err)
		}
		if current.Epoch != out.Evidence.Epoch {
			return Published[U]{}, failure("empty-gate", memy.ErrStaleInput)
		}
		return out, nil
	}
	publicationErr := b.Engine.WithDerivedWrite(
		ctx,
		b.Authority,
		memy.EpochFence{Scope: b.Scope, Epoch: out.Evidence.Epoch},
		b.Purpose,
		refs,
		func(writeCtx context.Context) error {
			if err := b.Read.Check(writeCtx); err != nil {
				return err
			}
			snapshot, decodeErr := Decode[U](writeCtx, out.Durable, Registry[U](b.Options.UncertaintyType))
			if decodeErr != nil {
				return decodeErr
			}
			if err := b.Publish(writeCtx, snapshot); err != nil {
				return err
			}
			return writeCtx.Err()
		},
	)
	if publicationErr != nil {
		return Published[U]{}, failure("publication", publicationErr)
	}
	return out, nil
}

func (b Bridge[P, R, A, M, U]) validate() error {
	if b.Engine == nil || b.Retrieve == nil || b.Map == nil || b.Project == nil || b.Publish == nil ||
		b.MaxCandidates < 1 || b.MaxCandidates > memy.MaxSearchCandidates {
		return memy.ErrInvalid
	}
	o := b.Options
	if o.UncertaintyType == "" || o.MessageID == "" || (o.Role != contexty.RoleUser && o.Role != contexty.RoleTool) ||
		o.Render.DedupKey != nil || o.TruncateRunes < 0 || !utf8.ValidString(o.Prefix+o.Suffix) {
		return memy.ErrInvalid
	}
	l := o.Limits
	if l.Bytes <= 0 || l.Runes <= 0 || l.JSONBytes <= 0 || l.Tokens <= 0 || l.Tokenizer == "" ||
		l.MeasureTokens == nil {
		return memy.ErrInvalid
	}
	return b.Scope.Validate()
}

func (b Bridge[P, R, A, M, U]) prepare(ctx context.Context) (Published[U], error) {
	if err := b.validate(); err != nil {
		return Published[U]{}, err
	}
	if err := b.Read.Check(ctx); err != nil {
		return Published[U]{}, err
	}
	fence, err := b.Engine.Fence(ctx, b.Authority, b.Scope, b.Purpose)
	if err != nil {
		return Published[U]{}, err
	}
	batch, err := b.Retrieve(ctx)
	if err != nil {
		return Published[U]{}, err
	}
	items, err := b.mapBatch(ctx, batch)
	if err != nil {
		return Published[U]{}, err
	}
	recalled, err := memy.Recall(
		ctx,
		b.Engine,
		b.Authority,
		b.Scope,
		"context",
		search[M]{scope: b.Scope, items: items, coverage: batch.Coverage},
		memy.ScoreRanker[P, R]{},
		memy.RecallOptions{
			Read:   memy.ReadOptions{Purpose: b.Purpose},
			Search: memy.SearchOptions{MaxCandidates: b.MaxCandidates},
			Limit:  b.MaxCandidates,
		},
	)
	if err != nil {
		return Published[U]{}, err
	}
	// A stale/unknown canonical reference is a protocol outcome, never empty success.
	if recalled.Progress.CanonicalFiltered > 0 {
		return Published[U]{}, memy.ErrStaleInput
	}
	docs, refs, uncertainties, err := b.project(ctx, recalled, items)
	if err != nil {
		return Published[U]{}, err
	}
	artifact, err := (retrieval.DefaultArtifactRenderer[M]{}).Render(ctx, b.Read,
		retrieval.NewResultSet(docs, retrieval.DocumentIDResolver[M]{}), b.Options.Render)
	if err != nil {
		return Published[U]{}, err
	}
	side := newSidecar(
		artifact,
		refs,
		uncertainties,
		b.Scope,
		fence.Epoch,
		batch,
		recalled.Progress,
		items,
		recalled.Records,
	)
	side.UncertaintyType = b.Options.UncertaintyType
	side, err = transform(ctx, side, b.Options)
	if err != nil {
		return Published[U]{}, err
	}
	return encode(ctx, b.Options, side)
}

func (b Bridge[P, R, A, M, U]) project(
	ctx context.Context,
	result memy.RecallResult[P, R],
	items []mapped[M],
) ([]retrieval.Document[M], []Reference, []*U, error) {
	docs := make([]retrieval.Document[M], 0, len(result.Records))
	refs := make([]Reference, 0, len(result.Records))
	uncertain := make([]*U, 0, len(result.Records))
	for _, ranked := range result.Records {
		if err := ctx.Err(); err != nil {
			return nil, nil, nil, err
		}
		r := ranked.Record
		input := findMapped(items, r.ID)
		if input == nil || input.ref.Revision != r.Revision || input.ref.Scope != r.Scope {
			return nil, nil, nil, memy.ErrStaleInput
		}
		doc, err := b.Project(ctx, r)
		if err != nil {
			return nil, nil, nil, err
		}
		if doc.ID != r.ID {
			return nil, nil, nil, memy.ErrInvalid
		}
		if provenanceErr := validateProvenance(doc, r); provenanceErr != nil {
			return nil, nil, nil, provenanceErr
		}
		// Restore retrieval score evidence only, never index payload or metadata.
		doc.Score = input.doc.Score
		doc.ScoreState = input.doc.ScoreState
		doc.ScoreSemantics = input.doc.ScoreSemantics
		doc.ScoreHistory = append([]retrieval.ScoreObservation(nil), input.doc.ScoreHistory...)
		doc.Rank = input.doc.Rank
		var value *U
		if b.Uncertainty != nil {
			value, err = b.Uncertainty(ctx, r)
			if err != nil {
				return nil, nil, nil, err
			}
		}
		docs = append(docs, doc)
		refs = append(refs, input.ref)
		uncertain = append(uncertain, value)
	}
	return docs, refs, uncertain, ctx.Err()
}

func validateProvenance[P, R, M any](doc retrieval.Document[M], record memy.Record[P, R]) error {
	if err := retrieval.ValidateDocument(doc); err != nil {
		return memy.ErrInvalid
	}
	locs := append(doc.SourceMapping.Supports(), doc.SourceSupports...)
	for _, loc := range locs {
		if err := loc.Validate(); err != nil {
			return memy.ErrInvalid
		}
		if loc.Reference.Namespace != record.Scope.Key() {
			return memy.ErrScopeViolation
		}
		found := false
		for _, s := range record.Provenance.Sources {
			if s.ID == loc.Reference.Source && s.Revision == loc.Reference.Revision {
				found = true
				break
			}
		}
		if !found {
			return memy.ErrStaleInput
		}
	}
	return nil
}

// PublicJSON exposes only host-approved role and rendered text; sidecar stays private.
func PublicJSON(role contexty.Role, text string) ([]byte, error) {
	return json.Marshal(struct {
		Role contexty.Role `json:"role"`
		Text string        `json:"text"`
	}{role, text})
}

// Wrap shifts snippet coordinates into a final decoded text envelope.
func Wrap(text, prefix, suffix string) string { return prefix + text + suffix }

func findMapped[M any](items []mapped[M], id string) *mapped[M] {
	for i := range items {
		if items[i].ref.RecordID == id {
			return &items[i]
		}
	}
	return nil
}
