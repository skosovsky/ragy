// Package graphingest provides explicit graph ingestion stages.
package graphingest

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/retrieval"
)

// Provider extracts a graph snapshot from chunks.
type Provider[TMeta any] interface {
	Extract(ctx context.Context, chunks []chunking.Chunk[TMeta]) (graph.Snapshot[TMeta], error)
}

// Stage runs chunking and graph upsert as an explicit ingestion stage.
type Stage[TMeta any] struct {
	base     chunking.Splitter[TMeta]
	provider Provider[TMeta]
	store    graph.Store[TMeta]
	schema   graph.Schema
}

// Result is the explicit output of a graph ingestion run.
type Result[TMeta any] struct {
	Chunks   []chunking.Chunk[TMeta]
	Snapshot graph.Snapshot[TMeta]
}

// NewStage constructs a graph ingestion stage.
func NewStage[TMeta any](
	base chunking.Splitter[TMeta],
	provider Provider[TMeta],
	store graph.Store[TMeta],
) (*Stage[TMeta], error) {
	switch {
	case base == nil:
		return nil, fmt.Errorf("%w: graph ingest base splitter", ragy.ErrInvalidArgument)
	case provider == nil:
		return nil, fmt.Errorf("%w: graph ingest provider", ragy.ErrInvalidArgument)
	case store == nil:
		return nil, fmt.Errorf("%w: graph ingest store", ragy.ErrInvalidArgument)
	default:
		schema := store.Schema()
		if err := schema.Validate(); err != nil {
			return nil, err
		}

		return &Stage[TMeta]{
			base:     base,
			provider: provider,
			store:    store,
			schema:   schema,
		}, nil
	}
}

// Run splits the source document and writes graph data as a side effect.
func (s *Stage[TMeta]) Run(ctx context.Context, doc retrieval.Document[TMeta]) (Result[TMeta], error) {
	chunks, err := s.base.Split(ctx, doc)
	if err != nil {
		return Result[TMeta]{}, err
	}

	snapshot, err := s.provider.Extract(ctx, chunks)
	if err != nil {
		return Result[TMeta]{}, err
	}
	snapshot, err = graph.NormalizeSnapshot(s.schema, snapshot)
	if err != nil {
		return Result[TMeta]{}, err
	}

	if err := s.store.Upsert(ctx, snapshot); err != nil {
		return Result[TMeta]{}, err
	}

	return Result[TMeta]{
		Chunks:   chunks,
		Snapshot: snapshot,
	}, nil
}
