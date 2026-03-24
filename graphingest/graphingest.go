// Package graphingest provides explicit graph ingestion stages.
package graphingest

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/graph"
)

// Provider extracts a graph snapshot from chunks.
type Provider interface {
	Extract(ctx context.Context, chunks []ragy.Chunk) (graph.Snapshot, error)
}

// Stage runs chunking and graph upsert as an explicit ingestion stage.
type Stage struct {
	base     chunking.Splitter
	provider Provider
	store    graph.Store
	schema   graph.Schema
}

// Result is the explicit output of a graph ingestion run.
type Result struct {
	Chunks   []ragy.Chunk
	Snapshot graph.Snapshot
}

// NewStage constructs a graph ingestion stage.
func NewStage(base chunking.Splitter, provider Provider, store graph.Store) (*Stage, error) {
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

		return &Stage{
			base:     base,
			provider: provider,
			store:    store,
			schema:   schema,
		}, nil
	}
}

// Run splits the source document and writes graph data as a side effect.
func (s *Stage) Run(ctx context.Context, doc ragy.Document) (Result, error) {
	chunks, err := s.base.Split(ctx, doc)
	if err != nil {
		return Result{}, err
	}

	snapshot, err := s.provider.Extract(ctx, chunks)
	if err != nil {
		return Result{}, err
	}
	snapshot, err = s.schema.NormalizeSnapshot(snapshot)
	if err != nil {
		return Result{}, err
	}

	if err := s.store.Upsert(ctx, snapshot); err != nil {
		return Result{}, err
	}

	return Result{
		Chunks:   chunks,
		Snapshot: snapshot,
	}, nil
}
