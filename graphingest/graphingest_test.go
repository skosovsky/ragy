package graphingest

import (
	"context"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/testutil"
)

type stubSplitter struct {
	chunks []ragy.Chunk
	err    error
}

func (s *stubSplitter) Split(_ context.Context, _ ragy.Document) ([]ragy.Chunk, error) {
	return s.chunks, s.err
}

type recordingStore struct {
	calls    int
	snapshot graph.Snapshot
	schema   graph.Schema
}

func (s *recordingStore) Traverse(context.Context, graph.TraversalRequest) (graph.Snapshot, error) {
	return graph.Snapshot{}, nil
}

func (s *recordingStore) Upsert(_ context.Context, snapshot graph.Snapshot) error {
	s.calls++
	s.snapshot = snapshot
	return nil
}

func (s *recordingStore) Schema() graph.Schema {
	return s.schema
}

func TestNewStageRejectsMissingDependencies(t *testing.T) {
	base := &stubSplitter{}
	provider := &testutil.GraphProvider{}
	store := &testutil.GraphStore{GraphSchema: graph.EmptySchema()}

	if _, err := NewStage(nil, provider, store); err == nil {
		t.Fatal("NewStage(nil, provider, store) error = nil, want error")
	}

	if _, err := NewStage(base, nil, store); err == nil {
		t.Fatal("NewStage(base, nil, store) error = nil, want error")
	}

	if _, err := NewStage(base, provider, nil); err == nil {
		t.Fatal("NewStage(base, provider, nil) error = nil, want error")
	}
}

func TestStageRunExtractsAndUpsertsGraph(t *testing.T) {
	base := &stubSplitter{
		chunks: []ragy.Chunk{{
			ID:       "chunk-1",
			SourceID: "doc-1",
			Index:    0,
			Content:  "hello",
		}},
	}
	provider := &testutil.GraphProvider{
		Snapshot: graph.Snapshot{
			Nodes: []graph.Node{{
				ID:     "node-1",
				Labels: []string{"Doc"},
			}},
		},
	}
	store := &testutil.GraphStore{GraphSchema: graph.EmptySchema()}

	stage, err := NewStage(base, provider, store)
	if err != nil {
		t.Fatalf("NewStage(): %v", err)
	}

	if _, ok := any(stage).(chunking.Splitter); ok {
		t.Fatal("Stage must not implement chunking.Splitter")
	}

	result, err := stage.Run(context.Background(), ragy.Document{
		ID:      "doc-1",
		Content: "hello",
	})
	if err != nil {
		t.Fatalf("Run(): %v", err)
	}

	if len(result.Chunks) != 1 {
		t.Fatalf("len(result.Chunks) = %d, want 1", len(result.Chunks))
	}

	if len(result.Snapshot.Nodes) != 1 || result.Snapshot.Nodes[0].ID != "node-1" {
		t.Fatalf("result.Snapshot = %#v, want node-1", result.Snapshot)
	}

	if len(store.Snapshot.Nodes) != 1 || store.Snapshot.Nodes[0].ID != "node-1" {
		t.Fatalf("store.Snapshot = %#v, want upserted snapshot", store.Snapshot)
	}
}

func TestStageRunRejectsInvalidProviderSnapshotBeforeUpsert(t *testing.T) {
	base := &stubSplitter{
		chunks: []ragy.Chunk{{
			ID:       "chunk-1",
			SourceID: "doc-1",
			Index:    0,
			Content:  "hello",
		}},
	}
	provider := &testutil.GraphProvider{
		Snapshot: graph.Snapshot{
			Nodes: []graph.Node{{
				ID:     "node-1",
				Labels: []string{"bad-label"},
			}},
		},
	}
	store := &recordingStore{schema: graph.EmptySchema()}

	stage, err := NewStage(base, provider, store)
	if err != nil {
		t.Fatalf("NewStage(): %v", err)
	}

	_, err = stage.Run(context.Background(), ragy.Document{
		ID:      "doc-1",
		Content: "hello",
	})
	if err == nil {
		t.Fatal("Run() error = nil, want error")
	}
	if store.calls != 0 {
		t.Fatalf("upsert calls = %d, want 0", store.calls)
	}
}

func TestStageRunRejectsDanglingProviderSnapshotBeforeUpsert(t *testing.T) {
	base := &stubSplitter{
		chunks: []ragy.Chunk{{
			ID:       "chunk-1",
			SourceID: "doc-1",
			Index:    0,
			Content:  "hello",
		}},
	}
	provider := &testutil.GraphProvider{
		Snapshot: graph.Snapshot{
			Nodes: []graph.Node{{
				ID:     "node-1",
				Labels: []string{"Doc"},
			}},
			Edges: []graph.Edge{{
				ID:       "edge-1",
				SourceID: "node-1",
				TargetID: "missing",
				Type:     "LINKS",
			}},
		},
	}
	store := &recordingStore{schema: graph.EmptySchema()}

	stage, err := NewStage(base, provider, store)
	if err != nil {
		t.Fatalf("NewStage(): %v", err)
	}

	_, err = stage.Run(context.Background(), ragy.Document{
		ID:      "doc-1",
		Content: "hello",
	})
	if err == nil {
		t.Fatal("Run() error = nil, want error")
	}
	if store.calls != 0 {
		t.Fatalf("upsert calls = %d, want 0", store.calls)
	}
}
