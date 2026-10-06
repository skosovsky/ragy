package graphingest_test

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/graphingest"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/graphingest/materialization"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type graphAttrs struct{ Label string }
type graphMeta struct {
	Label string `json:"-"`
}

func graphPipeline(t *testing.T) *graphingest.Pipeline[struct{}, string, string, graphAttrs, graphMeta] {
	t.Helper()
	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatal(err)
	}
	extractor, err := extraction.New(extraction.Config[struct{}, string, string, graphAttrs]{
		OntologyIdentity: "host-ontology",
		Configuration:    "fixture-extraction",
		Schema:           schema,
		MaxSnippets:      8,
		MaxInputBytes:    1024,
		MaxEntities:      8,
		MaxRelations:     8,
		MaxSupports:      32,
		Duration:         time.Second,
		Now:              time.Now,
		CloneAccess:      func(v struct{}) (struct{}, error) { return v, nil },
		Attributes:       func(struct{}) (filter.RawAttributes, error) { return filter.RawAttributes{}, nil },
		AdmitSnippet:     func(context.Context, access.Binding, extraction.Snippet[struct{}]) error { return nil },
		CloneAttributes:  func(v graphAttrs) (graphAttrs, error) { return v, nil },
		ValidateEntity:   func(string, graphAttrs) error { return nil },
		ValidateRelation: func(string, string, string, graphAttrs) error { return nil },
		Quote: func(context.Context) (budget.Reservation, error) {
			return budget.Reservation{
				Kind:      budget.Model,
				Usage:     budget.Usage{InputTokens: 128, OutputTokens: 128, Cost: 1},
				CostKnown: true,
			}, nil
		},
		CountInputTokens: func(extraction.ModelInput) (uint64, error) { return 10, nil },
		Model: func(context.Context, extraction.ModelInput) (extraction.ModelOutput[string, string, graphAttrs], extraction.Usage, error) {
			return extraction.ModelOutput[string, string, graphAttrs]{
				Entities: []extraction.Entity[string, graphAttrs]{
					{
						ID:         "svc",
						Name:       "Billing",
						Kind:       "Service",
						Attributes: graphAttrs{Label: "Billing"},
						Snippets:   []int{0},
					},
					{
						ID:         "db",
						Name:       "LedgerDB",
						Kind:       "Database",
						Attributes: graphAttrs{Label: "LedgerDB"},
						Snippets:   []int{0},
					},
				},
				Relations: []extraction.Relation[string, graphAttrs]{
					{ID: "dependency", From: "svc", To: "db", Kind: "depends_on", Snippets: []int{0}},
				},
			}, extraction.Usage{
				Known: true,
				Value: budget.Usage{InputTokens: 10, OutputTokens: 20, Cost: 1},
			}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	resolver, err := resolution.New(
		resolution.Config[string, string, graphAttrs]{
			OntologyIdentity: "host-ontology",
			PolicyIdentity:   "host-identity",
			MaxEntities:      8,
			MaxRelations:     8,
			MaxSupports:      32,
			ValidateEntity:   func(string, graphAttrs) error { return nil },
			ValidateRelation: func(string, string, string, graphAttrs) error { return nil },
			Identity: func(v resolution.Entity[string, graphAttrs]) (resolution.Decision, error) {
				return resolution.Decision{
					State:     resolution.Resolved,
					Namespace: v.Namespace,
					Key:       v.Kind + ":" + v.Name,
					Name:      v.Name,
				}, nil
			},
			RelationKey:     func(v resolution.Relation[string, graphAttrs]) (string, error) { return v.Kind, nil },
			CloneAttributes: func(v graphAttrs) (graphAttrs, error) { return v, nil },
			Equivalent:      func(a, b graphAttrs) bool { return a == b },
			AdmitSupport:    func(context.Context, access.Binding, source.Locator) error { return nil },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	materializer, err := materialization.New(
		materialization.Config[string, string, graphAttrs, graphMeta]{
			OntologyIdentity: "host-ontology",
			PolicyIdentity:   "host-identity",
			Schema:           graph.EmptySchema(),
			MaxFacts:         8,
			MaxSupports:      32,
			CloneAttributes:  func(v graphAttrs) (graphAttrs, error) { return v, nil },
			CloneMeta:        func(v graphMeta) (graphMeta, error) { return v, nil },
			Node: func(d resolution.Decision, kind string, _ graphAttrs) (materialization.NodeValue[graphMeta], error) {
				return materialization.NodeValue[graphMeta]{
					Labels:  []string{kind},
					Content: d.Name,
					Meta:    graphMeta{Label: d.Name},
				}, nil
			},
			Edge: func(kind string, _ graphAttrs) (materialization.EdgeValue[graphMeta], error) {
				return materialization.EdgeValue[graphMeta]{Type: kind}, nil
			},
			AdmitSupport: func(context.Context, access.Binding, source.Locator) error { return nil },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	pipeline, err := graphingest.New(
		graphingest.Config[struct{}, string, string, graphAttrs, graphMeta]{
			Extraction:      extractor,
			Resolution:      resolver,
			Materialization: materializer,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return pipeline
}

type interruptedStage struct{ adapter *managed.Adapter[graphMeta] }

func (p interruptedStage) Stage(
	ctx context.Context,
	r lifecycle.StageRequest,
	input managed.Payload[graphMeta],
) (lifecycle.StageResult, error) {
	if _, err := p.adapter.Stage(ctx, r, input); err != nil {
		return lifecycle.StageResult{}, err
	}
	return lifecycle.StageResult{}, context.DeadlineExceeded
}
func (p interruptedStage) Inspect(ctx context.Context, r lifecycle.StageRequest) (lifecycle.StageResult, error) {
	return p.adapter.Inspect(ctx, r)
}

func TestGraphPipelineExplicitManagedLifecycleHandoff(t *testing.T) {
	for _, failed := range []bool{false, true} {
		t.Run(map[bool]string{false: "published", true: "interrupted-stage"}[failed], func(t *testing.T) {
			checkGraphHandoff(t, failed)
		})
	}
}

func checkGraphHandoff(t *testing.T, failed bool) {
	t.Helper()
	// Arrange: actual mapped chunk, bounded extraction/resolution/materialization.
	result, ledger := graphPlan(t)
	store, adapter, executor := graphLifecycle(t, failed)
	var err error
	// Act: the host explicitly prepares/stages/publishes the returned plan.
	if _, err = executor.Prepare(t.Context(), result.Plan.Manifest); err != nil {
		t.Fatal(err)
	}
	_, stageErr := executor.Stage(t.Context(), "n", "graph-plan", "graph", result.Plan.Payload)
	_, publishErr := executor.Publish(t.Context(), "n", "graph-plan")
	// Assert: interrupted staging cannot masquerade as a publication.
	if failed {
		if !errors.Is(stageErr, context.DeadlineExceeded) || publishErr == nil {
			t.Fatalf("failed publication: %v %v", stageErr, publishErr)
		}
		snapshot, loadErr := store.Load(t.Context(), "n")
		if loadErr != nil || snapshot.Manifests[0].PublishedAt.IsZero() == false {
			t.Fatal("partial stage published", loadErr)
		}
		return
	}
	if stageErr != nil || publishErr != nil {
		t.Fatalf("handoff: %v %v", stageErr, publishErr)
	}
	publication, err := lifecycle.CapturePublication(t.Context(), store, "n", []string{"graph"})
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	graphResult, err := adapter.Traverse(
		t.Context(),
		managed.Request{
			Read: read,
			Traversal: graph.TraversalRequest{
				Seeds:     []string{result.Plan.Payload.Nodes[0].Value.ID},
				Depth:     1,
				Direction: graph.DirectionUndirected,
			},
			MaxNodes: 8,
			MaxEdges: 8,
		},
	)
	if err != nil || len(graphResult.Snapshot.Nodes) != 2 || len(graphResult.Snapshot.Edges) != 1 ||
		len(graphResult.Supports) != 3 ||
		ledger.Snapshot().Occupied.ModelCalls != 1 {
		t.Fatalf("managed result: %v %#v", err, graphResult)
	}
}

func graphPlan(t *testing.T) (graphingest.Result[string, string, graphAttrs, graphMeta], *budget.Ledger) {
	t.Helper()
	pipeline := graphPipeline(t)
	text := "Billing depends on LedgerDB."
	location := source.Locator{
		Reference: source.Reference{
			Namespace:         "n",
			Source:            "policy",
			Revision:          "r1",
			Transformation:    "original",
			AccessFingerprint: "acl",
			Artifact:          "original",
			Representation:    "utf8",
		},
		Kind: source.TextLocation,
		Span: source.ByteSpan{Start: 0, End: len(text)},
	}
	mapping, err := source.OriginalText(location, text)
	if err != nil {
		t.Fatal(err)
	}
	splitter, err := chunking.NewRecursive[struct{}](32, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := splitter.Split(
		t.Context(),
		retrieval.Document[struct{}]{ID: "policy", Content: text, SourceMapping: mapping},
	)
	if err != nil {
		t.Fatal(err)
	}
	input := []extraction.Snippet[struct{}]{{Namespace: "prod", Mapping: chunks[0].SourceMapping}}
	ledger, err := budget.New(
		budget.Config{
			Limits: budget.Limits{
				ModelCalls: 1,
				Usage:      budget.Usage{InputTokens: 128, OutputTokens: 128, Cost: 1},
			},
			Deadline:         time.Now().Add(time.Minute),
			Now:              time.Now,
			RequireKnownCost: true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := pipeline.Build(
		t.Context(),
		access.Unrestricted(),
		ledger,
		input,
		materialization.Request{
			Identity: lifecycle.Identity{
				Namespace:      "n",
				Source:         "policy",
				Revision:       "r1",
				Content:        "content-fingerprint",
				Transformation: "extraction",
				Access:         "acl",
			},
			Target:             "graph",
			ManifestID:         "graph-plan",
			Key:                "graph-plan",
			PayloadFingerprint: "payload-fingerprint",
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return result, ledger
}

func graphLifecycle(
	t *testing.T,
	failed bool,
) (*filestore.Store, *managed.Adapter[graphMeta], *lifecycle.Executor[managed.Payload[graphMeta]]) {
	t.Helper()
	store, err := filestore.New(filepath.Join(t.TempDir(), "ledger"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := managed.New(
		managed.Config[graphMeta]{
			Namespace:  "n",
			Target:     "graph",
			Store:      store,
			Schema:     graph.EmptySchema(),
			CloneMeta:  func(m graphMeta) (graphMeta, error) { return m, nil },
			MaxRecords: 8, MaxAdmissionRecords: 8,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	var port lifecycle.StagePort[managed.Payload[graphMeta]] = adapter
	if failed {
		port = interruptedStage{adapter: adapter}
	}
	executor, err := lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[managed.Payload[graphMeta]]{
			Store:           store,
			Targets:         []lifecycle.Registration[managed.Payload[graphMeta]]{{Name: "graph", Port: port}},
			ClonePayload:    func(p managed.Payload[graphMeta]) (managed.Payload[graphMeta], error) { return p, nil },
			ValidatePayload: func(lifecycle.Manifest, managed.Payload[graphMeta]) error { return nil },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return store, adapter, executor
}
