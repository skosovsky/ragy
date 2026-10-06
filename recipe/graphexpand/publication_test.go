package graphexpand_test

import (
	"context"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

func TestLocalExpansionRetainsPublishedOriginalSourceSupports(t *testing.T) {
	// Arrange: physically stage and publish the reference graph through durable lifecycle.
	f := newFixture(t)
	payload, manifest, original := sourcePlan()
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[managed.Payload[metadata]]{
		Store:   f.store,
		Targets: []lifecycle.Registration[managed.Payload[metadata]]{{Name: "graph", Port: f.config.Adapter}},
		ClonePayload: func(p managed.Payload[metadata]) (managed.Payload[metadata], error) {
			p.Nodes = slices.Clone(p.Nodes)
			p.Edges = slices.Clone(p.Edges)
			for i, node := range p.Nodes {
				p.Nodes[i].Value.Labels = slices.Clone(node.Value.Labels)
			}
			return p, nil
		},
		Now:             func() time.Time { return f.now },
		ValidatePayload: func(lifecycle.Manifest, managed.Payload[metadata]) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err = executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Stage(ctx, "n", manifest.ID, "graph", payload); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(ctx, "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
	publication, err := lifecycle.CapturePublication(ctx, f.store, "n", []string{"graph"})
	if err != nil {
		t.Fatal(err)
	}
	bindPublication(t, f, publication)
	f.request.HostBasis = ""
	f.store.loads = 0
	f.cloned = nil
	// Act.
	result, ledger, err := run(ctx, t, f)
	// Assert: graph fact IDs are distinct from the original source citation.
	if err != nil || len(result.Evidence.Snapshot.Nodes) != 3 || len(result.Evidence.Snapshot.Edges) != 3 ||
		!slices.Equal(
			result.SourceReferences(),
			[]source.Reference{original},
		) || ledger.Snapshot().Occupied.ModelCalls != 0 || f.store.loads != 1 {
		t.Fatal(result, err, ledger.Snapshot(), f.store.loads)
	}
}

func sourcePlan() (managed.Payload[metadata], lifecycle.Manifest, source.Reference) {
	original := source.Reference{
		Namespace:         "n",
		Source:            "policy",
		Revision:          "r1",
		Transformation:    "original",
		AccessFingerprint: "acl",
		Artifact:          "original-p1",
		Representation:    "utf8",
	}
	input := snapshot()
	payload := managed.Payload[metadata]{Nodes: nil, Edges: nil}
	var artifacts []lifecycle.Artifact
	for _, node := range input.Nodes {
		if node.ID == "private" {
			continue
		}
		reference := original
		reference.Transformation = "ingest"
		reference.Artifact = node.ID
		reference.Representation = "graph-node"
		payload.Nodes = append(payload.Nodes, managed.Node[metadata]{Reference: reference, Value: node})
		artifacts = append(artifacts, lifecycle.Artifact{Reference: reference, Supports: []source.Reference{original}})
	}
	for _, edge := range input.Edges {
		if edge.ID == "private-edge" {
			continue
		}
		reference := original
		reference.Transformation = "ingest"
		reference.Artifact = edge.ID
		reference.Representation = "graph-edge"
		payload.Edges = append(payload.Edges, managed.Edge[metadata]{Reference: reference, Value: edge})
		artifacts = append(artifacts, lifecycle.Artifact{Reference: reference, Supports: []source.Reference{original}})
	}
	manifest := lifecycle.Manifest{
		ID:                  "source-plan",
		Key:                 "source-plan",
		Payload:             "source-content",
		ExpectedPublication: "",
		State:               lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         "policy",
			Revision:       "r1",
			Content:        "source-content",
			Transformation: "ingest",
			Access:         "acl",
		},
		Targets: []lifecycle.Target{
			{Name: "graph", Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
		},
	}
	return payload, manifest, original
}

func bindPublication(t *testing.T, f *fixture, publication access.Publication) {
	t.Helper()
	schema := f.config.Adapter.Schema().NodeAttributes
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	f.request.Read, err = access.Scoped(
		access.ScopedConfig{
			Snapshot:    f.request.Read.Snapshot(),
			Mandatory:   mandatory,
			Schema:      schema,
			Publication: publication,
			Now:         func() time.Time { return f.now },
			Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
				if f.epoch != 7 {
					return ragy.ErrUnavailable
				}
				return nil
			}),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
}
