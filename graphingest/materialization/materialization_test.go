package materialization_test

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/graphingest/materialization"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/source"
)

type attrs struct{ Tags []string }
type metadata struct {
	Tags []string `json:"-"`
}

func location(src string) source.Locator {
	return source.Locator{
		Kind: source.DocumentLocation,
		Reference: source.Reference{
			Namespace:         "n",
			Source:            src,
			Revision:          "r1",
			Transformation:    "original",
			AccessFingerprint: "acl",
			Artifact:          src + "-chunk",
			Representation:    "text",
		},
	}
}
func resolved(t *testing.T) resolution.Result[string, string, attrs] {
	t.Helper()
	cfg := resolution.Config[string, string, attrs]{
		OntologyIdentity: "ontology",
		PolicyIdentity:   "alias-policy",
		MaxEntities:      20,
		MaxRelations:     20,
		MaxSupports:      40,
		ValidateEntity:   func(string, attrs) error { return nil },
		ValidateRelation: func(string, string, string, attrs) error { return nil },
		Identity: func(mention resolution.Entity[string, attrs]) (resolution.Decision, error) {
			name := mention.Name
			if name == "Pay" {
				name = "Billing"
			}
			return resolution.Decision{
				State:     resolution.Resolved,
				Namespace: mention.Namespace,
				Key:       mention.Kind + ":" + name,
				Name:      name,
			}, nil
		},
		RelationKey:     func(edge resolution.Relation[string, attrs]) (string, error) { return edge.Kind, nil },
		CloneAttributes: func(a attrs) (attrs, error) { a.Tags = slices.Clone(a.Tags); return a, nil },
		Equivalent:      func(a, b attrs) bool { return slices.Equal(a.Tags, b.Tags) },
		AdmitSupport:    func(context.Context, access.Binding, source.Locator) error { return nil },
	}
	resolver, err := resolution.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	input := resolution.Extraction[string, string, attrs]{}
	for _, row := range []struct{ id, src, name, kind string }{{"s1bill", "s1", "Pay", "Service"}, {"s1db", "s1", "LedgerDB", "Database"}, {"team", "s1", "Team A", "Team"}, {"s2bill", "s2", "Billing", "Service"}, {"s2db", "s2", "LedgerDB", "Database"}} {
		input.Entities = append(
			input.Entities,
			resolution.Entity[string, attrs]{
				ID:         row.id,
				Namespace:  "prod",
				Name:       row.name,
				Kind:       row.kind,
				Attributes: attrs{Tags: []string{"owned"}},
				Supports:   []source.Locator{location(row.src)},
			},
		)
	}
	for _, row := range []struct{ id, src, from, to, kind string }{{"dep1", "s1", "s1bill", "s1db", "depends_on"}, {"dep2", "s2", "s2bill", "s2db", "depends_on"}, {"owner", "s1", "s1bill", "team", "owned_by"}} {
		input.Relations = append(
			input.Relations,
			resolution.Relation[string, attrs]{
				ID:         row.id,
				From:       row.from,
				To:         row.to,
				Kind:       row.kind,
				Attributes: attrs{Tags: []string{"owned"}},
				Supports:   []source.Locator{location(row.src)},
			},
		)
	}
	output, err := resolver.Resolve(context.Background(), access.Unrestricted(), input)
	if err != nil {
		t.Fatal(err)
	}
	return output
}
func config() materialization.Config[string, string, attrs, metadata] {
	return materialization.Config[string, string, attrs, metadata]{
		OntologyIdentity: "ontology",
		PolicyIdentity:   "alias-policy",
		Schema:           graph.EmptySchema(),
		MaxFacts:         40,
		MaxSupports:      40,
		CloneAttributes:  func(a attrs) (attrs, error) { a.Tags = slices.Clone(a.Tags); return a, nil },
		CloneMeta:        func(a metadata) (metadata, error) { a.Tags = slices.Clone(a.Tags); return a, nil },
		Node: func(identity resolution.Decision, kind string, a attrs) (materialization.NodeValue[metadata], error) {
			return materialization.NodeValue[metadata]{
				Labels:  []string{kind},
				Content: identity.Name,
				Meta:    metadata(a),
			}, nil
		},
		Edge: func(kind string, a attrs) (materialization.EdgeValue[metadata], error) {
			return materialization.EdgeValue[metadata]{Type: kind, Meta: metadata(a)}, nil
		},
		AdmitSupport: func(context.Context, access.Binding, source.Locator) error { return nil },
	}
}
func request(src string) materialization.Request {
	return materialization.Request{
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         src,
			Revision:       "r1",
			Content:        src,
			Transformation: "extraction",
			Access:         "acl",
		},
		Target:             "graph",
		ManifestID:         src,
		Key:                src,
		PayloadFingerprint: src,
	}
}
func TestResolvedMaterializationPublicationAndSharedSupportCleanup(t *testing.T) {
	// Arrange: actual resolver, materializer, durable ledger and managed graph.
	ctx := context.Background()
	input := resolved(t)
	cfg := config()
	builder, err := materialization.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	store, err := filestore.New(filepath.Join(t.TempDir(), "ledger"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := managed.New(
		managed.Config[metadata]{
			Namespace:  "n",
			Target:     "graph",
			Store:      store,
			Schema:     cfg.Schema,
			CloneMeta:  cfg.CloneMeta,
			MaxRecords: 40, MaxAdmissionRecords: 40,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	executor, err := lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[managed.Payload[metadata]]{
			Store:           store,
			Targets:         []lifecycle.Registration[managed.Payload[metadata]]{{Name: "graph", Port: adapter}},
			ClonePayload:    func(p managed.Payload[metadata]) (managed.Payload[metadata], error) { return p, nil },
			ValidatePayload: func(lifecycle.Manifest, managed.Payload[metadata]) error { return nil },
			Now:             func() time.Time { return now },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	seed := ""
	// Act: independent source publications reference identical canonical node/edge IDs.
	for _, src := range []string{"s1", "s2"} {
		seed = publishSource(ctx, t, builder, executor, src, input)
	}
	before := readGraph(ctx, t, store, adapter, seed)
	if len(before.Snapshot.Nodes) != 3 || len(before.Snapshot.Edges) != 2 || len(before.Conflicts) != 0 {
		t.Fatal(before)
	}
	retire(ctx, t, store, executor, adapter, now, "s1")
	after := readGraph(ctx, t, store, adapter, seed)
	// Assert: source-only ownership disappears, shared dependency remains with s2 only.
	if len(after.Snapshot.Nodes) != 2 || len(after.Snapshot.Edges) != 1 ||
		after.Snapshot.Edges[0].Type != "depends_on" {
		t.Fatal(after)
	}
	for _, fact := range after.Supports {
		if len(fact.References) != 1 || fact.References[0].Source != "s2" || fact.References[0].Artifact != "s2-chunk" {
			t.Fatal(fact)
		}
	}
	retire(ctx, t, store, executor, adapter, now, "s2")
	if final := readGraph(
		ctx,
		t,
		store,
		adapter,
		seed,
	); len(final.Snapshot.Nodes) != 0 ||
		len(final.Snapshot.Edges) != 0 {
		t.Fatal("last support retained", final)
	}
}

func readGraph(
	ctx context.Context,
	t *testing.T,
	store lifecycle.Store,
	adapter *managed.Adapter[metadata],
	seed string,
) managed.Result[metadata] {
	t.Helper()
	publication, err := lifecycle.CapturePublication(ctx, store, "n", []string{"graph"})
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	result, err := adapter.Traverse(
		ctx,
		managed.Request{
			Read:      read,
			Traversal: graph.TraversalRequest{Seeds: []string{seed}, Direction: graph.DirectionOutbound, Depth: 2},
			MaxNodes:  50,
			MaxEdges:  100,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func retire(
	ctx context.Context,
	t *testing.T,
	store lifecycle.Store,
	executor *lifecycle.Executor[managed.Payload[metadata]],
	adapter *managed.Adapter[metadata],
	now time.Time,
	src string,
) {
	t.Helper()
	identity := request(src).Identity
	identity.Revision = "deleted"
	tombstone := lifecycle.Manifest{
		ID:                  "delete-" + src,
		Identity:            identity,
		Key:                 "delete-" + src,
		Payload:             "delete-" + src,
		ExpectedPublication: src,
		Tombstone:           true,
		State:               lifecycle.Planned,
	}
	if _, err := executor.Prepare(ctx, tombstone); err != nil {
		t.Fatal(err)
	}
	if _, err := executor.Publish(ctx, "n", tombstone.ID); err != nil {
		t.Fatal(err)
	}
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   store,
			Now:     func() time.Time { return now },
			Targets: []lifecycle.CleanupRegistration{{Name: "graph", Port: adapter}},
			Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(ctx, "n", tombstone.ID); err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Attempt(ctx, "n", tombstone.ID, src, "graph", false); err != nil {
		t.Fatal(err)
	}
}
func TestMaterializationRefusesIncompleteConflictingOrUnauthorizedSource(t *testing.T) {
	for _, scenario := range []string{"conflict", "unresolved", "endpoint", "denied", "policy", "cancelled"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			input := resolved(t)
			cfg := config()
			calls := 0
			cfg.Node = func(resolution.Decision, string, attrs) (materialization.NodeValue[metadata], error) {
				calls++
				return materialization.NodeValue[metadata]{}, nil
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch scenario {
			case "conflict":
				input.Entities[0].Variants = append(input.Entities[0].Variants, input.Entities[0].Variants[0])
			case "unresolved":
				input.Unresolved = []resolution.Unresolved{
					{Mention: "unknown", Kind: "entity", Supports: []source.Locator{location("s1")}},
				}
			case "endpoint":
				input.Entities = nil
			case "denied":
				cfg.AdmitSupport = func(context.Context, access.Binding, source.Locator) error { return ragy.ErrUnavailable }
			case "policy":
				input.PolicyIdentity = "other"
			case "cancelled":
				cancel()
			}
			builder, err := materialization.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			output, err := builder.Build(ctx, access.Unrestricted(), request("s1"), input)
			// Assert.
			if err == nil || output.Manifest.ID != "" || len(output.Payload.Nodes) != 0 || calls != 0 {
				t.Fatal(output, err, calls)
			}
			if scenario == "denied" && !errors.Is(err, ragy.ErrUnavailable) {
				t.Fatal(err)
			}
		})
	}
}

func publishSource(
	ctx context.Context,
	t *testing.T,
	builder *materialization.Materializer[string, string, attrs, metadata],
	executor *lifecycle.Executor[managed.Payload[metadata]],
	src string,
	input resolution.Result[string, string, attrs],
) string {
	t.Helper()
	seed := ""
	var err error
	plan, buildErr := builder.Build(ctx, access.Unrestricted(), request(src), input)
	if buildErr != nil {
		t.Fatal(buildErr)
	}
	for _, node := range plan.Payload.Nodes {
		if node.Value.Content == "Billing" {
			seed = node.Value.ID
		}
	}
	if _, err = executor.Prepare(ctx, plan.Manifest); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Stage(ctx, "n", src, "graph", plan.Payload); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(ctx, "n", src); err != nil {
		t.Fatal(err)
	}
	return seed
}

func TestMaterializationOwnsMetadataAndBindsPolicyTransformation(t *testing.T) {
	// Arrange: a policy/config change is reflected even when canonical facts remain the same.
	input := resolved(t)
	cfg := config()
	first, err := materialization.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	before, err := first.Build(context.Background(), access.Unrestricted(), request("s1"), input)
	if err != nil {
		t.Fatal(err)
	}
	cfg.PolicyIdentity = "changed-policy"
	input.PolicyIdentity = cfg.PolicyIdentity
	second, err := materialization.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	after, err := second.Build(context.Background(), access.Unrestricted(), request("s1"), input)
	// Assert: transformation provenance changes; original attributes remain independently owned.
	if err != nil || before.Manifest.Identity.Transformation == after.Manifest.Identity.Transformation ||
		before.Payload.Nodes[0].Value.ID != after.Payload.Nodes[0].Value.ID {
		t.Fatal(before.Manifest, after.Manifest, err)
	}
	after.Payload.Nodes[0].Value.Meta.Tags[0] = "mutated"
	if input.Entities[0].Variants[0].Attributes.Tags[0] != "owned" ||
		before.Payload.Nodes[0].Value.Meta.Tags[0] != "owned" {
		t.Fatal("materialized attributes aliased")
	}
}
