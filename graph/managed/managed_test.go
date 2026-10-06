package managed_test

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type metadata struct {
	Tenant     string `json:"tenant"`
	Visibility string `json:"visibility"`
	Name       string `json:"name"`
}
type fixture struct {
	adapter  *managed.Adapter[metadata]
	executor *lifecycle.Executor[managed.Payload[metadata]]
	store    lifecycle.Store
	schema   filter.Schema
	now      time.Time
	calls    []string
	epoch    int64
	revoke   bool
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	fields := filter.NewSchema()
	for _, name := range []string{"tenant", "visibility", "name"} {
		if _, err := fields.String(name); err != nil {
			t.Fatal(err)
		}
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	store, err := filestore.New(filepath.Join(t.TempDir(), "manifests"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	f := &fixture{
		adapter:  nil,
		executor: nil,
		store:    store,
		schema:   schema,
		now:      time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		calls:    nil,
		epoch:    7,
		revoke:   false,
	}
	f.adapter, err = managed.New(
		managed.Config[metadata]{
			Namespace:  "n",
			Target:     "graph",
			Store:      store,
			Schema:     graph.Schema{NodeAttributes: schema, EdgeAttributes: schema},
			MaxRecords: 100,
			CloneMeta: func(meta metadata) (metadata, error) {
				f.calls = append(f.calls, meta.Name)
				if f.revoke {
					f.epoch = 8
				}
				return meta, nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	f.executor, err = lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[managed.Payload[metadata]]{
			Store:   store,
			Targets: []lifecycle.Registration[managed.Payload[metadata]]{{Name: "graph", Port: f.adapter}},
			ClonePayload: func(input managed.Payload[metadata]) (managed.Payload[metadata], error) {
				input.Nodes = slices.Clone(input.Nodes)
				input.Edges = slices.Clone(input.Edges)
				for i := range input.Nodes {
					input.Nodes[i].Value.Labels = slices.Clone(input.Nodes[i].Value.Labels)
				}
				return input, nil
			},
			ValidatePayload: func(lifecycle.Manifest, managed.Payload[metadata]) error { return nil },
			Now:             func() time.Time { return f.now },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return f
}
func ref(sourceID, revision, id, representation string) source.Reference {
	return source.Reference{
		Namespace:         "n",
		Source:            sourceID,
		Revision:          revision,
		Transformation:    "ingest",
		AccessFingerprint: "acl",
		Artifact:          id,
		Representation:    representation,
	}
}
func payload(sourceID, revision string) managed.Payload[metadata] {
	nodes := []managed.Node[metadata]{}
	for _, row := range []struct{ id, label string }{{id: "svc", label: "Service"}, {id: "db", label: "Database"}} {
		nodes = append(
			nodes,
			managed.Node[metadata]{
				Reference: ref(sourceID, revision, row.id, "graph-node"),
				Value: graph.Node[metadata]{
					ID:      row.id,
					Labels:  []string{row.label},
					Content: row.id,
					Meta:    metadata{Tenant: "a", Visibility: "public", Name: row.id},
				},
			},
		)
	}
	return managed.Payload[metadata]{
		Nodes: nodes,
		Edges: []managed.Edge[metadata]{
			{
				Reference: ref(sourceID, revision, "e1", "graph-edge"),
				Value: graph.Edge[metadata]{
					ID:       "e1",
					SourceID: "svc",
					TargetID: "db",
					Type:     "depends_on",
					Meta:     metadata{Tenant: "a", Visibility: "public", Name: "e1"},
				},
			},
		},
	}
}
func plan(id, expected string, input managed.Payload[metadata], chunk string) lifecycle.Manifest {
	identity := input.Nodes[0].Reference
	support := ref(identity.Source, identity.Revision, chunk, "utf8")
	artifacts := []lifecycle.Artifact{}
	for _, node := range input.Nodes {
		artifacts = append(
			artifacts,
			lifecycle.Artifact{Reference: node.Reference, Supports: []source.Reference{support}},
		)
	}
	for _, edge := range input.Edges {
		artifacts = append(
			artifacts,
			lifecycle.Artifact{Reference: edge.Reference, Supports: []source.Reference{support}},
		)
	}
	return lifecycle.Manifest{
		ID:                  id,
		Key:                 id,
		Payload:             id,
		ExpectedPublication: expected,
		State:               lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         identity.Source,
			Revision:       identity.Revision,
			Content:        id,
			Transformation: "ingest",
			Access:         "acl",
		},
		Targets: []lifecycle.Target{
			{Name: "graph", Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
		},
	}
}
func (f *fixture) ingest(t *testing.T, manifest lifecycle.Manifest, input managed.Payload[metadata]) {
	t.Helper()
	ctx := context.Background()
	if _, err := f.executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Stage(ctx, "n", manifest.ID, "graph", input); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(ctx, "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
}
func (f *fixture) pin(t *testing.T) access.Binding {
	t.Helper()
	publication, err := lifecycle.CapturePublication(context.Background(), f.store, "n", []string{"graph"})
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(f.schema)
	if err != nil {
		t.Fatal(err)
	}
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	visibility, err := fields.String("visibility")
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.In(filter.Eq(builder, tenant, "a"), visibility, "public").Build()
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "policy",
				PolicyEpoch: 7,
				IssuedAt:    f.now,
				ExpiresAt:   f.now.Add(30 * time.Second),
			},
			Mandatory:   mandatory,
			Schema:      f.schema,
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
	return read
}
func request(read access.Binding) managed.Request {
	return managed.Request{
		Read:      read,
		Traversal: graph.TraversalRequest{Seeds: []string{"svc"}, Direction: graph.DirectionOutbound, Depth: 2},
		MaxNodes:  50,
		MaxEdges:  100,
	}
}
func (f *fixture) delete(t *testing.T, id, sourceID, expected string) {
	t.Helper()
	manifest := lifecycle.Manifest{
		ID:                  id,
		Key:                 id,
		Payload:             id,
		ExpectedPublication: expected,
		Tombstone:           true,
		State:               lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         sourceID,
			Revision:       "deleted",
			Content:        id,
			Transformation: "ingest",
			Access:         "acl",
		},
	}
	if _, err := f.executor.Prepare(context.Background(), manifest); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(context.Background(), "n", id); err != nil {
		t.Fatal(err)
	}
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   f.store,
			Now:     func() time.Time { return f.now },
			Targets: []lifecycle.CleanupRegistration{{Name: "graph", Port: f.adapter}},
			Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(context.Background(), "n", id); err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Attempt(context.Background(), "n", id, expected, "graph", false); err != nil {
		t.Fatal(err)
	}
}

func TestManagedGraphSharedSupportAndExactCleanup(t *testing.T) {
	// Arrange: actual graph records from two published sources support one shared edge.
	f := newFixture(t)
	policy, faq := payload("policy", "r1"), payload("faq", "r1")
	f.ingest(t, plan("policy", "", policy, "p1"), policy)
	f.ingest(t, plan("faq", "", faq, "f1"), faq)
	captured := f.pin(t)
	// Act/Assert.
	result, err := f.adapter.Traverse(context.Background(), request(captured))
	if err != nil || len(result.Snapshot.Nodes) != 2 || len(result.Snapshot.Edges) != 1 ||
		len(result.Supports[2].References) != 2 {
		t.Fatal("shared graph/support union failed", err)
	}
	f.delete(t, "delete-policy", "policy", "policy")
	result, err = f.adapter.Traverse(context.Background(), request(f.pin(t)))
	if err != nil || len(result.Snapshot.Edges) != 1 || len(result.Supports[2].References) != 1 ||
		result.Supports[2].References[0].Artifact != "f1" {
		t.Fatal("cleanup removed shared faq support", err)
	}
	unavailable, err := f.adapter.Traverse(context.Background(), request(captured))
	if !errors.Is(err, ragy.ErrUnavailable) || len(unavailable.Snapshot.Nodes) != 0 {
		t.Fatal("cleaned captured graph silently changed", err)
	}
	f.delete(t, "delete-faq", "faq", "faq")
	result, err = f.adapter.Traverse(context.Background(), request(f.pin(t)))
	if err != nil || len(result.Snapshot.Edges) != 0 || len(result.Snapshot.Nodes) != 0 {
		t.Fatal("last managed support remained", err)
	}
}
func TestManagedGraphPrivateBridgeNotTraversedOrLoaded(t *testing.T) {
	// Arrange: the only path to db crosses a forbidden node.
	f := newFixture(t)
	input := payload("policy", "r1")
	input.Nodes = append(
		input.Nodes,
		managed.Node[metadata]{
			Reference: ref("policy", "r1", "secret", "graph-node"),
			Value: graph.Node[metadata]{
				ID:      "secret",
				Labels:  []string{"Service"},
				Content: "secret payload",
				Meta:    metadata{Tenant: "b", Visibility: "private", Name: "secret"},
			},
		},
	)
	input.Edges[0].Value.TargetID = "secret"
	input.Edges = append(
		input.Edges,
		managed.Edge[metadata]{
			Reference: ref("policy", "r1", "e2", "graph-edge"),
			Value: graph.Edge[metadata]{
				ID:       "e2",
				SourceID: "secret",
				TargetID: "db",
				Type:     "depends_on",
				Meta:     metadata{Tenant: "a", Visibility: "public", Name: "e2"},
			},
		},
	)
	f.ingest(t, plan("policy", "", input, "p1"), input)
	f.calls = nil
	// Act.
	result, err := f.adapter.Traverse(context.Background(), request(f.pin(t)))
	// Assert: no payload callback for the bridge, its edges or unreachable db.
	if err != nil || len(result.Snapshot.Nodes) != 1 || len(result.Snapshot.Edges) != 0 ||
		!slices.Equal(f.calls, []string{"svc"}) {
		t.Fatal("scope allowed forbidden traversal/payload", f.calls, err)
	}
	f.calls = nil
	lookup := request(f.pin(t))
	lookup.Traversal.Seeds = []string{"secret", "svc", "nonexistent"}
	result, err = f.adapter.FindByIDs(context.Background(), lookup)
	if err != nil || len(result.Snapshot.Nodes) != 1 || len(result.Snapshot.Edges) != 0 ||
		!slices.Equal(f.calls, []string{"svc"}) {
		t.Fatal("FindByIDs loaded forbidden payload", err)
	}
}
func TestManagedGraphConflictAndRevocationFailClosed(t *testing.T) {
	// Arrange: same logical ID has conflicting canonical payloads.
	f := newFixture(t)
	first, second := payload("policy", "r1"), payload("faq", "r1")
	second.Nodes[0].Value.Content = "different canonical service"
	f.ingest(t, plan("policy", "", first, "p1"), first)
	f.ingest(t, plan("faq", "", second, "f1"), second)
	// Act/Assert: no implicit winner; conflicts retain both admitted source refs.
	result, err := f.adapter.Traverse(context.Background(), request(f.pin(t)))
	if err != nil || len(result.Conflicts) != 1 || len(result.Conflicts[0].References) != 2 ||
		len(result.Snapshot.Nodes) != 0 {
		t.Fatal("conflict was silently merged", err)
	}
	// Arrange a single-source graph for revocation during a payload callback.
	fresh := newFixture(t)
	fresh.ingest(t, plan("policy", "", first, "p1"), first)
	read := fresh.pin(t)
	fresh.revoke = true
	// Act/Assert: callback revocation suppresses the entire batch.
	result, err = fresh.adapter.Traverse(context.Background(), request(read))
	if !errors.Is(err, ragy.ErrUnavailable) || len(result.Snapshot.Nodes) != 0 || len(result.Supports) != 0 {
		t.Fatal("revoked graph payload escaped", err)
	}
}

func TestManagedGraphUnpublishedAndLostInventoryFailClosed(t *testing.T) {
	// Arrange: actual staged records exist, but no logical publication was confirmed.
	f := newFixture(t)
	input := payload("policy", "r1")
	manifest := plan("policy", "", input, "p1")
	ctx := context.Background()
	if _, err := f.executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Stage(ctx, "n", manifest.ID, "graph", input); err != nil {
		t.Fatal(err)
	}
	forged, err := access.PinPublication(
		"unconfirmed",
		[]access.TargetRevision{
			{
				Target:            "graph",
				Namespace:         "n",
				Source:            "policy",
				Revision:          "r1",
				Transformation:    "ingest",
				AccessFingerprint: "acl",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.UnrestrictedAt(forged)
	if err != nil {
		t.Fatal(err)
	}
	f.calls = nil
	// Act/Assert: even a manually constructed pinned target cannot select staged data.
	result, err := f.adapter.Traverse(ctx, request(read))
	if !errors.Is(err, ragy.ErrUnavailable) || len(result.Snapshot.Nodes) != 0 || len(f.calls) != 0 {
		t.Fatal("staged graph became readable", err)
	}
	if _, err = f.executor.Publish(ctx, "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
	captured := f.pin(t)
	restarted, err := managed.New(
		managed.Config[metadata]{
			Namespace:  "n",
			Target:     "graph",
			Store:      f.store,
			Schema:     f.adapter.Schema(),
			CloneMeta:  func(meta metadata) (metadata, error) { return meta, nil },
			MaxRecords: 100,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err = restarted.Traverse(ctx, request(captured))
	if !errors.Is(err, ragy.ErrUnavailable) || len(result.Snapshot.Nodes) != 0 {
		t.Fatal("lost in-process graph inventory substituted a snapshot", err)
	}
}

func TestManagedGraphRevisionSwapRetainsCapturedSupport(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	first := payload("policy", "r1")
	f.ingest(t, plan("r1", "", first, "p1"), first)
	old := f.pin(t)
	second := payload("policy", "r2")
	f.ingest(t, plan("r2", "r1", second, "p3"), second)
	// Act/Assert: the same logical facts now carry only the selected source revision.
	current, err := f.adapter.Traverse(context.Background(), request(f.pin(t)))
	if err != nil || current.Supports[2].References[0].Revision != "r2" ||
		current.Supports[2].References[0].Artifact != "p3" {
		t.Fatal("new graph support not selected", err)
	}
	retained, err := f.adapter.Traverse(context.Background(), request(old))
	if err != nil || retained.Supports[2].References[0].Revision != "r1" ||
		retained.Supports[2].References[0].Artifact != "p1" {
		t.Fatal("captured graph support silently changed", err)
	}
}

func TestManagedGraphCyclesAndBudgetsBeforePayloadCallbacks(t *testing.T) {
	// Arrange: a two-node cycle, observed under explicit visited/edge budgets.
	f := newFixture(t)
	input := payload("policy", "r1")
	input.Edges = append(
		input.Edges,
		managed.Edge[metadata]{
			Reference: ref("policy", "r1", "return", "graph-edge"),
			Value: graph.Edge[metadata]{
				ID:       "return",
				SourceID: "db",
				TargetID: "svc",
				Type:     "depends_on",
				Meta:     metadata{Tenant: "a", Visibility: "public", Name: "return"},
			},
		},
	)
	f.ingest(t, plan("policy", "", input, "p1"), input)
	req := request(f.pin(t))
	req.Traversal.Depth = 20
	// Act/Assert: the cycle stops at a visited set, with each node/edge once.
	result, err := f.adapter.Traverse(context.Background(), req)
	if err != nil || len(result.Snapshot.Nodes) != 2 || len(result.Snapshot.Edges) != 2 {
		t.Fatal("cycle traversal unbounded or duplicated", err)
	}
	f.calls = nil
	req.MaxNodes = 1
	result, err = f.adapter.Traverse(context.Background(), req)
	if !errors.Is(err, ragy.ErrInvalidArgument) || len(result.Snapshot.Nodes) != 0 || len(f.calls) != 0 {
		t.Fatal("visited budget exceeded or partial payload escaped", err)
	}
	req.MaxNodes = 50
	req.MaxEdges = 1
	result, err = f.adapter.Traverse(context.Background(), req)
	if !errors.Is(err, ragy.ErrInvalidArgument) || len(result.Snapshot.Edges) != 0 || len(f.calls) != 0 {
		t.Fatal("edge budget exceeded", err)
	}
}

func hostSnapshot(input managed.Payload[metadata]) graph.Snapshot[metadata] {
	out := graph.Snapshot[metadata]{Nodes: nil, Edges: nil}
	for _, node := range input.Nodes {
		out.Nodes = append(out.Nodes, node.Value)
	}
	for _, edge := range input.Edges {
		out.Edges = append(out.Edges, edge.Value)
	}
	return out
}

func TestManagedCleanupPreservesExplicitHostBasis(t *testing.T) {
	// Arrange: same facts have one managed source and a separate explicit foundation.
	f := newFixture(t)
	input := payload("policy", "r1")
	snapshot := hostSnapshot(input)
	if err := f.adapter.SetHostBasis(context.Background(), "foundation", snapshot); err != nil {
		t.Fatal(err)
	}
	f.ingest(t, plan("policy", "", input, "p1"), input)
	req := request(f.pin(t))
	req.HostBasis = "foundation"
	// Act/Assert: provenance distinguishes managed refs from host foundation identity.
	result, err := f.adapter.Traverse(context.Background(), req)
	if err != nil || len(result.Snapshot.Edges) != 1 || len(result.Supports[2].References) != 1 ||
		!slices.Equal(result.Supports[2].HostBases, []string{"foundation"}) {
		t.Fatal("host basis not associated", err)
	}
	f.delete(t, "deleted", "policy", "policy")
	req = request(f.pin(t))
	req.HostBasis = "foundation"
	result, err = f.adapter.Traverse(context.Background(), req)
	if err != nil || len(result.Snapshot.Edges) != 1 || len(result.Supports[2].References) != 0 ||
		!slices.Equal(result.Supports[2].HostBases, []string{"foundation"}) {
		t.Fatal("managed cleanup deleted host-owned facts or invented refs", err)
	}
	// No basis is selected implicitly, even though the adapter retains it.
	result, err = f.adapter.Traverse(context.Background(), request(f.pin(t)))
	if err != nil || len(result.Snapshot.Nodes) != 0 {
		t.Fatal("host basis silently selected", err)
	}
	if err = f.adapter.ReleaseHostBasis(context.Background(), "foundation"); err != nil {
		t.Fatal(err)
	}
	result, err = f.adapter.Traverse(context.Background(), req)
	if !errors.Is(err, ragy.ErrUnavailable) || len(result.Snapshot.Nodes) != 0 {
		t.Fatal("released host basis substituted", err)
	}
}

func TestHostBasisIsOwnedImmutableAndScoped(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	input := payload("policy", "r1")
	snapshot := hostSnapshot(input)
	if err := f.adapter.SetHostBasis(context.Background(), "foundation", snapshot); err != nil {
		t.Fatal(err)
	}
	snapshot.Nodes[0].Labels[0] = "Changed"
	if err := f.adapter.SetHostBasis(
		context.Background(),
		"foundation",
		snapshot,
	); !errors.Is(
		err,
		lifecycle.ErrConflict,
	) {
		t.Fatal("host basis identity reused for changed facts", err)
	}
	req := request(f.pin(t))
	req.HostBasis = "foundation"
	// Act/Assert: caller label mutation cannot alter stored foundation.
	result, err := f.adapter.Traverse(context.Background(), req)
	if err != nil || len(result.Snapshot.Nodes) != 2 || result.Snapshot.Nodes[1].Labels[0] != "Service" {
		t.Fatal("host basis ownership lost", err)
	}
	private := hostSnapshot(payload("policy", "r1"))
	private.Nodes[0].Meta.Tenant = "b"
	if err = f.adapter.SetHostBasis(context.Background(), "private", private); err != nil {
		t.Fatal(err)
	}
	f.calls = nil
	req.HostBasis = "private"
	result, err = f.adapter.Traverse(context.Background(), req)
	if err != nil || len(result.Snapshot.Nodes) != 0 || len(f.calls) != 0 {
		t.Fatal("host foundation bypassed scope", err)
	}
}

func backendQuery(read access.Binding) retrieval.Query[struct{}] {
	return retrieval.Query[struct{}]{
		Read: read,
		Options: retrieval.RetrieveOptions{
			TopK:  10,
			Graph: &retrieval.GraphOptions{Seeds: []string{"svc"}, Direction: graph.DirectionOutbound, Depth: 2},
		},
	}
}

func TestGraphBackendRankOnlyAndRevokedProjection(t *testing.T) {
	// Arrange: actual managed graph with a captured binding.
	f := newFixture(t)
	input := payload("policy", "r1")
	f.ingest(t, plan("policy", "", input, "p1"), input)
	read := f.pin(t)
	backend, err := managed.NewBackend(
		managed.BackendConfig[metadata]{Adapter: f.adapter, HostBasis: "", MaxNodes: 50, MaxEdges: 100, Project: nil},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := backend.Retrieve(context.Background(), backendQuery(read))
	// Assert: graph reachability has no fabricated numeric similarity.
	if err != nil || result.Len() != 2 {
		t.Fatal("graph backend failed", err)
	}
	for _, doc := range result.Documents() {
		if doc.ScoreState != retrieval.ScoreAbsent || doc.Score != 0 || doc.ScoreSemantics != "" || doc.Rank <= 0 {
			t.Fatal("graph backend fabricated score")
		}
	}
	projected, err := managed.NewBackend(
		managed.BackendConfig[metadata]{
			Adapter:   f.adapter,
			HostBasis: "",
			MaxNodes:  50,
			MaxEdges:  100,
			Project: func(result managed.Result[metadata]) ([]managed.Projection[metadata], error) {
				f.epoch = 8
				return []managed.Projection[metadata]{
					{
						Document: retrieval.Document[metadata]{
							ID:      result.Snapshot.Nodes[0].ID,
							Content: "must not deliver",
							Meta:    result.Snapshot.Nodes[0].Meta,
						},
						Facts: []managed.FactIdentity{{Kind: managed.NodeFact, ID: result.Snapshot.Nodes[0].ID}},
					},
				}, nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err = projected.Retrieve(context.Background(), backendQuery(read))
	if !errors.Is(err, ragy.ErrUnavailable) || result.Len() != 0 {
		t.Fatal("revoked graph projection delivered", err)
	}
}

func TestGraphBackendConflictRequiresExplicitPolicy(t *testing.T) {
	// Arrange: canonical conflict, default projection has no authority to pick a winner.
	f := newFixture(t)
	first, second := payload("policy", "r1"), payload("faq", "r1")
	second.Nodes[0].Value.Content = "different"
	f.ingest(t, plan("policy", "", first, "p1"), first)
	f.ingest(t, plan("faq", "", second, "f1"), second)
	backend, err := managed.NewBackend(
		managed.BackendConfig[metadata]{Adapter: f.adapter, HostBasis: "", MaxNodes: 50, MaxEdges: 100, Project: nil},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act/Assert: empty conflicted graph is not reported as successful completion.
	result, err := backend.Retrieve(context.Background(), backendQuery(f.pin(t)))
	if !errors.Is(err, managed.ErrConflictingFacts) || result.Len() != 0 {
		t.Fatal("conflict silently became complete-empty", err)
	}
}

func TestGraphBackendOriginalSupportsSurviveSharedFactsAndProjectionMutation(t *testing.T) {
	// Arrange: equal canonical facts supported by two independent sources.
	f := newFixture(t)
	first, second := payload("policy", "r1"), payload("faq", "r1")
	f.ingest(t, plan("policy", "", first, "p1"), first)
	f.ingest(t, plan("faq", "", second, "f1"), second)
	backend, err := managed.NewBackend(managed.BackendConfig[metadata]{Adapter: f.adapter, MaxNodes: 50, MaxEdges: 100})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := backend.Retrieve(context.Background(), backendQuery(f.pin(t)))
	// Assert: supports belong to original chunks, not fabricated graph document IDs.
	if err != nil {
		t.Fatal(err)
	}
	for _, doc := range result.Documents() {
		refs := doc.SourceLocations()
		if len(refs) != 2 || refs[0].Reference.Revision != "r1" || refs[1].Reference.Revision != "r1" {
			t.Fatal(doc)
		}
		if refs[0].Reference.Artifact != "p1" || refs[1].Reference.Artifact != "f1" {
			t.Fatal("lost original chunks", refs)
		}
	}
	// Arrange: host can mutate its graph view, but not the captured support inventory.
	custom, err := managed.NewBackend(managed.BackendConfig[metadata]{Adapter: f.adapter, MaxNodes: 50, MaxEdges: 100,
		Project: func(view managed.Result[metadata]) ([]managed.Projection[metadata], error) {
			node := view.Snapshot.Nodes[0]
			view.Supports[0].References[0].Revision = "forged"
			return []managed.Projection[metadata]{
				{
					Document: retrieval.Document[metadata]{ID: "summary", Content: "summary", Meta: node.Meta},
					Facts:    []managed.FactIdentity{{Kind: managed.NodeFact, ID: node.ID}},
				},
			}, nil
		}})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	projected, err := custom.Retrieve(context.Background(), backendQuery(f.pin(t)))
	// Assert.
	if err != nil || projected.Len() != 1 {
		t.Fatal(err)
	}
	for _, location := range projected.Documents()[0].SourceLocations() {
		if location.Reference.Revision != "r1" {
			t.Fatal("projection mutated admitted inventory")
		}
	}
}

func TestGraphBackendRejectsUnadmittedProjectionFactsAndSources(t *testing.T) {
	for _, scenario := range []string{"unknown_fact", "foreign_source", "missing_facts", "private_metadata"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange: all source admission comes from an actual managed traversal.
			f := newFixture(t)
			input := payload("policy", "r1")
			f.ingest(t, plan("policy", "", input, "p1"), input)
			backend, err := managed.NewBackend(
				managed.BackendConfig[metadata]{Adapter: f.adapter, MaxNodes: 50, MaxEdges: 100,
					Project: func(view managed.Result[metadata]) ([]managed.Projection[metadata], error) {
						node := view.Snapshot.Nodes[0]
						projection := managed.Projection[metadata]{
							Document: retrieval.Document[metadata]{
								ID:      "summary",
								Content: "must not leak",
								Meta:    node.Meta,
							},
							Facts: []managed.FactIdentity{{Kind: managed.NodeFact, ID: node.ID}},
						}
						switch scenario {
						case "unknown_fact":
							projection.Facts[0].ID = "private"
						case "foreign_source":
							reference := view.Supports[0].References[0]
							reference.Revision = "other"
							projection.Document.SourceSupports = []source.Locator{
								{Kind: source.DocumentLocation, Reference: reference},
							}
						case "private_metadata":
							projection.Document.Meta.Tenant = "b"
						case "missing_facts":
							projection.Facts = nil
						}
						f.calls = nil
						return []managed.Projection[metadata]{projection}, nil
					}},
			)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			result, err := backend.Retrieve(context.Background(), backendQuery(f.pin(t)))
			// Assert: invalid projection fails before post-projection metadata materialization.
			if err == nil || result.Len() != 0 || len(f.calls) != 0 {
				t.Fatal(result, err, f.calls)
			}
		})
	}
}

func TestGraphBackendHostBasisDoesNotInventSourceCitations(t *testing.T) {
	// Arrange: host-owned graph facts have a basis, not managed source references.
	f := newFixture(t)
	if err := f.adapter.SetHostBasis(
		context.Background(),
		"foundation",
		hostSnapshot(payload("policy", "r1")),
	); err != nil {
		t.Fatal(err)
	}
	backend, err := managed.NewBackend(
		managed.BackendConfig[metadata]{Adapter: f.adapter, HostBasis: "foundation", MaxNodes: 50, MaxEdges: 100},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := backend.Retrieve(context.Background(), backendQuery(f.pin(t)))
	// Assert.
	if err != nil || result.Len() == 0 {
		t.Fatal(err)
	}
	for _, doc := range result.Documents() {
		if len(doc.SourceLocations()) != 0 || doc.SourceMapping.Text() != "" {
			t.Fatal("host basis turned into fake source", doc)
		}
	}
}

func TestGraphBackendCleanupReprojectsRemainingOriginalSupport(t *testing.T) {
	// Arrange: two sources support the same canonical graph.
	f := newFixture(t)
	first, second := payload("policy", "r1"), payload("faq", "r1")
	f.ingest(t, plan("policy", "", first, "p1"), first)
	f.ingest(t, plan("faq", "", second, "f1"), second)
	backend, err := managed.NewBackend(managed.BackendConfig[metadata]{Adapter: f.adapter, MaxNodes: 50, MaxEdges: 100})
	if err != nil {
		t.Fatal(err)
	}
	f.delete(t, "delete-policy", "policy", "policy")
	// Act: a new publication excludes the retired support.
	result, err := backend.Retrieve(context.Background(), backendQuery(f.pin(t)))
	// Assert: shared facts retain FAQ support and do not cite the deleted policy.
	if err != nil || result.Len() != 2 {
		t.Fatal(result, err)
	}
	for _, doc := range result.Documents() {
		refs := doc.SourceLocations()
		if len(refs) != 1 || refs[0].Reference.Source != "faq" || refs[0].Reference.Artifact != "f1" {
			t.Fatal("retired support leaked into projection", refs)
		}
	}
}
