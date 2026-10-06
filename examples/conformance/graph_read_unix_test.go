//go:build darwin || linux

package consumer_test

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type graphMetadata struct {
	Organization string `json:"organization"`
	Access       string `json:"access"`
	Identity     string `json:"identity"`
}

// observedGraphBackend forwards directly. It records context without adding any
// authorization, final delivery gate or payload filtering that could mask a defect.
type observedGraphBackend struct {
	target  *managed.Adapter[graphMetadata]
	next    *managed.Backend[graphMetadata]
	payload *payloadPort
}

func (b observedGraphBackend) Schema() filter.Schema { return b.next.Schema() }
func (b observedGraphBackend) ReadCapabilities() access.Capabilities {
	return b.next.ReadCapabilities()
}

func (b observedGraphBackend) AdmitRead(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ReadCoverage, error) {
	return b.next.AdmitRead(ctx, req)
}

func (b observedGraphBackend) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[graphMetadata], error) {
	b.payload.mu.Lock()
	b.payload.deadline, b.payload.hasDeadline = ctx.Deadline()
	b.payload.mu.Unlock()
	return b.next.Retrieve(ctx, req)
}

func graphReference(id, representation string) source.Reference {
	return source.Reference{
		Namespace:         "n",
		Source:            "graph-source",
		Revision:          "r1",
		Transformation:    "ingest",
		AccessFingerprint: "acl",
		Artifact:          id,
		Representation:    representation,
	}
}
func graphCorpus() managed.Payload[graphMetadata] {
	var nodes []managed.Node[graphMetadata]
	for _, row := range []struct{ id, organization, visibility string }{
		{publicID, "a", "public"}, {privateID, "a", "private"}, {foreignID, "b", "public"}, {"behind-private", "a", "public"},
	} {
		nodes = append(
			nodes,
			managed.Node[graphMetadata]{
				Reference: graphReference(row.id, "graph-node"),
				Value: graph.Node[graphMetadata]{
					ID:      row.id,
					Labels:  []string{"Service"},
					Content: row.id,
					Meta:    graphMetadata{Organization: row.organization, Access: row.visibility, Identity: row.id},
				},
			},
		)
	}
	var edges []managed.Edge[graphMetadata]
	for _, row := range []struct{ id, from, to string }{{"bridge-in", publicID, privateID}, {"bridge-out", privateID, "behind-private"}} {
		edges = append(
			edges,
			managed.Edge[graphMetadata]{
				Reference: graphReference(row.id, "graph-edge"),
				Value: graph.Edge[graphMetadata]{
					ID:       row.id,
					SourceID: row.from,
					TargetID: row.to,
					Type:     "depends_on",
					Meta:     graphMetadata{Organization: "a", Access: "public", Identity: row.id},
				},
			},
		)
	}
	return managed.Payload[graphMetadata]{Nodes: nodes, Edges: edges}
}
func graphManifest(input managed.Payload[graphMetadata]) lifecycle.Manifest {
	var artifacts []lifecycle.Artifact
	support := graphReference("original", "utf8")
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
		ID:      "graph-op",
		Key:     "graph-op",
		Payload: "graph-payload",
		State:   lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         "graph-source",
			Revision:       "r1",
			Content:        "content",
			Transformation: "ingest",
			Access:         "acl",
		},
		Targets: []lifecycle.Target{
			{Name: "graph", Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
		},
	}
}

func publishGraphFixture(
	ctx context.Context,
	t *testing.T,
	store lifecycle.Store,
	target *managed.Adapter[graphMetadata],
	clock *payloadPort,
) access.Publication {
	t.Helper()
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[managed.Payload[graphMetadata]]{
		Store: store, Targets: []lifecycle.Registration[managed.Payload[graphMetadata]]{{Name: "graph", Port: target}},
		ClonePayload: func(input managed.Payload[graphMetadata]) (managed.Payload[graphMetadata], error) {
			input.Nodes = slices.Clone(input.Nodes)
			input.Edges = slices.Clone(input.Edges)
			for i := range input.Nodes {
				input.Nodes[i].Value.Labels = slices.Clone(input.Nodes[i].Value.Labels)
			}
			return input, nil
		}, ValidatePayload: func(lifecycle.Manifest, managed.Payload[graphMetadata]) error { return nil }, Now: clock.now,
	})
	if err != nil {
		t.Fatal(err)
	}
	input := graphCorpus()
	manifest := graphManifest(input)
	if _, err = executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Stage(ctx, "n", manifest.ID, "graph", input); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(ctx, "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
	pin, err := lifecycle.CapturePublication(ctx, store, "n", []string{"graph"})
	if err != nil {
		t.Fatal(err)
	}
	return pin
}

func graphReadSchema(t *testing.T) filter.Schema {
	t.Helper()
	fields := filter.NewSchema()
	for _, name := range []string{"organization", "access", "identity"} {
		if _, err := fields.String(name); err != nil {
			t.Fatal(err)
		}
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	return schema
}

func newGraphReadFixture(
	t *testing.T,
) contracttest.ScopedReadFixture[struct{}, retrieval.NoRequestMeta, graphMetadata] {
	t.Helper()
	// Arrange: derive the public scope/negative predicates from the external BYOT fixture.
	base := newFixture(t)
	host := base.Backend.(*adapter)
	port := host.payload
	schema := graphReadSchema(t)
	store, err := filestore.New(filepath.Join(t.TempDir(), "manifests"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	target, err := managed.New(
		managed.Config[graphMetadata]{
			Namespace:  "n",
			Target:     "graph",
			Store:      store,
			Schema:     graph.Schema{NodeAttributes: schema, EdgeAttributes: schema},
			MaxRecords: 100,
			CloneMeta: func(meta graphMetadata) (graphMetadata, error) {
				port.mu.Lock()
				defer port.mu.Unlock()
				port.calls++
				port.ids = append(port.ids, meta.Identity)
				if port.revokeDuring {
					port.epoch = 8
				}
				return meta, nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	pin := publishGraphFixture(t.Context(), t, store, target, port)
	// Payload observation starts after real lifecycle capture/staging/publication.
	port.mu.Lock()
	port.calls = 0
	port.ids = nil
	port.mu.Unlock()
	mandatory, err := base.Request.Read.Prepare(
		t.Context(),
		host.Schema(),
		filter.Condition{},
		access.Capabilities{ScopeProfile: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	binding, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "graph-policy",
				PolicyEpoch: 7,
				IssuedAt:    port.now(),
				ExpiresAt:   port.now().Add(30 * time.Second),
			},
			Mandatory:   mandatory,
			Schema:      schema,
			Publication: pin,
			Now:         port.now,
			Authority:   access.AuthorityFunc(port.validate),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	backend, err := managed.NewBackend(
		managed.BackendConfig[graphMetadata]{Adapter: target, MaxNodes: 50, MaxEdges: 100},
	)
	if err != nil {
		t.Fatal(err)
	}
	return contracttest.ScopedReadFixture[struct{}, retrieval.NoRequestMeta, graphMetadata]{
		Backend: observedGraphBackend{target: target, next: backend, payload: port},
		Request: retrieval.Query[struct{}]{
			Read: binding,
			Options: retrieval.RetrieveOptions{
				TopK: 10,
				Graph: &retrieval.GraphOptions{
					Seeds:     []string{publicID, foreignID},
					Direction: graph.DirectionOutbound,
					Depth:     2,
				},
			},
		},
		Conflict:            base.Conflict,
		Unsupported:         base.Unsupported,
		ExpectedIDs:         []string{publicID},
		ForbiddenPayloadIDs: []string{privateID, foreignID, "behind-private", "bridge-in", "bridge-out"},
		IOCount:             port.ioCount,
		PayloadIDs:          port.payloadIDs,
		Revoke:              port.revoke,
		Expire:              port.expire,
		RevokeDuringIO:      port.revokeOnRead,
		ObservedDeadline:    port.observedDeadline,
	}
}
func TestExternalManagedGraphScopedReadConformance(t *testing.T) {
	// Act and Assert: public suite directly exercises the actual published leaf.
	contracttest.RunScopedReadSuite(t, newGraphReadFixture)
}

func TestExternalManagedGraphPlannerScope(t *testing.T) {
	for _, scenario := range []string{"empty", "conflicting", "unsupported"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange: fresh published graph and immutable mandatory scope.
			f := newGraphReadFixture(t)
			condition := filter.Condition{}
			if scenario == "conflicting" {
				condition = f.Conflict
			}
			if scenario == "unsupported" {
				condition = f.Unsupported
			}
			request := f.Request.WithPlan(retrieval.PlannedQuery[struct{}]{Filters: condition})
			// Act: admission and direct leaf execution, without a protecting wrapper.
			backend := f.Backend.(observedGraphBackend)
			_, admitErr := backend.AdmitRead(t.Context(), request)
			if f.IOCount() != 0 {
				t.Fatal("planner admission loaded graph payload")
			}
			result, err := backend.Retrieve(t.Context(), request)
			// Assert: a plan never replaces mandatory predicates or reaches hidden facts.
			assertGraphPlannerScope(t, f, scenario, result, admitErr, err)
		})
	}
}

func assertGraphPlannerScope(
	t *testing.T,
	f contracttest.ScopedReadFixture[struct{}, retrieval.NoRequestMeta, graphMetadata],
	scenario string,
	result retrieval.ResultSet[graphMetadata],
	admitErr, err error,
) {
	t.Helper()
	if scenario == "unsupported" {
		if !access.IsUnsupportedCapability(admitErr) || !access.IsProtectionFailure(err) ||
			!errors.Is(err, ragy.ErrUnsupported) ||
			f.IOCount() != 0 ||
			!result.IsEmpty() {
			t.Fatal("unsupported graph plan dispatched", admitErr, err)
		}
		return
	}
	if admitErr != nil || err != nil {
		t.Fatal(admitErr, err)
	}
	want := []string{publicID}
	if scenario == "conflicting" {
		want = nil
	}
	var actual []string
	for _, doc := range result.Documents() {
		actual = append(actual, doc.ID)
	}
	if !slices.Equal(actual, want) || containsForbiddenGraphPayload(f.PayloadIDs()) {
		t.Fatal("graph plan widened output or traversal", actual)
	}
}
func containsForbiddenGraphPayload(ids []string) bool {
	return slices.Contains(ids, privateID) || slices.Contains(ids, "behind-private") || slices.Contains(ids, foreignID)
}

func TestExternalManagedGraphFindByIDsScope(t *testing.T) {
	for _, scenario := range []string{"allowed", "conflicting", "unsupported", "missing-binding", "revoked", "expired", "canceled", "revoked-during-clone"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange: actual published graph, including a node unreachable through a private bridge.
			f := newGraphReadFixture(t)
			request := managed.Request{
				Read: f.Request.Read,
				Traversal: graph.TraversalRequest{
					Seeds:     []string{publicID, privateID, foreignID, "behind-private", "nonexistent", publicID},
					Direction: graph.DirectionOutbound,
					Depth:     2,
				},
				MaxNodes: 50,
				MaxEdges: 100,
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			expected := configureGraphLookup(f, &request, scenario, cancel)
			// Act: direct graph lookup; no retrieval Backend projection or wrapper gates.
			result, err := f.Backend.(observedGraphBackend).target.FindByIDs(ctx, request)
			// Assert: explicit lookup can select the public tail, but never private/foreign payload.
			assertGraphLookup(t, f, scenario, result, err, expected)
		})
	}
}

func configureGraphLookup(
	f contracttest.ScopedReadFixture[struct{}, retrieval.NoRequestMeta, graphMetadata],
	request *managed.Request,
	scenario string,
	cancel context.CancelFunc,
) error {
	switch scenario {
	case "conflicting":
		request.Traversal.NodeFilter = f.Conflict
	case "unsupported":
		request.Traversal.NodeFilter = f.Unsupported
		return ragy.ErrUnsupported
	case "missing-binding":
		request.Read = access.Binding{}
		return ragy.ErrInvalidArgument
	case "revoked":
		f.Revoke()
		return ragy.ErrUnavailable
	case "expired":
		f.Expire()
		return ragy.ErrUnavailable
	case "canceled":
		cancel()
		return context.Canceled
	case "revoked-during-clone":
		f.RevokeDuringIO()
		return ragy.ErrUnavailable
	}
	return nil
}

func assertGraphLookup(
	t *testing.T,
	f contracttest.ScopedReadFixture[struct{}, retrieval.NoRequestMeta, graphMetadata],
	scenario string,
	result managed.Result[graphMetadata],
	err, expected error,
) {
	t.Helper()
	if expected == nil {
		assertAllowedGraphLookup(t, f, scenario, result, err)
		return
	}
	if !errors.Is(err, expected) || !access.IsProtectionFailure(err) || len(result.Snapshot.Nodes) != 0 ||
		len(result.Snapshot.Edges) != 0 ||
		len(result.Supports) != 0 ||
		len(result.Conflicts) != 0 {
		t.Fatal("lookup failed open", err, result)
	}
	if scenario == "revoked-during-clone" && f.IOCount() == 0 {
		t.Fatal("missing mid-clone injection")
	}
	if scenario != "revoked-during-clone" && f.IOCount() != 0 {
		t.Fatal("denied lookup materialized payload")
	}
}

func assertAllowedGraphLookup(
	t *testing.T,
	f contracttest.ScopedReadFixture[struct{}, retrieval.NoRequestMeta, graphMetadata],
	scenario string,
	result managed.Result[graphMetadata],
	err error,
) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, node := range result.Snapshot.Nodes {
		ids = append(ids, node.ID)
	}
	want := []string{publicID, "behind-private"}
	if scenario == "conflicting" {
		want = nil
	}
	if !slices.Equal(ids, want) || len(result.Snapshot.Edges) != 0 || len(result.Supports) != len(want) ||
		slices.Contains(f.PayloadIDs(), privateID) ||
		slices.Contains(f.PayloadIDs(), foreignID) {
		t.Fatal("lookup payload/identity mismatch", ids, result.Supports)
	}
	for _, support := range result.Supports {
		if support.Kind != "node" || len(support.References) != 1 ||
			support.References[0] != graphReference("original", "utf8") {
			t.Fatal("lookup lost original source support", support)
		}
	}
}
