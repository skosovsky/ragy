//go:build darwin || linux

package graphsummary_test

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
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/source"
)

type summaryGraphMeta struct {
	Tenant string `json:"tenant"`
}

type summaryLifecycle struct {
	store    lifecycle.Store
	graph    *managed.Adapter[summaryGraphMeta]
	executor *lifecycle.Executor[managed.Payload[summaryGraphMeta]]
}

// The host chooses live-source retention policy. A still readable old graph pin
// does not authorize a summary to bypass fresh original-source catalog admission.
type liveSourceCatalog struct {
	store lifecycle.Store
	host  *sourceHost
}

func (c liveSourceCatalog) Describe(ctx context.Context, req source.LookupRequest) ([]source.Descriptor[acl], error) {
	snapshot, err := c.store.Load(ctx, "n")
	if err != nil {
		return nil, err
	}
	for _, ref := range req.References {
		current := false
		for _, pub := range snapshot.Publications {
			if pub.Source != ref.Source {
				continue
			}
			for _, manifest := range snapshot.Manifests {
				if manifest.ID == pub.Manifest && !manifest.Tombstone && manifest.Identity.Revision == ref.Revision &&
					manifest.Identity.Access == ref.AccessFingerprint {
					current = true
				}
			}
		}
		if !current {
			return nil, nil
		}
	}
	return c.host.Describe(ctx, req)
}

func TestPublishedGraphSummaryInvalidatedBySourceTombstoneAndCleanup(t *testing.T) {
	for _, global := range []bool{false, true} {
		t.Run(map[bool]string{false: "community", true: "global"}[global], func(t *testing.T) {
			runSummaryLifecycleCase(t, global)
		})
	}
}

func newSummaryLifecycle(t *testing.T, f *fixture) summaryLifecycle {
	t.Helper()
	store, err := filestore.New(filepath.Join(t.TempDir(), "ledger"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	schema, err := graph.NewSchema(f.config.Schema, f.config.Schema)
	if err != nil {
		t.Fatal(err)
	}
	target, err := managed.New(
		managed.Config[summaryGraphMeta]{Namespace: "n", Target: "graph", Store: store, Schema: schema, MaxRecords: 10,
			CloneMeta: func(m summaryGraphMeta) (summaryGraphMeta, error) { return m, nil }},
	)
	if err != nil {
		t.Fatal(err)
	}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[managed.Payload[summaryGraphMeta]]{Store: store,
		Targets: []lifecycle.Registration[managed.Payload[summaryGraphMeta]]{{Name: "graph", Port: target}},
		ClonePayload: func(p managed.Payload[summaryGraphMeta]) (managed.Payload[summaryGraphMeta], error) {
			p.Nodes = slices.Clone(p.Nodes)
			p.Edges = slices.Clone(p.Edges)
			for i := range p.Nodes {
				p.Nodes[i].Value.Labels = slices.Clone(p.Nodes[i].Value.Labels)
			}
			return p, nil
		},
		ValidatePayload: func(_ lifecycle.Manifest, p managed.Payload[summaryGraphMeta]) error {
			if len(p.Nodes) != 1 || len(p.Edges) != 0 || p.Nodes[0].Value.Meta.Tenant != "a" {
				return ragy.ErrInvalidArgument
			}
			return nil
		}, Now: func() time.Time { return f.now }})
	if err != nil {
		t.Fatal(err)
	}
	return summaryLifecycle{store: store, graph: target, executor: executor}
}

func publishSummarySource(t *testing.T, life summaryLifecycle, original source.Reference, id string) {
	t.Helper()
	indexed := original
	indexed.Transformation = "index"
	indexed.Artifact = id
	indexed.Representation = "graph-node"
	manifest := lifecycle.Manifest{
		ID:      original.Source,
		Key:     original.Source,
		Payload: original.Source,
		State:   lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         original.Source,
			Revision:       "r1",
			Content:        original.Source,
			Transformation: "index",
			Access:         "acl",
		},
		Targets: []lifecycle.Target{
			{
				Name:      "graph",
				Required:  true,
				State:     lifecycle.TargetPending,
				Artifacts: []lifecycle.Artifact{{Reference: indexed, Supports: []source.Reference{original}}},
			},
		},
	}
	payload := managed.Payload[summaryGraphMeta]{
		Nodes: []managed.Node[summaryGraphMeta]{
			{
				Reference: indexed,
				Value: graph.Node[summaryGraphMeta]{
					ID:      id,
					Labels:  []string{"Community"},
					Content: id,
					Meta:    summaryGraphMeta{Tenant: "a"},
				},
			},
		},
	}
	ctx := context.Background()
	if _, err := life.executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	if _, err := life.executor.Stage(ctx, "n", manifest.ID, "graph", payload); err != nil {
		t.Fatal(err)
	}
	if _, err := life.executor.Publish(ctx, "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
}

func bindSummaryPublication(t *testing.T, f *fixture, store lifecycle.Store) access.Binding {
	t.Helper()
	ctx := context.Background()
	publication, err := lifecycle.CapturePublication(ctx, store, "n", []string{"graph"})
	if err != nil {
		t.Fatal(err)
	}
	targets := publication.Targets()
	// Retained source entries are host-owned. The live catalog additionally checks
	// the durable publication, so an old source entry cannot defeat a tombstone.
	for _, community := range f.request.Communities {
		ref := community.Snippets[0].Mapping.Supports()[0].Reference
		targets = append(
			targets,
			access.TargetRevision{
				Target:            "source",
				Namespace:         ref.Namespace,
				Source:            ref.Source,
				Revision:          ref.Revision,
				Transformation:    ref.Transformation,
				AccessFingerprint: ref.AccessFingerprint,
			},
		)
	}
	publication, err = access.PinPublication(publication.Reference(), targets)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := f.request.Read.Prepare(
		ctx,
		f.config.Schema,
		filter.Condition{},
		access.Capabilities{ScopeProfile: true, PinnedPublication: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.Scoped(
		access.ScopedConfig{
			Snapshot:    f.request.Read.Snapshot(),
			Mandatory:   mandatory,
			Schema:      f.config.Schema,
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

func liveSummaryReader(t *testing.T, f *fixture, store lifecycle.Store, host *sourceHost) *source.Reader[acl, string] {
	t.Helper()
	retained, err := source.NewReader(
		source.ReadConfig[acl, string]{
			Target:     "source",
			Schema:     f.config.Schema,
			Catalog:    liveSourceCatalog{store: store, host: host},
			Loader:     host,
			Attributes: func(a acl) (filter.RawAttributes, error) { return filter.RawAttributes{"tenant": a.Tenant}, nil },
			ValidatePayload: func(_ source.Reference, text string) error {
				if text == "" {
					return ragy.ErrUnavailable
				}
				return nil
			},
			ClonePayload: func(s string) (string, error) { return s, nil },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return retained
}

func summaryGraphRequest(read access.Binding, id string) managed.Request {
	return managed.Request{
		Read:      read,
		Traversal: graph.TraversalRequest{Seeds: []string{id}, Direction: graph.DirectionOutbound, Depth: 1},
		MaxNodes:  10,
		MaxEdges:  10,
	}
}

func tombstoneSummarySource(t *testing.T, life summaryLifecycle, src string) lifecycle.Manifest {
	t.Helper()
	m := lifecycle.Manifest{
		ID:                  "delete-" + src,
		Key:                 "delete-" + src,
		Payload:             "delete-" + src,
		ExpectedPublication: src,
		Tombstone:           true,
		State:               lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         src,
			Revision:       "deleted",
			Content:        "deleted",
			Transformation: "index",
			Access:         "acl",
		},
	}
	if _, err := life.executor.Prepare(context.Background(), m); err != nil {
		t.Fatal(err)
	}
	if _, err := life.executor.Publish(context.Background(), "n", m.ID); err != nil {
		t.Fatal(err)
	}
	return m
}

func cleanupSummarySource(
	t *testing.T,
	life summaryLifecycle,
	now time.Time,
	tombstone lifecycle.Manifest,
	retired string,
) {
	t.Helper()
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   life.store,
			Now:     func() time.Time { return now },
			Targets: []lifecycle.CleanupRegistration{{Name: "graph", Port: life.graph}},
			Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(context.Background(), "n", tombstone.ID); err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Attempt(context.Background(), "n", tombstone.ID, retired, "graph", false); err != nil {
		t.Fatal(err)
	}
}

func runSummaryLifecycleCase(t *testing.T, global bool) {
	t.Helper()
	// Arrange: actual durable graph publications and a fresh live-source catalog.
	f := newFixture(t)
	life := newSummaryLifecycle(t, f)
	host := &sourceHost{rows: make(map[source.Reference]string)}
	for _, community := range f.request.Communities {
		snippet := community.Snippets[0]
		host.rows[snippet.Mapping.Supports()[0].Reference] = snippet.Mapping.Text()
		publishSummarySource(t, life, snippet.Mapping.Supports()[0].Reference, community.ID)
	}
	f.request.Read = bindSummaryPublication(t, f, life.store)
	retained := liveSummaryReader(t, f, life.store, host)
	f.config.AdmitSource = func(ctx context.Context, read access.Binding, loc source.Locator) error {
		rows, err := retained.Lookup(
			ctx,
			source.LookupRequest{Read: read, References: []source.Reference{loc.Reference}},
		)
		if err != nil {
			return err
		}
		if len(rows) != 1 {
			return ragy.ErrUnavailable
		}
		return loc.Span.ValidateText(rows[0].Payload)
	}
	result, ledger, err := run(context.Background(), t, f, global)
	if err != nil {
		t.Fatal(result, err)
	}
	summary := result.Communities[0]
	if global {
		if result.Global == nil {
			t.Fatal(result)
		}
		summary = *result.Global
	}
	initial, err := summary.Resolve(context.Background(), f.request.Read, f.config.AdmitSource)
	if err != nil || initial.Text() == "" {
		t.Fatal(initial, err)
	}
	calls := f.calls
	// Act: durable tombstone first; source policy denies before payload loading.
	tombstone := tombstoneSummarySource(t, life, "s1")
	before := host.loads
	denied, denyErr := summary.Resolve(context.Background(), f.request.Read, f.config.AdmitSource)
	pinnedBefore, err := life.graph.FindByIDs(context.Background(), summaryGraphRequest(f.request.Read, "C1"))
	if err != nil {
		t.Fatal(err)
	}
	cleanupSummarySource(t, life, f.now, tombstone, "s1")
	pinnedAfter, pinErr := life.graph.FindByIDs(context.Background(), summaryGraphRequest(f.request.Read, "C1"))
	changedRead := bindSummaryPublication(t, f, life.store)
	changed, changedErr := summary.Resolve(context.Background(), changedRead, f.config.AdmitSource)
	assertSummaryLifecycle(
		t,
		global,
		denied,
		denyErr,
		host,
		before,
		pinnedBefore,
		pinnedAfter,
		pinErr,
		changed,
		changedErr,
		f,
		calls,
		ledger,
	)
}

func assertSummaryLifecycle(
	t *testing.T,
	global bool,
	denied source.MappedText,
	denyErr error,
	host *sourceHost,
	before int,
	pinnedBefore, pinnedAfter managed.Result[summaryGraphMeta],
	pinErr error,
	changed source.MappedText,
	changedErr error,
	f *fixture,
	calls int,
	ledger *budget.Ledger,
) {
	t.Helper()
	// Assert: neither old source availability nor a fresh binding resurrects text.
	expectedCalls := uint64(1)
	if global {
		expectedCalls = 3
	}
	if !access.IsProtectionFailure(denyErr) || denied.Text() != "" || host.loads != before ||
		len(
			pinnedBefore.Snapshot.Nodes,
		) != 1 || !errors.Is(pinErr, ragy.ErrUnavailable) || len(pinnedAfter.Snapshot.Nodes) != 0 ||
		!access.IsProtectionFailure(changedErr) || changed.Text() != "" || f.calls != calls ||
		ledger.Snapshot().Occupied.ModelCalls != expectedCalls || len(host.rows) != 2 {
		t.Fatal(
			denied,
			denyErr,
			host.loads,
			before,
			pinnedBefore,
			pinnedAfter,
			pinErr,
			changed,
			changedErr,
			f.calls,
			ledger.Snapshot(),
		)
	}
}
