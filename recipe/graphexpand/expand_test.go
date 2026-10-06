package graphexpand_test

import (
	"context"
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
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphexpand"
)

type metadata struct {
	Tenant string `json:"tenant"`
	Name   string `json:"name"`
}
type countedStore struct {
	lifecycle.Store

	loads int
}

func (s *countedStore) Load(ctx context.Context, namespace string) (lifecycle.Snapshot, error) {
	s.loads++
	return s.Store.Load(ctx, namespace)
}

type fixture struct {
	now          time.Time
	epoch        int64
	store        *countedStore
	config       graphexpand.Config[metadata]
	request      graphexpand.Request
	ledgerConfig budget.Config
	cloned       []string
	quotes       int
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{now: time.Now(), epoch: 7}
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = fields.String("name"); err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	publication, err := access.PinPublication("host-publication", nil)
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "scope",
			PolicyEpoch: 7,
			IssuedAt:    f.now,
			ExpiresAt:   f.now.Add(time.Minute),
		},
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
	})
	if err != nil {
		t.Fatal(err)
	}
	base, err := filestore.New(filepath.Join(t.TempDir(), "manifests"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	f.store = &countedStore{Store: base, loads: 0}
	adapter, err := managed.New(managed.Config[metadata]{
		Namespace: "n",
		Target:    "graph",
		Store:     f.store,
		Schema: graph.Schema{
			NodeAttributes: schema,
			EdgeAttributes: schema,
		},
		NodeCodec:  nil,
		EdgeCodec:  nil,
		MaxRecords: 100, MaxAdmissionRecords: 100,
		CloneMeta: func(m metadata) (metadata, error) { f.cloned = append(f.cloned, m.Name); return m, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if err = adapter.SetHostBasis(context.Background(), "foundation", snapshot()); err != nil {
		t.Fatal(err)
	}
	f.cloned = nil
	f.config = graphexpand.Config[metadata]{
		Adapter:   adapter,
		MaxDepth:  2,
		MaxNodes:  50,
		MaxEdges:  100,
		Duration:  5 * time.Second,
		Now:       func() time.Time { return f.now },
		CloneMeta: func(m metadata) (metadata, error) { return m, nil },
		Quote: func(context.Context) (budget.Reservation, error) {
			f.quotes++
			return budget.Reservation{
				Kind:      budget.Retrieval,
				Usage:     budget.Usage{InputTokens: 0, OutputTokens: 0, Cost: 10},
				CostKnown: true,
			}, nil
		},
	}
	f.request = graphexpand.Request{Read: read, HostBasis: "foundation", Traversal: graph.TraversalRequest{
		Seeds: []string{
			"team-a",
		},
		Direction:  graph.DirectionUndirected,
		Depth:      2,
		NodeFilter: filter.Condition{},
		EdgeFilter: filter.Condition{},
		Page:       nil,
	}}
	f.ledgerConfig = budget.Config{
		Limits: budget.Limits{
			RetrievalCalls: 4,
			ModelCalls:     0,
			Usage:          budget.Usage{InputTokens: 4096, OutputTokens: 1024, Cost: 100},
		},
		Deadline:         f.now.Add(5 * time.Second),
		Now:              func() time.Time { return f.now },
		RequireKnownCost: true,
	}
	return f
}

func snapshot() graph.Snapshot[metadata] {
	var nodes []graph.Node[metadata]
	for _, row := range []struct{ id, label, tenant string }{{"team-a", "Team", "a"}, {"billing", "Service", "a"}, {"ledger-db", "Database", "a"}, {"private", "Service", "b"}} {
		nodes = append(
			nodes,
			graph.Node[metadata]{
				ID:      row.id,
				Labels:  []string{row.label},
				Content: row.id,
				Meta:    metadata{Tenant: row.tenant, Name: row.id},
			},
		)
	}
	var edges []graph.Edge[metadata]
	for _, row := range []struct{ id, from, to, kind string }{{"owner", "billing", "team-a", "owned_by"}, {"dependency", "billing", "ledger-db", "depends_on"}, {"cycle", "billing", "billing", "depends_on"}, {"private-edge", "team-a", "private", "depends_on"}} {
		edges = append(
			edges,
			graph.Edge[metadata]{
				ID:       row.id,
				SourceID: row.from,
				TargetID: row.to,
				Type:     row.kind,
				Meta:     metadata{Tenant: "a", Name: row.id},
			},
		)
	}
	return graph.Snapshot[metadata]{Nodes: nodes, Edges: edges}
}

func run(ctx context.Context, t *testing.T, f *fixture) (graphexpand.Result[metadata], *budget.Ledger, error) {
	t.Helper()
	r, err := graphexpand.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	ledger, err := budget.New(f.ledgerConfig)
	if err != nil {
		t.Fatal(err)
	}
	result, err := r.Run(ctx, f.request, ledger)
	return result, ledger, err
}

func TestLocalExpansionPathCyclesScopeAndModelFreeBudget(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	// Act.
	result, ledger, err := run(context.Background(), t, f)
	// Assert: one target call expands owner→service→DB, with no private node or model call.
	var ids []string
	for _, node := range result.Evidence.Snapshot.Nodes {
		ids = append(ids, node.ID)
	}
	if err != nil || result.Outcome != recipe.Complete || result.Stop != graphexpand.Expanded ||
		result.GraphCalls != 1 ||
		f.store.loads != 1 ||
		!slices.Equal(ids, []string{"billing", "ledger-db", "team-a"}) ||
		len(result.Evidence.Snapshot.Edges) != 3 ||
		slices.Contains(f.cloned, "private") ||
		slices.Contains(f.cloned, "private-edge") {
		t.Fatal(result, err, f.store.loads, f.cloned)
	}
	if ledger.Snapshot().Occupied.RetrievalCalls != 1 || ledger.Snapshot().Occupied.ModelCalls != 0 ||
		ledger.Snapshot().Actual.Cost != 10 ||
		len(result.SourceReferences()) != 0 ||
		len(result.Evidence.Supports) == 0 {
		t.Fatal(result, ledger.Snapshot())
	}
	// Explicit host foundations are not converted into fabricated original source citations.
	for _, support := range result.Evidence.Supports {
		if !slices.Equal(support.HostBases, []string{"foundation"}) {
			t.Fatal(support)
		}
	}
}

func TestLocalExpansionStopsBudgetAndUnknownPriceBeforeTargetIO(t *testing.T) {
	for _, scenario := range []string{"budget", "unknown-price"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			stop := graphexpand.BudgetExhausted
			if scenario == "budget" {
				f.ledgerConfig.Limits.RetrievalCalls = 0
			} else {
				stop = graphexpand.PriceUnavailable
				f.config.Quote = func(context.Context) (budget.Reservation, error) {
					return budget.Reservation{
						Kind:      budget.Retrieval,
						Usage:     budget.Usage{InputTokens: 0, OutputTokens: 0, Cost: 0},
						CostKnown: false,
					}, nil
				}
			}
			// Act.
			result, ledger, err := run(context.Background(), t, f)
			// Assert.
			if err != nil || result.Outcome != recipe.Insufficient || result.Stop != stop || result.GraphCalls != 0 ||
				f.store.loads != 0 ||
				len(f.cloned) != 0 ||
				ledger.Snapshot().Occupied.RetrievalCalls != 0 {
				t.Fatal(result, err, f.store.loads, ledger.Snapshot())
			}
		})
	}
}

func TestLocalExpansionAdmissionAndRevocationSuppressOutput(t *testing.T) {
	for _, scenario := range []string{"depth", "canceled", "pricing-revokes", "snapshot-revokes", "nodes", "edges", "fake-deadline"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			configureFailure(f, scenario, cancel)
			// Act.
			result, _, err := run(ctx, t, f)
			// Assert.
			if err == nil || len(result.Evidence.Snapshot.Nodes) != 0 || result.GraphCalls != 0 {
				t.Fatal(result, err)
			}
			if (scenario == "depth" || scenario == "canceled" || scenario == "pricing-revokes") && f.store.loads != 0 {
				t.Fatal("failure reached target", f.store.loads)
			}
			if scenario == "depth" && f.quotes != 0 {
				t.Fatal("invalid topology reached pricing")
			}
		})
	}
}

func configureFailure(f *fixture, scenario string, cancel context.CancelFunc) {
	switch scenario {
	case "depth":
		f.request.Traversal.Depth = 3
	case "canceled":
		cancel()
	case "pricing-revokes":
		f.config.Quote = func(context.Context) (budget.Reservation, error) {
			f.epoch++
			return budget.Reservation{
				Kind:      budget.Retrieval,
				Usage:     budget.Usage{InputTokens: 0, OutputTokens: 0, Cost: 10},
				CostKnown: true,
			}, nil
		}
	case "snapshot-revokes":
		f.config.CloneMeta = func(m metadata) (metadata, error) { f.epoch++; return m, nil }
	case "nodes":
		f.config.MaxNodes = 1
	case "edges":
		f.config.MaxEdges = 1
	case "fake-deadline":
		f.config.CloneMeta = func(m metadata) (metadata, error) { f.now = f.now.Add(6 * time.Second); return m, nil }
	}
}
