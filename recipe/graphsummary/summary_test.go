package graphsummary_test

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
	"github.com/skosovsky/ragy/source"
)

type acl struct {
	Tenant string
	Secret []string
}
type fixture struct {
	now             time.Time
	epoch           int64
	deleted         bool
	calls           int
	sourceCalls     int
	membershipCalls int
	config          graphsummary.Config[acl]
	request         graphsummary.Request[acl]
	limits          budget.Config
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{now: time.Now(), epoch: 7}
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
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
	publication, err := access.PinPublication("pub1", []access.TargetRevision{
		{
			Target:            "graph",
			Namespace:         "n",
			Source:            "s1",
			Revision:          "r1",
			Transformation:    "index",
			AccessFingerprint: "acl",
		},
		{
			Target:            "graph",
			Namespace:         "n",
			Source:            "s2",
			Revision:          "r1",
			Transformation:    "index",
			AccessFingerprint: "acl",
		},
		{
			Target:            "source",
			Namespace:         "n",
			Source:            "s1",
			Revision:          "r1",
			Transformation:    "original",
			AccessFingerprint: "acl",
		},
		{
			Target:            "source",
			Namespace:         "n",
			Source:            "s2",
			Revision:          "r1",
			Transformation:    "original",
			AccessFingerprint: "acl",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "scope",
			PolicyEpoch: 7,
			IssuedAt:    f.now,
			ExpiresAt:   f.now.Add(30 * time.Second),
		},
		Schema:      schema,
		Mandatory:   mandatory,
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
	f.request = graphsummary.Request[acl]{
		Read:        read,
		Question:    "Какие зависимости у сервисов команд A и B?",
		Communities: communities(t),
	}
	f.config = graphsummary.Config[acl]{
		Schema:          schema,
		MaxMembers:      50,
		MaxSnippets:     20,
		MaxSupports:     100,
		MaxInputBytes:   16384,
		MaxSummaryBytes: 2048,
		Duration:        5 * time.Second,
		Now:             func() time.Time { return f.now },
		CloneAccess:     func(a acl) (acl, error) { a.Secret = slices.Clone(a.Secret); return a, nil },
		Attributes:      func(a acl) (filter.RawAttributes, error) { return filter.RawAttributes{"tenant": a.Tenant}, nil },
		Membership: func(_ context.Context, _ access.Binding, id string, members []string) error {
			f.membershipCalls++
			expected := map[string][]string{
				"C1": {"Billing", "LedgerDB", "TeamA"},
				"C2": {"Search", "IndexDB", "TeamB"},
			}
			if !slices.Equal(members, expected[id]) {
				return ragy.ErrUnavailable
			}
			members[0] = "mutated-host-copy"
			return nil
		},
		AdmitSource: f.admitSource,
		Quote: func(_ context.Context, stage graphsummary.Stage) (budget.Reservation, error) {
			usage := budget.Usage{InputTokens: 1024, OutputTokens: 256, Cost: 30}
			if stage == graphsummary.Reduce {
				usage.InputTokens = 2048
				usage.OutputTokens = 512
			}
			return budget.Reservation{Kind: budget.Model, Usage: usage, CostKnown: true}, nil
		},
		CountInputTokens: func(input graphsummary.ModelInput) (uint64, error) {
			input.Snippets[0].Text = "counter-mutated"
			return 32, nil
		},
		Model: func(_ context.Context, input graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
			f.calls++
			assertModelPrivacy(t, input)
			text := input.Snippets[0].Text
			selected := []int{0}
			if input.Stage == graphsummary.Reduce {
				text += " " + input.Snippets[1].Text
				selected = []int{0, 1}
			}
			input.Snippets[0].Text = "model-mutated"
			return graphsummary.ModelOutput{
				Text:     text,
				Selected: selected,
			}, graphsummary.Usage{
				Known: true,
				Value: budget.Usage{InputTokens: 32, OutputTokens: 16, Cost: 30},
			}, nil
		},
	}
	f.limits = budget.Config{
		Limits: budget.Limits{
			RetrievalCalls: 0,
			ModelCalls:     3,
			Usage:          budget.Usage{InputTokens: 4096, OutputTokens: 1024, Cost: 100},
		},
		Deadline:         f.now.Add(5 * time.Second),
		Now:              func() time.Time { return f.now },
		RequireKnownCost: true,
	}
	return f
}

func (f *fixture) admitSource(_ context.Context, _ access.Binding, location source.Locator) error {
	f.sourceCalls++
	if f.deleted || location.Reference.Revision != "r1" {
		return ragy.ErrUnavailable
	}
	return nil
}

func communities(t *testing.T) []graphsummary.Community[acl] {
	t.Helper()
	var result []graphsummary.Community[acl]
	for i, row := range []struct {
		id, text string
		members  []string
	}{
		{"C1", "Billing команды A зависит от LedgerDB.", []string{"Billing", "LedgerDB", "TeamA"}},
		{"C2", "Search команды B зависит от IndexDB.", []string{"Search", "IndexDB", "TeamB"}},
	} {
		location := source.Locator{
			Kind: source.TextLocation,
			Reference: source.Reference{
				Namespace:         "n",
				Source:            []string{"s1", "s2"}[i],
				Revision:          "r1",
				Transformation:    "original",
				AccessFingerprint: "acl",
				Artifact:          "original-p1",
				Representation:    "utf8",
			},
			Span: source.ByteSpan{Start: 0, End: len(row.text)},
		}
		mapping, err := source.OriginalText(location, row.text)
		if err != nil {
			t.Fatal(err)
		}
		result = append(
			result,
			graphsummary.Community[acl]{
				ID:      row.id,
				Members: slices.Clone(row.members),
				Snippets: []graphsummary.Snippet[acl]{
					{
						Mapping: mapping,
						Access:  acl{Tenant: "a", Secret: []string{"private-credential"}},
						Members: slices.Clone(row.members),
					},
				},
			},
		)
	}
	return result
}

func assertModelPrivacy(t *testing.T, input graphsummary.ModelInput) {
	t.Helper()
	encoded, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"private-credential", "access_fingerprint", "source_revision", "original-p1", "counter-mutated"} {
		if strings.Contains(string(encoded), value) {
			t.Fatal("private identity or mutated input reached model", value)
		}
	}
}

func run(ctx context.Context, t *testing.T, f *fixture, global bool) (graphsummary.Result, *budget.Ledger, error) {
	t.Helper()
	r, err := graphsummary.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	ledger, err := budget.New(f.limits)
	if err != nil {
		t.Fatal(err)
	}
	if global {
		result, runErr := r.Global(ctx, f.request, ledger)
		return result, ledger, runErr
	}
	request := f.request
	request.Communities = request.Communities[:1]
	result, err := r.Community(ctx, request, ledger)
	return result, ledger, err
}

func TestCommunityAndGlobalSummaryBoundedSupportAndOwnership(t *testing.T) {
	for _, global := range []bool{false, true} {
		t.Run(map[bool]string{false: "community", true: "global"}[global], func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			// Act.
			result, ledger, err := run(context.Background(), t, f, global)
			// Assert.
			wantCalls := uint64(1)
			wantCommunities := 1
			if global {
				wantCalls = 3
				wantCommunities = 2
			}
			if err != nil || result.Outcome != recipe.Complete || result.Stop != graphsummary.Summarized ||
				result.ModelCalls != wantCalls ||
				len(result.Communities) != wantCommunities ||
				ledger.Snapshot().Occupied.ModelCalls != wantCalls ||
				ledger.Snapshot().Actual.Cost != wantCalls*30 {
				t.Fatal(result, err, ledger.Snapshot())
			}
			assertResolved(t, f, result, global)
		})
	}
}

func assertResolved(t *testing.T, f *fixture, result graphsummary.Result, global bool) {
	t.Helper()
	summary := result.Communities[0]
	if global {
		if result.Global == nil {
			t.Fatal("missing global summary")
		}
		summary = *result.Global
	}
	mapping, err := summary.Resolve(context.Background(), f.request.Read, f.admitSource)
	if err != nil {
		t.Fatal(err)
	}
	if !summary.CoversMembership() || !strings.Contains(mapping.Text(), "LedgerDB") {
		t.Fatal(mapping.Text())
	}
	want := 1
	if global {
		want = 2
		if !strings.Contains(mapping.Text(), "IndexDB") {
			t.Fatal(mapping.Text())
		}
	}
	if len(summary.Supports()) != want || len(summary.CommunityIDs()) != want {
		t.Fatal(summary.Supports(), summary.CommunityIDs())
	}
	for _, fragment := range mapping.Fragments() {
		if fragment.Origin != source.DerivedContent || fragment.Precision != source.UnavailablePrecision {
			t.Fatal(fragment)
		}
	}
	supports := summary.Supports()
	supports[0].Reference.Revision = "mutated"
	ids := summary.CommunityIDs()
	ids[0] = "mutated"
	if summary.Supports()[0].Reference.Revision != "r1" || summary.CommunityIDs()[0] != "C1" ||
		f.request.Communities[0].Members[0] != "Billing" {
		t.Fatal("summary/membership aliases caller data")
	}
}
