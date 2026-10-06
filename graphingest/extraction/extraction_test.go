package extraction_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/source"
)

type attributes struct{ Tags []string }
type acl struct {
	Tenant string
	Secret []string
}
type fixture struct {
	now      time.Time
	epoch    int
	calls    int
	admitted int
	read     access.Binding
	ledger   *budget.Ledger
	config   extraction.Config[acl, string, string, attributes]
	input    []extraction.Snippet[acl]
	output   extraction.ModelOutput[string, string, attributes]
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{now: time.Now(), epoch: 7}
	schema := initBinding(t, f)
	var err error
	f.ledger, err = budget.New(
		budget.Config{
			Limits: budget.Limits{
				ModelCalls: 2,
				Usage:      budget.Usage{InputTokens: 2048, OutputTokens: 512, Cost: 100},
			},
			Deadline:         f.now.Add(5 * time.Second),
			Now:              func() time.Time { return f.now },
			RequireKnownCost: true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	for i, text := range []string{"Billing depends on LedgerDB.", "LedgerDB is a database."} {
		id := []string{"p1", "p2"}[i]
		loc := source.Locator{
			Kind: source.TextLocation,
			Reference: source.Reference{
				Namespace:         "n",
				Source:            "policy",
				Revision:          "r1",
				Transformation:    "original",
				AccessFingerprint: "acl",
				Artifact:          id,
				Representation:    "text",
			},
			Span: source.ByteSpan{Start: 0, End: len(text)},
		}
		mapping, mapErr := source.OriginalText(loc, text)
		if mapErr != nil {
			t.Fatal(mapErr)
		}
		f.input = append(
			f.input,
			extraction.Snippet[acl]{
				Namespace: "prod",
				Mapping:   mapping,
				Access:    acl{Tenant: "a", Secret: []string{"credential"}},
			},
		)
	}
	f.output = extraction.ModelOutput[string, string, attributes]{
		Entities: []extraction.Entity[string, attributes]{
			{
				ID:         "svc",
				Name:       "Billing",
				Kind:       "Service",
				Attributes: attributes{Tags: []string{"owned"}},
				Snippets:   []int{0},
			},
			{
				ID:         "db",
				Name:       "LedgerDB",
				Kind:       "Database",
				Attributes: attributes{Tags: []string{"owned"}},
				Snippets:   []int{1},
			},
		},
		Relations: []extraction.Relation[string, attributes]{
			{
				ID:         "dep",
				From:       "svc",
				To:         "db",
				Kind:       "depends_on",
				Attributes: attributes{Tags: []string{"owned"}},
				Snippets:   []int{0},
			},
		},
	}
	f.config = extraction.Config[acl, string, string, attributes]{
		OntologyIdentity: "service-ontology",
		Configuration:    "extractor-config",
		Schema:           schema,
		MaxSnippets:      20,
		MaxInputBytes:    4096,
		MaxEntities:      20,
		MaxRelations:     20,
		MaxSupports:      100,
		Duration:         5 * time.Second,
		Now:              func() time.Time { return f.now },
		CloneAccess:      func(a acl) (acl, error) { a.Secret = slices.Clone(a.Secret); return a, nil },
		Attributes:       func(a acl) (filter.RawAttributes, error) { return filter.RawAttributes{"tenant": a.Tenant}, nil },
		AdmitSnippet: func(_ context.Context, _ access.Binding, snippet extraction.Snippet[acl]) error {
			f.admitted++
			if snippet.Mapping.Supports()[0].Reference.Revision != "r1" {
				return ragy.ErrUnavailable
			}
			return nil
		},
		CloneAttributes: func(a attributes) (attributes, error) { a.Tags = slices.Clone(a.Tags); return a, nil },
		ValidateEntity: func(k string, _ attributes) error {
			if k != "Service" && k != "Database" {
				return ragy.ErrInvalidGraph
			}
			return nil
		},
		ValidateRelation: func(k, from, to string, _ attributes) error {
			if k != "depends_on" || from != "Service" || to != "Database" {
				return ragy.ErrInvalidGraph
			}
			return nil
		},
		Quote: func(context.Context) (budget.Reservation, error) {
			return budget.Reservation{
				Kind:      budget.Model,
				CostKnown: true,
				Usage:     budget.Usage{InputTokens: 100, OutputTokens: 50, Cost: 30},
			}, nil
		},
		CountInputTokens: func(input extraction.ModelInput) (uint64, error) {
			input.Snippets[0].Text = "counter-mutated"
			return 20, nil
		},
		Model: func(_ context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
			f.calls++
			assertModelInput(t, input)
			return f.output, extraction.Usage{
				Known: true,
				Value: budget.Usage{InputTokens: 20, OutputTokens: 10, Cost: 30},
			}, nil
		},
	}
	return f
}
func TestExtractionBindsSourceEvidenceAndReservesOneCall(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	adapter, err := extraction.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	output, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
	// Assert: native mention identity remains local, revision and namespace come from admitted snippets.
	if err != nil || f.calls != 1 || f.admitted != 2 || len(output.Extraction.Entities) != 2 ||
		len(output.Extraction.Relations) != 1 {
		t.Fatal(output, err)
	}
	entity := output.Extraction.Entities[0]
	if entity.Namespace != "prod" || entity.Supports[0].Reference.Revision != "r1" ||
		entity.Supports[0].Reference.Artifact != "p1" {
		t.Fatal(entity)
	}
	output.Extraction.Entities[0].Attributes.Tags[0] = "changed"
	if f.output.Entities[0].Attributes.Tags[0] != "owned" {
		t.Fatal("model output aliases extraction")
	}
	snapshot := f.ledger.Snapshot()
	if snapshot.Occupied.ModelCalls != 1 || snapshot.Actual.Cost != 30 || snapshot.Actual.InputTokens != 20 {
		t.Fatal(snapshot)
	}
}
func TestExtractionRejectsPrivateOrUnsupportedBeforeModel(t *testing.T) {
	for _, scenario := range []string{"private", "token_budget", "unknown_price", "bytes", "cancelled", "revoked_admission"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch scenario {
			case "private":
				f.input[1].Access.Tenant = "b"
			case "token_budget":
				f.config.CountInputTokens = func(extraction.ModelInput) (uint64, error) { return 101, nil }
			case "unknown_price":
				f.config.Quote = func(context.Context) (budget.Reservation, error) {
					return budget.Reservation{
						Kind:  budget.Model,
						Usage: budget.Usage{InputTokens: 100, OutputTokens: 50},
					}, nil
				}
			case "bytes":
				f.config.MaxInputBytes = 1
			case "cancelled":
				cancel()
			case "revoked_admission":
				f.config.AdmitSnippet = func(context.Context, access.Binding, extraction.Snippet[acl]) error { f.epoch++; return nil }
			}
			adapter, err := extraction.New(f.config)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			output, err := adapter.Extract(ctx, f.read, f.ledger, f.input)
			// Assert.
			if err == nil || f.calls != 0 || len(output.Extraction.Entities) != 0 {
				t.Fatal(output, f.calls, err)
			}
		})
	}
}
func TestExtractionMalformedOutputAndRevocationSuppressPayload(t *testing.T) {
	for _, scenario := range []string{"foreign_snippet", "endpoint", "kind", "overrun", "model_error", "revoked_model", "fake_deadline"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			original := f.config.Model
			switch scenario {
			case "foreign_snippet":
				f.output.Entities[0].Snippets = []int{2}
			case "endpoint":
				f.output.Relations[0].To = "unknown"
			case "kind":
				f.output.Entities[0].Kind = "invented"
			case "overrun":
				f.config.Model = func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
					out, usage, err := original(ctx, input)
					usage.Value.OutputTokens = 51
					return out, usage, err
				}
			case "model_error":
				f.config.Model = func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
					out, usage, _ := original(ctx, input)
					return out, usage, errors.New("model failed")
				}
			case "revoked_model":
				f.config.Model = func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
					out, usage, err := original(ctx, input)
					f.epoch++
					return out, usage, err
				}
			case "fake_deadline":
				f.config.Model = func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
					out, usage, err := original(ctx, input)
					f.now = f.now.Add(6 * time.Second)
					return out, usage, err
				}
			}
			adapter, err := extraction.New(f.config)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			output, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
			// Assert: call stays charged, no retries or partial result.
			if err == nil || f.calls != 1 || len(output.Extraction.Entities) != 0 ||
				f.ledger.Snapshot().Occupied.ModelCalls != 1 {
				t.Fatal(output, err)
			}
		})
	}
}

func initBinding(t *testing.T, f *fixture) filter.Schema {
	t.Helper()
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
	publication, err := access.PinPublication(
		"pub",
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
	f.read, err = access.Scoped(
		access.ScopedConfig{
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
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return schema
}

func assertModelInput(t *testing.T, input extraction.ModelInput) {
	t.Helper()
	data, _ := json.Marshal(input)
	if strings.Contains(string(data), "credential") || strings.Contains(string(data), "scope") ||
		strings.Contains(string(data), "counter-mutated") {
		t.Fatal("unexpected model input", string(data))
	}
}

func TestExtractionMixedNamespacesRemainAmbiguous(t *testing.T) {
	// Arrange: model merges mentions, but cannot choose a namespace for conflicting host contexts.
	f := newFixture(t)
	f.input[1].Namespace = "staging"
	f.output.Entities[0].Snippets = []int{0, 1}
	adapter, err := extraction.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
	// Assert.
	if err != nil || result.Extraction.Entities[0].Namespace != "" || len(result.Extraction.Entities[0].Supports) != 2 {
		t.Fatal(result, err)
	}
}

func TestExtractionAdvisoryUnknownCostStillRejectsTokenOverrun(t *testing.T) {
	// Arrange: unknown price cannot bypass known token caps.
	f := newFixture(t)
	ledger, err := budget.New(
		budget.Config{
			Limits: budget.Limits{
				ModelCalls: 1,
				Usage:      budget.Usage{InputTokens: 2048, OutputTokens: 512, Cost: 100},
			},
			Deadline:         f.now.Add(5 * time.Second),
			Now:              func() time.Time { return f.now },
			RequireKnownCost: false,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	f.config.Quote = func(context.Context) (budget.Reservation, error) {
		return budget.Reservation{Kind: budget.Model, Usage: budget.Usage{InputTokens: 100, OutputTokens: 50}}, nil
	}
	original := f.config.Model
	f.config.Model = func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
		out, usage, modelErr := original(ctx, input)
		usage.Value.OutputTokens = 51
		return out, usage, modelErr
	}
	adapter, err := extraction.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := adapter.Extract(context.Background(), f.read, ledger, f.input)
	// Assert: output suppressed; reservations retained conservatively.
	if !errors.Is(err, budget.ErrUsageExceeded) || len(result.Extraction.Entities) != 0 ||
		ledger.Snapshot().UnknownUsage != 1 ||
		ledger.Snapshot().Occupied.Usage.OutputTokens != 50 {
		t.Fatal(result, err, ledger.Snapshot())
	}
}
