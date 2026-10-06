package structured_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/adapters/openai/structured"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/source"
)

func TestCoreExtractionHTTPAdmissionAndLedger(t *testing.T) {
	for _, tenant := range []string{"a", "private", "truncated"} {
		t.Run(tenant, func(t *testing.T) {
			// Arrange: real HTTP transport, mandatory scope and retained original mapping.
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				for _, secret := range []string{"private-source", "access-secret", "scope-identity"} {
					if strings.Contains(string(body), secret) {
						t.Error("source/access identity escaped model input")
					}
				}
				_, _ = io.WriteString(
					w,
					envelope(
						`{"entities":[{"id":"e1","name":"Billing","kind":"Service","attributes":{"value":9007199254740993},"snippets":[0]}],"relations":[]}`,
						finishReason(tenant),
					),
				)
			}))
			t.Cleanup(server.Close)
			core, ledger, read, mapping, location := extractionFixture(t, server.URL)
			acl := tenant
			if tenant == "truncated" {
				acl = "a"
			}
			// Act.
			out, err := core.Extract(
				context.Background(),
				read,
				ledger,
				[]extraction.Snippet[string]{{Namespace: "prod", Mapping: mapping, Access: acl}},
			)
			assertExtraction(t, tenant, out, err, calls.Load(), ledger, location)
		})
	}
}

func scopedRead(t *testing.T) (access.Binding, filter.Schema) {
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
	condition, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "scope-identity",
			PolicyEpoch: 7,
			IssuedAt:    now,
			ExpiresAt:   now.Add(time.Minute),
		},
		Mandatory:   condition,
		Schema:      schema,
		Publication: access.CurrentPublication(),
		Now:         time.Now,
		Authority:   access.AuthorityFunc(func(context.Context, access.Snapshot) error { return nil }),
	})
	if err != nil {
		t.Fatal(err)
	}
	return read, schema
}

func coreConfig(
	schema filter.Schema,
	client *structured.Extractor[string, string, payload],
) extraction.Config[string, string, string, payload] {
	return extraction.Config[string, string, string, payload]{
		OntologyIdentity: "host-ontology",
		Configuration:    "host-extraction",
		Schema:           schema,
		MaxSnippets:      2,
		MaxInputBytes:    1024,
		MaxEntities:      2,
		MaxRelations:     2,
		MaxSupports:      4,
		Duration:         time.Second,
		Now:              time.Now,
		CloneAccess:      func(a string) (string, error) { return a, nil },
		Attributes:       func(a string) (filter.RawAttributes, error) { return filter.RawAttributes{"tenant": a}, nil },
		AdmitSnippet: func(_ context.Context, _ access.Binding, s extraction.Snippet[string]) error {
			if s.Mapping.Supports()[0].Reference.Revision != "r1" {
				return ragy.ErrUnavailable
			}
			return nil
		},
		CloneAttributes: func(a payload) (payload, error) { return a, nil },
		ValidateEntity: func(kind string, _ payload) error {
			if kind != "Service" {
				return ragy.ErrInvalidGraph
			}
			return nil
		},
		ValidateRelation: func(string, string, string, payload) error { return ragy.ErrInvalidGraph },
		Quote: func(context.Context) (budget.Reservation, error) {
			return budget.Reservation{
				Kind: budget.Model, Usage: budget.Usage{InputTokens: 100, OutputTokens: 50, Cost: 100}, CostKnown: true,
			}, nil
		},
		CountInputTokens: client.CountInputTokens,
		Model:            client.Model,
	}
}

func extractionSchema() json.RawMessage {
	return json.RawMessage(
		`{"type":"object","properties":{"entities":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"name":{"type":"string"},"kind":{"type":"string","enum":["Service"]},"attributes":{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false},"snippets":{"type":"array","items":{"type":"integer"}}},"required":["id","name","kind","attributes","snippets"],"additionalProperties":false}},"relations":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"from":{"type":"string"},"to":{"type":"string"},"kind":{"type":"string"},"attributes":{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false},"snippets":{"type":"array","items":{"type":"integer"}}},"required":["id","from","to","kind","attributes","snippets"],"additionalProperties":false}}},"required":["entities","relations"],"additionalProperties":false}`,
	)
}

func validateExtractionFixture(raw json.RawMessage) error {
	var output extraction.ModelOutput[string, string, payload]
	if err := json.Unmarshal(raw, &output); err != nil {
		return err
	}
	if len(output.Entities) != 1 || len(output.Relations) != 0 {
		return ragy.ErrProtocol
	}
	entity := output.Entities[0]
	if entity.ID == "" || entity.Name == "" || entity.Kind != "Service" || len(entity.Snippets) != 1 ||
		entity.Snippets[0] != 0 {
		return ragy.ErrProtocol
	}
	_, err := entity.Attributes.Value.Int64()
	return err
}

func extractionFixture(
	t *testing.T,
	endpoint string,
) (*extraction.Adapter[string, string, string, payload], *budget.Ledger, access.Binding, source.MappedText, source.Locator) {
	t.Helper()
	read, schema := scopedRead(t)
	cfg := config(endpoint)
	cfg.Schema = extractionSchema()
	cfg.Validate = validateExtractionFixture
	client, err := structured.NewExtractor[string, string, payload](
		cfg,
		func(u structured.Usage) (uint64, error) { return u.InputTokens + u.OutputTokens, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	core, err := extraction.New(coreConfig(schema, client))
	if err != nil {
		t.Fatal(err)
	}
	ledger, err := budget.New(budget.Config{
		Limits: budget.Limits{
			RetrievalCalls: 0,
			ModelCalls:     1,
			Usage:          budget.Usage{InputTokens: 100, OutputTokens: 50, Cost: 100},
		},
		Deadline:         time.Now().Add(time.Second),
		Now:              time.Now,
		RequireKnownCost: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	text := "Billing is a service."
	location := source.Locator{Kind: source.TextLocation, Reference: source.Reference{
		Namespace:         "n",
		Source:            "private-source",
		Revision:          "r1",
		Transformation:    "original",
		AccessFingerprint: "access-secret",
		Artifact:          "p1",
		Representation:    "text",
	}, Span: source.ByteSpan{Start: 0, End: len(text)}}
	mapping, err := source.OriginalText(location, text)
	if err != nil {
		t.Fatal(err)
	}
	return core, ledger, read, mapping, location
}

func assertExtraction(
	t *testing.T,
	tenant string,
	out extraction.Result[string, string, payload],
	err error,
	calls int32,
	ledger *budget.Ledger,
	location source.Locator,
) {
	t.Helper()
	// Assert.
	if tenant == "truncated" {
		if err == nil || calls != 1 || len(out.Extraction.Entities) != 0 || ledger.Snapshot().Actual.Cost != 45 ||
			ledger.Snapshot().Occupied.ModelCalls != 1 {
			t.Fatal(out, err, calls, ledger.Snapshot())
		}
		return
	}
	if tenant != "a" {
		if err == nil || calls != 0 || len(out.Extraction.Entities) != 0 ||
			ledger.Snapshot().Occupied.ModelCalls != 0 {
			t.Fatal(out, err, calls, ledger.Snapshot())
		}
		return
	}
	if err != nil || calls != 1 || len(out.Extraction.Entities) != 1 {
		t.Fatal(out, err, calls)
	}
	entity := out.Extraction.Entities[0]
	if entity.Namespace != "prod" || len(entity.Supports) != 1 || entity.Supports[0] != location ||
		entity.Attributes.Value.String() != "9007199254740993" ||
		ledger.Snapshot().Actual.Cost != 45 {
		t.Fatal(entity, ledger.Snapshot())
	}
}

func finishReason(scenario string) string {
	if scenario == "truncated" {
		return "length"
	}
	return "stop"
}
