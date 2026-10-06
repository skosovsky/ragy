package structured_test

import (
	"context"
	"encoding/json"
	"errors"
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
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

func recipeConfig(endpoint string, assess bool) structured.Config {
	cfg := config(endpoint)
	if assess {
		cfg.Schema = json.RawMessage(
			`{"type":"object","properties":{"selected":{"type":"array","items":{"type":"integer"}},"sufficient":{"type":"boolean"}},"required":["selected","sufficient"],"additionalProperties":false}`,
		)
	} else {
		cfg.Schema = json.RawMessage(
			`{"type":"object","properties":{"queries":{"type":"array","items":{"type":"string"}}},"required":["queries"],"additionalProperties":false}`,
		)
	}
	cfg.Validate = func(raw json.RawMessage) error {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return err
		}
		if assess {
			if len(fields) != 2 || fields["selected"] == nil || fields["sufficient"] == nil {
				return ragy.ErrProtocol
			}
			var out structured.AssessorOutput
			return json.Unmarshal(raw, &out)
		}
		if len(fields) != 1 || fields["queries"] == nil {
			return ragy.ErrProtocol
		}
		var out structured.PlannerOutput
		return json.Unmarshal(raw, &out)
	}
	return cfg
}

func recipePrice(usage structured.Usage) (uint64, error) {
	return usage.InputTokens + usage.OutputTokens, nil
}

func TestRecipeHTTPBindingsLimitsAndPrivacy(t *testing.T) {
	for _, strategy := range []recipe.Strategy{recipe.SingleRewrite, recipe.MultiQuery, recipe.Decomposition} {
		t.Run(string(strategy), func(t *testing.T) {
			// Arrange.
			var calls atomic.Int32
			server := httptest.NewServer(recipeHandler(t, strategy, &calls))
			t.Cleanup(server.Close)
			planner, err := structured.NewPlanner[string, string](
				recipeConfig(server.URL, false),
				strategy,
				100,
				recipePrice,
			)
			if err != nil {
				t.Fatal(err)
			}
			assessor, err := structured.NewAssessor[string, string, string](
				recipeConfig(server.URL, true),
				3,
				10,
				1000,
				recipePrice,
			)
			if err != nil {
				t.Fatal(err)
			}
			req := retrieval.Request[string, string]{
				Read:   access.Unrestricted(),
				Text:   "original",
				Intent: "private-intent",
				Meta:   "private-request",
			}
			input := recipe.AssessmentInput[string, string, string]{
				Original: req,
				Queries: []recipe.QueryEvidence[string]{
					{
						Index: 7,
						Text:  "derived",
						Keys:  []string{"private-key"},
						Documents: []retrieval.Document[string]{
							{ID: "private-id", Content: "allowed snippet", Meta: "private-meta"},
						},
					},
				},
			}
			limits := recipe.ModelLimits{InputTokens: 40, OutputTokens: 17}
			// Act.
			planning, planErr := planner.Plan(context.Background(), req, limits)
			assessment, assessErr := assessor.Assess(context.Background(), input, limits)
			// Assert.
			if planErr != nil || assessErr != nil || calls.Load() != 2 || len(planning.Queries) != 1 ||
				!assessment.Sufficient ||
				assessment.Selected[0] != 7 ||
				!planning.Usage.Known ||
				planning.Usage.Value.Cost != 45 {
				t.Fatal("HTTP recipe binding failed", planErr, assessErr)
			}
		})
	}
}

func TestRecipeHTTPInvalidOutputsRetainUsage(t *testing.T) {
	for _, payload := range []string{`{"queries":["a","b"]}`, `{"queries":[""]}`, `{"queries":["a","a"]}`, `{"queries":["a"],"private":true}`} {
		t.Run(payload, func(t *testing.T) {
			// Arrange.
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				_, _ = io.WriteString(w, envelope(payload, "stop"))
			}))
			t.Cleanup(server.Close)
			planner, err := structured.NewPlanner[string, string](
				recipeConfig(server.URL, false),
				recipe.SingleRewrite,
				100,
				recipePrice,
			)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			output, err := planner.Plan(
				context.Background(),
				retrieval.Request[string, string]{Read: access.Unrestricted(), Text: "original"},
				recipe.ModelLimits{InputTokens: 40, OutputTokens: 17},
			)
			// Assert.
			if !errors.Is(err, ragy.ErrProtocol) || len(output.Queries) != 0 || !output.Usage.Known ||
				calls.Load() != 1 {
				t.Fatal("invalid output erased usage or retried", err)
			}
		})
	}
}

func TestRecipeHTTPPreflightAndForeignSelection(t *testing.T) {
	// Arrange.
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, envelope(`{"selected":[99],"sufficient":true}`, "stop"))
	}))
	t.Cleanup(server.Close)
	assessor, err := structured.NewAssessor[string, string, string](
		recipeConfig(server.URL, true),
		3,
		1,
		100,
		recipePrice,
	)
	if err != nil {
		t.Fatal(err)
	}
	input := recipe.AssessmentInput[string, string, string]{
		Original: retrieval.Request[string, string]{Read: access.Unrestricted(), Text: "original"},
		Queries: []recipe.QueryEvidence[string]{
			{Index: 0, Text: "q", Documents: []retrieval.Document[string]{{Content: "snippet"}}},
		},
	}
	// Act.
	low, lowErr := assessor.Assess(context.Background(), input, recipe.ModelLimits{InputTokens: 39, OutputTokens: 17})
	denied := input
	denied.Original.Read = access.Binding{}
	_, deniedErr := assessor.Assess(context.Background(), denied, recipe.ModelLimits{InputTokens: 40, OutputTokens: 17})
	output, outputErr := assessor.Assess(
		context.Background(),
		input,
		recipe.ModelLimits{InputTokens: 40, OutputTokens: 17},
	)
	// Assert.
	if !errors.Is(lowErr, budget.ErrExhausted) || low.Usage.Known || deniedErr == nil ||
		!errors.Is(outputErr, ragy.ErrProtocol) ||
		!output.Usage.Known ||
		len(output.Selected) != 0 ||
		calls.Load() != 1 {
		t.Fatal("preflight/foreign selection admitted invalid IO", lowErr, deniedErr, outputErr)
	}
}

func recipeHandler(t *testing.T, strategy recipe.Strategy, calls *atomic.Int32) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		call := calls.Add(1)
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		var wire struct {
			Maximum  uint64 `json:"max_completion_tokens"`
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err = json.Unmarshal(body, &wire); err != nil {
			t.Error(err)
		}
		if wire.Maximum != 17 || len(wire.Messages) != 2 {
			t.Error("reserved output limit lost")
			return
		}
		content := wire.Messages[1].Content
		if strings.Contains(content, "private-") {
			t.Error("metadata or identity leaked to model")
		}
		output := recipeReply(t, strategy, call, content)
		_, _ = io.WriteString(w, envelope(output, "stop"))
	}
}

func recipeReply(t *testing.T, strategy recipe.Strategy, call int32, content string) string {
	t.Helper()
	if call == 1 {
		var input structured.PlannerInput
		if err := json.Unmarshal([]byte(content), &input); err != nil {
			t.Error(err)
		}
		if input.Text != "original" || input.Strategy != strategy {
			t.Error("planner projection changed")
		}
		return `{"queries":["derived"]}`
	}
	var input structured.AssessorInput
	if err := json.Unmarshal([]byte(content), &input); err != nil {
		t.Error(err)
	}
	if len(input.Queries) != 1 {
		t.Error("assessment query count changed")
		return `{}`
	}
	query := input.Queries[0]
	if query.Index != 7 || len(query.Documents) != 1 || query.Documents[0] != "allowed snippet" {
		t.Error("assessment projection changed")
	}
	return `{"selected":[7],"sufficient":true}`
}

func recipeScopedRead(t *testing.T, revoked *atomic.Bool) access.Binding {
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
	mandatory, err := filter.Eq(builder, tenant, "allowed").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	binding, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "host-policy",
			PolicyEpoch: 7,
			IssuedAt:    now,
			ExpiresAt:   now.Add(time.Minute),
		},
		Mandatory:   mandatory,
		Schema:      schema,
		Publication: access.CurrentPublication(),
		Now:         func() time.Time { return now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			if revoked.Load() {
				return ragy.ErrUnavailable
			}
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	return binding
}

func TestRecipeHTTPTokenCounterRevocationPreventsDispatch(t *testing.T) {
	// Arrange.
	var calls atomic.Int32
	var revoked atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, envelope(`{"queries":["derived"]}`, "stop"))
	}))
	t.Cleanup(server.Close)
	cfg := recipeConfig(server.URL, false)
	cfg.CountTokens = func([]byte) (uint64, error) { revoked.Store(true); return 40, nil }
	planner, err := structured.NewPlanner[string, string](cfg, recipe.SingleRewrite, 100, recipePrice)
	if err != nil {
		t.Fatal(err)
	}
	request := retrieval.Request[string, string]{Read: recipeScopedRead(t, &revoked), Text: "original"}
	// Act.
	output, err := planner.Plan(context.Background(), request, recipe.ModelLimits{InputTokens: 40, OutputTokens: 17})
	// Assert.
	if !access.IsProtectionFailure(err) || calls.Load() != 0 || len(output.Queries) != 0 || output.Usage.Known {
		t.Fatal("counter revocation leaked model input", err)
	}
}
