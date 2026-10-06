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
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/recording"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type corpusMeta struct {
	Tenant string `json:"tenant"`
	Secret string `json:"secret"`
}
type corpusRequest = retrieval.Request[string, string]

func integrationRead(t *testing.T, revoked *atomic.Bool) (access.Binding, filter.Schema) {
	t.Helper()
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = fields.String("secret"); err != nil {
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
		"pub1",
		[]access.TargetRevision{
			{
				Target:            "lexical",
				Namespace:         "n",
				Source:            "corpus",
				Revision:          "r1",
				Transformation:    "chunk",
				AccessFingerprint: "acl",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot:  access.Snapshot{Identity: "policy", PolicyEpoch: 7, IssuedAt: now, ExpiresAt: now.Add(time.Minute)},
		Mandatory: mandatory, Schema: schema, Publication: publication, Now: time.Now,
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
	return read, schema
}

func corpusLocation(id string) source.Locator {
	return source.Locator{
		Kind: source.DocumentLocation,
		Reference: source.Reference{
			Namespace:         "n",
			Source:            "corpus",
			Revision:          "r1",
			Transformation:    "original",
			AccessFingerprint: "acl",
			Artifact:          id,
			Representation:    "text",
		},
	}
}

func integrationBackend(
	t *testing.T,
	read access.Binding,
	schema filter.Schema,
) retrieval.ProjectedBackend[string, string, struct{}, retrieval.NoRequestMeta, corpusMeta] {
	t.Helper()
	texts := []string{
		"Возврат оплаты за заказ: срок 10 дней.",
		"Сброс пароля: используйте ссылку восстановления доступа.",
		"Возврат устройства: срок 30 дней.",
		"Оплата банковской картой поддерживается.",
	}
	ids := []string{"d1", "d2", "d3", "d4"}
	docs := make([]retrieval.Document[corpusMeta], 0, 5)
	for i, text := range texts {
		docs = append(
			docs,
			retrieval.Document[corpusMeta]{
				ID:             ids[i],
				Content:        text,
				Meta:           corpusMeta{Tenant: "a", Secret: "private-metadata"},
				SourceSupports: []source.Locator{corpusLocation(ids[i])},
			},
		)
	}
	docs = append(
		docs,
		retrieval.Document[corpusMeta]{
			ID:      "private-id",
			Content: "private-payload возврат оплаты срок сброс пароля банковской картой",
			Meta:    corpusMeta{Tenant: "b", Secret: "private-metadata"},
		},
	)
	index, err := lexical.NewBM25Snapshot(
		context.Background(),
		schema,
		lexical.Config[corpusMeta]{SearchFields: []string{"content"}},
		read,
		docs,
		func(m corpusMeta) (corpusMeta, error) { return m, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	return retrieval.ProjectedBackend[string, string, struct{}, retrieval.NoRequestMeta, corpusMeta]{
		Next: index,
		Project: func(req corpusRequest) retrieval.Query[struct{}] {
			return retrieval.Query[struct{}]{
				Read:    req.Read,
				Text:    req.Text,
				Intent:  struct{}{},
				Meta:    retrieval.NoRequestMeta{},
				Options: req.Options,
				Plan:    retrieval.ProjectPlannedQuery(req.Plan, struct{}{}),
			}
		},
	}
}

func integrationRecipe(
	t *testing.T,
	strategy recipe.Strategy,
	endpoint string,
	read access.Binding,
	schema filter.Schema,
) *recipe.Recipe[string, string, corpusMeta] {
	t.Helper()
	price := func(structured.Usage) (uint64, error) { return 30, nil }
	planner, err := structured.NewPlanner[string, string](recipeConfig(endpoint, false), strategy, 1000, price)
	if err != nil {
		t.Fatal(err)
	}
	assessor, err := structured.NewAssessor[string, string, corpusMeta](
		recipeConfig(endpoint, true),
		3,
		10,
		8000,
		price,
	)
	if err != nil {
		t.Fatal(err)
	}
	backend := integrationBackend(t, read, schema)
	maxQueries := 2
	if strategy == recipe.SingleRewrite {
		maxQueries = 1
	}
	if strategy == recipe.Decomposition {
		maxQueries = 3
	}
	configured, err := recipe.New(recipe.Config[string, string, corpusMeta]{
		Strategy: strategy,
		Revision: "bounded-http-fixture",
		Backend:  backend,
		Admission: func(ctx context.Context, req corpusRequest) (retrieval.ReadCoverage, error) {
			_, e := retrieval.PrepareRead(ctx, req, backend)
			return retrieval.CompleteReadCoverage(), e
		},
		Planner:  planner.Plan,
		Assessor: assessor.Assess,
		Pricing: func(_ context.Context, op recipe.Operation) (recipe.Quote, error) {
			if op == recipe.Retrieve {
				return recipe.Quote{CostKnown: true}, nil
			}
			return recipe.Quote{
				Usage:     budget.Usage{InputTokens: 1024, OutputTokens: 256, Cost: 30},
				CostKnown: true,
			}, nil
		},
		CloneIntent:      func(s string) (string, error) { return s, nil },
		CloneRequestMeta: func(s string) (string, error) { return s, nil },
		CloneMeta:        func(m corpusMeta) (corpusMeta, error) { return m, nil },
		Supports: func(_ context.Context, _ access.Binding, doc retrieval.Document[corpusMeta]) ([]source.Locator, error) {
			return doc.SourceLocations(), nil
		},
		Identity: retrieval.DocumentIDResolver[corpusMeta]{},
		Limits: budget.Limits{
			RetrievalCalls: 3,
			ModelCalls:     2,
			Usage:          budget.Usage{InputTokens: 2048, OutputTokens: 512, Cost: 100},
		},
		RequireKnownCost: true,
		Duration:         5 * time.Second,
		Now:              time.Now,
		MaxQueries:       maxQueries,
		MaxDocuments:     10,
		FusionK:          60,
	})
	if err != nil {
		t.Fatal(err)
	}
	return configured
}

func integrationReply(t *testing.T, strategy recipe.Strategy, call int32, body []byte) string {
	t.Helper()
	var wire struct {
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
		Maximum uint64 `json:"max_completion_tokens"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Error(err)
		return `{}`
	}
	if len(wire.Messages) != 2 || wire.Maximum != 256 {
		t.Error("reservation projection invalid")
		return `{}`
	}
	content := wire.Messages[1].Content
	if strings.Contains(content, "private-") {
		t.Error("private evidence/metadata reached model")
	}
	if call == 1 {
		switch strategy {
		case recipe.SingleRewrite:
			return `{"queries":["возврат оплаты срок"]}`
		case recipe.MultiQuery:
			return `{"queries":["сброс пароля","восстановления доступа"]}`
		case recipe.Decomposition:
			return `{"queries":["возврат оплаты срок","банковской картой"]}`
		}
	}
	var input structured.AssessorInput
	if err := json.Unmarshal([]byte(content), &input); err != nil {
		t.Error(err)
	}
	if len(input.Queries) == 0 {
		t.Error("no actual retrieval evidence")
	}
	selected := []int{1}
	if strategy == recipe.MultiQuery {
		selected = []int{1, 2}
	}
	if strategy == recipe.Decomposition {
		selected = []int{0, 1}
	}
	output, err := json.Marshal(structured.AssessorOutput{Selected: selected, Sufficient: true})
	if err != nil {
		t.Error(err)
	}
	return string(output)
}

func TestThreeRecipesScopedBM25ThroughHTTPModels(t *testing.T) {
	for _, strategy := range []recipe.Strategy{recipe.SingleRewrite, recipe.MultiQuery, recipe.Decomposition} {
		t.Run(string(strategy), func(t *testing.T) {
			// Arrange: actual BM25 and actual HTTP transport; responses are protocol fixtures.
			var revoked atomic.Bool
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				call := calls.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				_, _ = io.WriteString(w, envelope(integrationReply(t, strategy, call, body), "stop"))
			}))
			t.Cleanup(server.Close)
			read, schema := integrationRead(t, &revoked)
			configured := integrationRecipe(t, strategy, server.URL, read, schema)
			// Act.
			result, err := configured.Run(
				context.Background(),
				corpusRequest{
					Read:    read,
					Text:    "original",
					Intent:  "private-intent",
					Meta:    "private-request",
					Options: retrieval.RetrieveOptions{TopK: 3},
				},
			)
			// Assert.
			assertHTTPRecipeResult(t, strategy, result, err, calls.Load())
		})
	}
}

func assertHTTPRecipeResult(
	t *testing.T,
	strategy recipe.Strategy,
	result recipe.Result[corpusMeta],
	err error,
	calls int32,
) {
	t.Helper()
	if err != nil || result.Outcome != recipe.Complete || calls != 2 || result.Budget.Actual.Cost != 60 ||
		result.Budget.Actual.InputTokens != 80 {
		t.Fatal("combined recipe accounting failed", err)
	}
	found := map[string]bool{}
	for _, hit := range result.Selected {
		if hit.Document.Meta.Tenant != "a" || hit.Document.ID == "private-id" || len(hit.Contributors) == 0 {
			t.Fatal("scope/provenance failed")
		}
		found[hit.Document.ID] = true
		assertOriginalContributors(t, hit.Contributors)
	}
	if strategy == recipe.MultiQuery {
		if !found["d2"] {
			t.Fatal("multi-query lost d2")
		}
		for _, hit := range result.Selected {
			if hit.Document.ID == "d2" && len(hit.Contributors) != 2 {
				t.Fatal("dedup lost a query support")
			}
		}
		return
	}
	if !found["d1"] || (strategy == recipe.Decomposition && !found["d4"]) {
		t.Fatal("required corpus evidence lost")
	}
}

func TestHTTPPlannerRevocationFailsCombinedRecipeClosed(t *testing.T) {
	// Arrange.
	var revoked atomic.Bool
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		revoked.Store(true)
		_, _ = io.WriteString(w, envelope(`{"queries":["возврат оплаты срок"]}`, "stop"))
	}))
	t.Cleanup(server.Close)
	read, schema := integrationRead(t, &revoked)
	configured := integrationRecipe(t, recipe.SingleRewrite, server.URL, read, schema)
	// Act.
	result, err := configured.Run(
		context.Background(),
		corpusRequest{Read: read, Text: "возврат оплаты", Options: retrieval.RetrieveOptions{TopK: 3}},
	)
	// Assert: the already retrieved original evidence cannot escape revocation.
	if !access.IsProtectionFailure(err) || !errors.Is(err, ragy.ErrUnavailable) || len(result.Queries) != 0 ||
		len(result.Selected) != 0 ||
		calls.Load() != 1 {
		t.Fatal("combined recipe leaked stale evidence", err)
	}
}

func assertOriginalContributors(t *testing.T, contributors []recipe.Contribution) {
	t.Helper()
	for _, contributor := range contributors {
		if len(contributor.Supports) != 1 || contributor.Supports[0].Reference.Revision != "r1" {
			t.Fatal("original source association lost")
		}
	}
}

type integrationSink struct {
	Calls  int
	Record evidence.Record
	Err    error
}

func (s *integrationSink) Write(_ context.Context, record evidence.Record) error {
	s.Calls++
	s.Record = record
	return s.Err
}

func TestHTTPRecipeRecordingPolicies(t *testing.T) {
	for _, mode := range []evidence.Mode{evidence.Disabled, evidence.BestEffort, evidence.Required} {
		t.Run(string(mode), func(t *testing.T) {
			// Arrange: failing sink after real scoped retrieval and model transport.
			var calls atomic.Int32
			var revoked atomic.Bool
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				call := calls.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				_, _ = io.WriteString(w, envelope(integrationReply(t, recipe.SingleRewrite, call, body), "stop"))
			}))
			t.Cleanup(server.Close)
			read, schema := integrationRead(t, &revoked)
			configured := integrationRecipe(t, recipe.SingleRewrite, server.URL, read, schema)
			sink := &integrationSink{Err: errors.New("private-sink-error")}
			request := corpusRequest{Read: read, Text: "original", Options: retrieval.RetrieveOptions{TopK: 3}}
			cfg := integrationRecordingConfig(configured, mode, sink, schema)
			// Act.
			execution, err := recording.Run(context.Background(), request, cfg)
			// Assert: no sink outcome redispatches retrieval/models; completed result retained.
			if recordingModeDisabled(t, mode, execution, err, sink, calls.Load()) {
				return
			}
			assertRecordedHTTP(t, execution, sink)
		})
	}
}

func assertRecordedHTTP(t *testing.T, execution evidence.Execution[recipe.Result[corpusMeta]], sink *integrationSink) {
	t.Helper()
	if sink.Calls != 1 || execution.Receipt.State != evidence.RecordingFailed {
		t.Fatal("sink failure lost receipt")
	}
	encoded, err := sink.Record.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "private-") || strings.Contains(string(encoded), "Возврат оплаты") {
		t.Fatal("record leaked raw payload/metadata")
	}
	snapshot, err := sink.Record.Snapshot()
	if err != nil || snapshot.Query.State != evidence.Omitted {
		t.Fatal("query privacy failed", err)
	}
	observed := map[string]bool{}
	for _, stage := range snapshot.Stages {
		if stage.Name.Value != nil {
			observed[*stage.Name.Value] = stage.Status == evidence.StageObserved
		}
	}
	if !observed["retrieve/0"] || !observed["retrieve/1"] || !observed["plan"] || !observed["assess"] ||
		!observed["fusion"] {
		t.Fatal("actual stage association lost", observed)
	}
	execution.Result.Selected[0].Document.Content = "private-mutation"
	unchanged, err := sink.Record.MarshalJSON()
	if err != nil || string(unchanged) != string(encoded) {
		t.Fatal("record aliases result", err)
	}
}

func recordingModeDisabled(
	t *testing.T,
	mode evidence.Mode,
	execution evidence.Execution[recipe.Result[corpusMeta]],
	err error,
	sink *integrationSink,
	calls int32,
) bool {
	t.Helper()
	if calls != 2 || execution.Result.Outcome != recipe.Complete {
		t.Fatal("recording repeated or erased completed recipe", err)
	}
	if mode == evidence.Required {
		if !errors.Is(err, evidence.ErrRecordingFailed) {
			t.Fatal("required recording claimed success", err)
		}
	} else if err != nil {
		t.Fatal(err)
	}
	if mode != evidence.Disabled {
		return false
	}
	if sink.Calls != 0 || execution.Receipt.State != evidence.RecordingDisabled {
		t.Fatal("disabled recording dispatched sink")
	}
	return true
}

func integrationRecordingConfig(
	configured *recipe.Recipe[string, string, corpusMeta],
	mode evidence.Mode,
	sink *integrationSink,
	schema filter.Schema,
) recording.Config[string, string, corpusMeta] {
	return recording.Config[string, string, corpusMeta]{
		Recipe:      configured,
		Mode:        mode,
		Sink:        sink,
		RetrievalID: "recorded-http",
		Schema:      schema,
		Codec:       retrieval.NewJSONCodec[corpusMeta](schema),
		SourceAdmission: func(ctx context.Context, binding access.Binding, ref source.Reference) error {
			if ref.Namespace != "n" || ref.Source != "corpus" || ref.Revision != "r1" ||
				ref.AccessFingerprint != "acl" {
				return access.Protect(ragy.ErrUnavailable)
			}
			return binding.Check(ctx)
		},
		CloneMeta: func(m corpusMeta) (corpusMeta, error) { return m, nil },
		Policy: evidence.Policy{
			AllowIdentifier: func(_ evidence.IdentifierKind, id string) bool { return !strings.Contains(id, "private-") },
			AllowNumbers:    true,
		},
	}
}

func TestHTTPFailedAssessmentRetainsRecipeJournal(t *testing.T) {
	// Arrange: valid HTTP planner followed by a schema-valid foreign selection.
	var calls atomic.Int32
	var revoked atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := calls.Add(1)
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		response := `{"selected":[99],"sufficient":true}`
		if call == 1 {
			response = integrationReply(t, recipe.SingleRewrite, call, body)
		}
		_, _ = io.WriteString(w, envelope(response, "stop"))
	}))
	t.Cleanup(server.Close)
	read, schema := integrationRead(t, &revoked)
	configured := integrationRecipe(t, recipe.SingleRewrite, server.URL, read, schema)
	sink := &integrationSink{}
	cfg := integrationRecordingConfig(configured, evidence.Required, sink, schema)
	// Act.
	execution, err := recording.Run(
		context.Background(),
		corpusRequest{Read: read, Text: "original", Options: retrieval.RetrieveOptions{TopK: 3}},
		cfg,
	)
	// Assert: the failed model call settles, original revision and actual earlier hits remain recorded.
	if !errors.Is(err, ragy.ErrProtocol) || calls.Load() != 2 || sink.Calls != 1 ||
		execution.Receipt.State != evidence.Recorded ||
		execution.Result.Outcome != recipe.Failure ||
		len(execution.Result.Selected) != 0 ||
		len(execution.Result.Queries) != 2 ||
		execution.Result.Budget.Actual.Cost != 60 {
		t.Fatal("HTTP failure journal lost or retried", err)
	}
	snapshot, decodeErr := sink.Record.Snapshot()
	if decodeErr != nil || snapshot.Outcome != evidence.Failed {
		t.Fatal("failed attempt recorded success", decodeErr)
	}
	stages := map[string]evidence.WireStage{}
	for _, stage := range snapshot.Stages {
		if stage.Name.Value != nil {
			stages[*stage.Name.Value] = stage
		}
	}
	if len(stages["retrieve/1"].Hits) == 0 || stages["assess"].Status != evidence.StageObserved ||
		stages["fusion"].Status != evidence.NotRun {
		t.Fatal("failed assessment lost actual stage evidence")
	}
}
