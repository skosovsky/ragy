package recipe_test

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type boundedEncoder struct {
	inputs    []string
	limits    []recipe.ModelLimits
	malformed bool
}

func (*boundedEncoder) Admit(ctx context.Context, _ dense.Request) error { return ctx.Err() }

func (e *boundedEncoder) Encode(
	_ context.Context,
	r dense.Request,
	l recipe.ModelLimits,
) (dense.Result, recipe.Usage, error) {
	e.inputs = append(e.inputs, r.Inputs[0])
	e.limits = append(e.limits, l)
	space := dense.Space{
		Model:         "test",
		ModelRevision: "1",
		Configuration: "query",
		VectorSpace:   "same",
		Dimension:     2,
		Metric:        embedding.Dot,
	}
	vector := []float32{float32(len(e.inputs)), 1}
	if e.malformed {
		vector = []float32{1}
	}
	return dense.Result{Embeddings: []dense.Embedding{{Space: space, Vector: vector}}}, recipe.Usage{}, nil
}

type encodingBackend struct {
	next    retrieval.RequestBackend[[]string, []string, meta]
	vectors [][]float32
	spaces  []dense.Space
}

func (b *encodingBackend) Retrieve(ctx context.Context, r request) (retrieval.ResultSet[meta], error) {
	b.vectors = append(b.vectors, slices.Clone(r.Options.Vector))
	b.spaces = append(b.spaces, r.Options.Space)
	return b.next.Retrieve(ctx, r)
}

func TestIndependentQueryEncodingRetainsUnknownReservation(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.planned = []string{"rewrite"}
	f.selected = []int{1}
	f.results["rewrite"] = []retrieval.Document[meta]{document("d1")}
	encoder := &boundedEncoder{}
	backend := &encodingBackend{next: f.config.Backend}
	f.config.Backend = backend
	f.config.QueryEncoder = encoder
	f.config.Limits.ModelCalls = 4
	f.config.Limits.Usage = budget.Usage{InputTokens: 4096, OutputTokens: 1024, Cost: 200}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || !slices.Equal(encoder.inputs, []string{"original", "rewrite"}) || len(result.Encoding) != 2 ||
		result.Encoding[0].Embeddings[0].Vector[0] == result.Encoding[1].Embeddings[0].Vector[0] ||
		result.Budget.UnknownUsage != 2 ||
		result.Budget.Occupied.ModelCalls != 4 {
		t.Fatal(result, encoder, err)
	}
	if len(backend.vectors) != 2 || !slices.Equal(backend.vectors[0], []float32{1, 1}) ||
		!slices.Equal(backend.vectors[1], []float32{2, 1}) ||
		backend.spaces[0] != result.Encoding[0].Embeddings[0].Space {
		t.Fatal(backend)
	}
	for _, limit := range encoder.limits {
		if limit.InputTokens != 1024 || limit.OutputTokens != 256 {
			t.Fatal(limit)
		}
	}
}
func TestUnsupportedEncodingRejectsBeforeReservation(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.config.QueryEncoder = recipe.UnsupportedQueryEncoderBridge{}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if !errors.Is(err, ragy.ErrUnsupported) || len(f.retrieved) != 0 || f.modelCalls != 0 || len(result.Stages) != 0 {
		t.Fatal(result, err)
	}
}
func TestMalformedEncodingRejectsBeforeBackend(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.config.QueryEncoder = &boundedEncoder{malformed: true}
	// Act.
	_, err := f.run(context.Background(), t)
	// Assert.
	if !errors.Is(err, ragy.ErrProtocol) || len(f.retrieved) != 0 {
		t.Fatal(err)
	}
}
func TestTopKAndPackingCannotCompleteMissingSubquestion(t *testing.T) {
	for _, packed := range []bool{false, true} {
		t.Run(map[bool]string{false: "topk", true: "packing"}[packed], func(t *testing.T) {
			// Arrange.
			f := newFixture(t, recipe.Decomposition)
			f.planned = []string{"one", "two"}
			f.selected = []int{0, 1}
			f.sufficient = true
			f.results["one"] = []retrieval.Document[meta]{document("d1")}
			f.results["two"] = []retrieval.Document[meta]{document("d2")}
			req := recordedRequest(f)
			req.Options.TopK = 1
			if packed {
				req.Options.TopK = 2
				f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{
					Resource:  retrieval.RuneResource(45),
					CloneMeta: f.config.CloneMeta,
				}
			}
			r, err := recipe.New(f.config)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			result, err := r.RunOwn(context.Background(), req)
			// Assert.
			if err != nil || result.Outcome == recipe.Complete || len(result.Coverage) != 2 {
				t.Fatal(result, err)
			}
			if !result.Coverage[0].HasEvidence || !result.Coverage[1].HasEvidence {
				t.Fatal("retrieved evidence lost")
			}
		})
	}
}
func TestCallerLedgerDeadlineLimitsTextCallback(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.Decomposition)
	ledger, err := budget.New(
		budget.Config{Limits: f.config.Limits, Now: f.config.Now, Deadline: f.now.Add(40 * time.Millisecond)},
	)
	if err != nil {
		t.Fatal(err)
	}
	f.config.Planner = func(ctx context.Context, _ request, _ recipe.ModelLimits) (recipe.Planning, error) {
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) > 60*time.Millisecond {
			t.Fatal(deadline)
		}
		<-ctx.Done()
		return recipe.Planning{}, ctx.Err()
	}
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := r.Run(context.Background(), recordedRequest(f), ledger)
	// Assert.
	if err != nil || result.Stop != recipe.DeadlineReached || ledger.Snapshot().Occupied.ModelCalls != 1 {
		t.Fatal(result, err)
	}
}
func TestBackendMustExplicitlyDeclareModelFree(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.config.BackendModelFree = false
	// Act.
	_, err := recipe.New(f.config)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal(err)
	}
}

func TestProjectedPartialSnippetCannotClaimFullDelivery(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.Decomposition)
	f.planned = []string{"one"}
	f.selected = []int{0}
	f.sufficient = true
	f.results["one"] = []retrieval.Document[meta]{document("d1")}
	f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{
		Resource:  retrieval.RuneResource(100),
		CloneMeta: f.config.CloneMeta,
		Snippet:   func(retrieval.Document[meta]) string { return "partial" },
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Outcome != recipe.Partial || len(result.Coverage) != 1 ||
		!result.Coverage[0].DeliveryUncertain ||
		result.Coverage[0].DeliveredEvidence {
		t.Fatal(result, err)
	}
}

func TestArtifactMeasurementRespectsAttemptDeadline(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.Decomposition)
	f.planned = []string{"one"}
	f.selected = []int{0}
	f.sufficient = true
	f.results["one"] = []retrieval.Document[meta]{document("d1")}
	f.config.Duration = 35 * time.Millisecond
	resource := retrieval.RuneResource(100)
	resource.Measure = func(ctx context.Context, _ string) (int64, error) { <-ctx.Done(); return 0, ctx.Err() }
	f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{Resource: resource, CloneMeta: f.config.CloneMeta}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Stop != recipe.DeadlineReached || result.Outcome == recipe.Complete ||
		result.Artifact != nil || !result.ArtifactRequested ||
		result.Coverage[0].DeliveredEvidence ||
		!result.Coverage[0].DeliveryUncertain {
		t.Fatal(result, err)
	}
}

type contentIdentity struct{}

func (contentIdentity) Resolve(doc retrieval.Document[meta]) retrieval.Identity {
	return retrieval.Identity{DocumentID: doc.ID, MergeKey: doc.Content}
}
func TestSameDocumentIDDifferentFragmentsCannotCompleteDelivery(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.Decomposition)
	f.planned = []string{"one", "two"}
	f.selected = []int{0, 1}
	f.sufficient = true
	one, two := document("fact-one"), document("fact-two")
	one.ID = "same-storage-key"
	two.ID = "same-storage-key"
	f.results["one"] = []retrieval.Document[meta]{one}
	f.results["two"] = []retrieval.Document[meta]{two}
	f.config.Identity = contentIdentity{}
	f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{
		Resource:  retrieval.RuneResource(80),
		CloneMeta: f.config.CloneMeta,
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Outcome != recipe.Partial || len(result.Selected) != 2 ||
		len(result.Artifact.Snippets) != 1 ||
		!result.Coverage[0].DeliveredEvidence ||
		result.Coverage[1].DeliveredEvidence {
		t.Fatal(result, err)
	}
}
func TestSamePackedContentRetainsBothActualContributors(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.Decomposition)
	f.planned = []string{"one", "two"}
	f.selected = []int{0, 1}
	f.sufficient = true
	one, two := document("d1"), document("d2")
	two.Content = one.Content
	f.results["one"] = []retrieval.Document[meta]{one}
	f.results["two"] = []retrieval.Document[meta]{two}
	f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{
		Resource:  retrieval.RuneResource(80),
		CloneMeta: f.config.CloneMeta,
		DedupKey:  func(doc retrieval.Document[meta]) string { return doc.Content },
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Outcome != recipe.Complete || len(result.Artifact.Snippets) != 1 ||
		len(result.Artifact.Snippets[0].Contributors) != 2 ||
		!result.Coverage[0].DeliveredEvidence ||
		!result.Coverage[1].DeliveredEvidence {
		t.Fatal(result, err)
	}
}

func TestPackedContributorSnapshotOwnsAndValidatesIndices(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.Decomposition)
	f.planned = []string{"one"}
	f.selected = []int{0}
	f.sufficient = true
	f.results["one"] = []retrieval.Document[meta]{document("d1")}
	f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{
		Resource:  retrieval.RuneResource(80),
		CloneMeta: f.config.CloneMeta,
	}
	original, err := f.run(context.Background(), t)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	snapshot, err := recipe.SnapshotResult(context.Background(), f.read, original, f.config.CloneMeta)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	snapshot.Artifact.Snippets[0].Contributors[0].InputIndex = 100
	if original.Artifact.Snippets[0].Contributors[0].InputIndex != 0 {
		t.Fatal("snapshot aliases contributor input")
	}
	_, err = recipe.SnapshotResult(context.Background(), f.read, snapshot, f.config.CloneMeta)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal(err)
	}
}

func TestSamePackedTextPreservesPerContributorUncertainty(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.Decomposition)
	f.planned = []string{"one", "two"}
	f.selected = []int{0, 1}
	f.sufficient = true
	one, two := document("d1"), document("d2")
	two.Content = one.Content
	mapped, err := source.DerivedText(two.Content, []source.Locator{location("d2")})
	if err != nil {
		t.Fatal(err)
	}
	two.SourceMapping = mapped
	f.results["one"] = []retrieval.Document[meta]{one}
	f.results["two"] = []retrieval.Document[meta]{two}
	f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{
		Resource:  retrieval.RuneResource(80),
		CloneMeta: f.config.CloneMeta,
		DedupKey:  func(doc retrieval.Document[meta]) string { return doc.Content },
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Outcome != recipe.Partial || !result.Coverage[0].DeliveredEvidence ||
		result.Coverage[0].DeliveryUncertain ||
		result.Coverage[1].DeliveredEvidence ||
		!result.Coverage[1].DeliveryUncertain {
		t.Fatal(result, err)
	}
}

func TestDisabledArtifactDeliversSelectedDocumentsExplicitly(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("one")}
	f.selected = []int{0}
	// Act.
	result, err := f.run(context.Background(), t)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := recipe.SnapshotResult(context.Background(), f.read, result, f.config.CloneMeta)
	// Assert.
	if err != nil || result.ArtifactRequested || snapshot.ArtifactRequested || result.Artifact != nil ||
		result.Outcome != recipe.Complete ||
		!result.Coverage[0].DeliveredEvidence ||
		result.Coverage[0].DeliveryUncertain {
		t.Fatal(result, snapshot, err)
	}
}

func TestRequestedArtifactProtectionFailureSuppressesJournal(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("one")}
	f.selected = []int{0}
	resource := retrieval.RuneResource(100)
	resource.Measure = func(context.Context, string) (int64, error) { return 0, errors.New("measurement failed") }
	f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{Resource: resource, CloneMeta: f.config.CloneMeta}
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := r.RunOwnObserved(context.Background(), recordedRequest(f))
	// Assert.
	if err == nil || result.ArtifactRequested || result.Artifact != nil || result.Outcome != "" ||
		len(result.Queries) != 0 {
		t.Fatal(result, err)
	}
	snapshot, snapshotErr := recipe.SnapshotResult(context.Background(), f.read, result, f.config.CloneMeta)
	if snapshotErr != nil || snapshot.ArtifactRequested || snapshot.Artifact != nil {
		t.Fatal(snapshot, snapshotErr)
	}
}
