//go:build darwin || linux

package integration_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"path/filepath"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	graphmanaged "github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lexical"
	lexicalmanaged "github.com/skosovsky/ragy/lexical/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorfs "github.com/skosovsky/ragy/tensor/persistent"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

type meta struct {
	Tenant     string `json:"tenant"`
	Visibility string `json:"visibility"`
	Source     string `json:"source"`
	Revision   string `json:"revision"`
	Artifact   string `json:"artifact"`
}
type batch struct {
	Graph   graphmanaged.Payload[meta]    `json:"graph"`
	Dense   []densefs.Record[meta]        `json:"dense"`
	Tensor  []tensorfs.Record[meta]       `json:"tensor"`
	Lexical []lexicalmanaged.Record[meta] `json:"lexical"`
}
type densePort struct{ adapter *densefs.Adapter[meta] }

func (p densePort) Stage(
	ctx context.Context,
	request lifecycle.StageRequest,
	input batch,
) (lifecycle.StageResult, error) {
	return p.adapter.Stage(ctx, request, input.Dense)
}
func (p densePort) Inspect(ctx context.Context, request lifecycle.StageRequest) (lifecycle.StageResult, error) {
	return p.adapter.Inspect(ctx, request)
}

type secondaryPort struct {
	tensor    *tensorfs.Adapter[meta]
	graph     *graphmanaged.Adapter[meta]
	lexical   *lexicalmanaged.Adapter[meta]
	failAfter bool
	calls     int
}

func (p *secondaryPort) Stage(
	ctx context.Context,
	request lifecycle.StageRequest,
	input batch,
) (lifecycle.StageResult, error) {
	p.calls++
	var result lifecycle.StageResult
	var err error
	switch {
	case p.tensor != nil:
		result, err = p.tensor.Stage(ctx, request, input.Tensor)
	case p.graph != nil:
		result, err = p.graph.Stage(ctx, request, input.Graph)
	default:
		result, err = p.lexical.Stage(ctx, request, input.Lexical)
	}
	if err != nil {
		return result, err
	}
	if p.failAfter {
		p.failAfter = false
		return lifecycle.StageResult{}, context.DeadlineExceeded
	}
	return result, nil
}
func (p *secondaryPort) Inspect(ctx context.Context, request lifecycle.StageRequest) (lifecycle.StageResult, error) {
	if p.tensor != nil {
		return p.tensor.Inspect(ctx, request)
	}
	if p.graph != nil {
		return p.graph.Inspect(ctx, request)
	}
	return p.lexical.Inspect(ctx, request)
}

type fixture struct {
	store     lifecycle.Store
	schema    filter.Schema
	dense     *densefs.Adapter[meta]
	secondary *secondaryPort
	executor  *lifecycle.Executor[batch]
	target    string
	now       time.Time
}

func denseSpace() dense.Space {
	return dense.Space{Metric: "normalized-dot",
		Model:         "dense-fixture",
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
}
func tensorSpace() tensor.Space {
	return tensor.Space{Metric: "normalized-dot",
		Model:         "tensor-fixture",
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
}
func cloneMeta(value meta) (meta, error) { return value, nil }
func schema(t *testing.T) filter.Schema {
	t.Helper()
	fields := filter.NewSchema()
	for _, name := range []string{"tenant", "visibility", "source", "revision", "artifact"} {
		if _, err := fields.String(name); err != nil {
			t.Fatal(err)
		}
	}
	result, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	return result
}
func newFixture(t *testing.T, target string) *fixture {
	t.Helper()
	store, err := filestore.New(filepath.Join(t.TempDir(), "manifests"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	fields := schema(t)
	denseAdapter, err := densefs.New(
		densefs.Config[meta]{
			Root:            filepath.Join(t.TempDir(), "dense"),
			Namespace:       "fixture-a",
			Target:          "dense",
			Store:           store,
			Schema:          fields,
			Space:           denseSpace(),
			CloneMeta:       cloneMeta,
			MaxCatalogBytes: 1 << 20,
			MaxPayloadBytes: 1 << 20,
			MaxRecords:      100,
			MaxScanRecords:  100,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	secondary := &secondaryPort{tensor: nil, graph: nil, lexical: nil, failAfter: false, calls: 0}
	switch target {
	case "tensor":
		secondary.tensor, err = tensorfs.New(
			tensorfs.Config[meta]{
				Root:            filepath.Join(t.TempDir(), "tensor"),
				Namespace:       "fixture-a",
				Target:          target,
				Store:           store,
				Schema:          fields,
				Space:           tensorSpace(),
				CloneMeta:       cloneMeta,
				MaxCatalogBytes: 1 << 20,
				MaxPayloadBytes: 1 << 20,
				MaxRecords:      100,
			},
		)
	case "graph":
		secondary.graph, err = graphmanaged.New(
			graphmanaged.Config[meta]{
				Namespace:  "fixture-a",
				Target:     target,
				Store:      store,
				Schema:     graph.Schema{NodeAttributes: fields, EdgeAttributes: fields},
				CloneMeta:  cloneMeta,
				MaxRecords: 100, MaxAdmissionRecords: 100,
			},
		)
	default:
		secondary.lexical, err = lexicalmanaged.New(
			lexicalmanaged.Config[meta]{
				MaxCachedSnapshots: 32,
				Namespace:          "fixture-a",
				Target:             target,
				Store:              store,
				Schema:             fields,
				BM25:               lexical.Config[meta]{SearchFields: []string{"content"}},
				CloneMeta:          cloneMeta,
			},
		)
	}
	if err != nil {
		t.Fatal(err)
	}
	out := &fixture{
		store:     store,
		schema:    fields,
		dense:     denseAdapter,
		secondary: secondary,
		executor:  nil,
		target:    target,
		now:       time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
	}
	out.executor, err = lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[batch]{
			Store: store,
			Targets: []lifecycle.Registration[batch]{
				{Name: "dense", Port: densePort{adapter: denseAdapter}},
				{Name: target, Port: secondary},
			},
			ClonePayload:    cloneBatch,
			ValidatePayload: validateBatch,
			Now:             func() time.Time { return out.now },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return out
}
func cloneBatch(input batch) (batch, error) {
	input.Dense = slices.Clone(input.Dense)
	for i := range input.Dense {
		input.Dense[i].Value.Vector = slices.Clone(input.Dense[i].Value.Vector)
	}
	input.Tensor = slices.Clone(input.Tensor)
	for i := range input.Tensor {
		input.Tensor[i].Value.Tensor = slices.Clone(input.Tensor[i].Value.Tensor)
		for j := range input.Tensor[i].Value.Tensor {
			input.Tensor[i].Value.Tensor[j] = slices.Clone(input.Tensor[i].Value.Tensor[j])
		}
	}
	input.Graph.Nodes = slices.Clone(input.Graph.Nodes)
	input.Graph.Edges = slices.Clone(input.Graph.Edges)
	for i := range input.Graph.Nodes {
		input.Graph.Nodes[i].Value.Labels = slices.Clone(input.Graph.Nodes[i].Value.Labels)
	}
	input.Lexical = slices.Clone(input.Lexical)
	return input, nil
}

type edgeFingerprint struct {
	Reference source.Reference `json:"reference"`
	ID        string           `json:"id"`
	Source    string           `json:"source"`
	Target    string           `json:"target"`
	Relation  string           `json:"relation"`
	Meta      meta             `json:"meta"`
}

func fingerprint(input batch) string {
	// Fingerprint the fixture's explicit wire projection, not untagged index types.
	wire := struct {
		References   []source.Reference `json:"references"`
		DenseSpaces  []dense.Space      `json:"dense_spaces"`
		TensorSpaces []tensor.Space     `json:"tensor_spaces"`
		Vectors      [][]float32        `json:"vectors"`
		Tokens       []tensor.Tensor    `json:"tokens"`
		Metadata     []meta             `json:"metadata"`
		Contents     []string           `json:"contents"`
		IDs          []string           `json:"ids"`
		GraphLabels  [][]string         `json:"graph_labels"`
		GraphEdges   []edgeFingerprint  `json:"graph_edges"`
	}{References: nil, DenseSpaces: nil, TensorSpaces: nil, Vectors: nil, Tokens: nil, Metadata: nil, Contents: nil, IDs: nil}
	for _, record := range input.Dense {
		wire.References = append(wire.References, record.Reference)
		wire.DenseSpaces = append(wire.DenseSpaces, record.Value.Space)
		wire.Vectors = append(wire.Vectors, record.Value.Vector)
		wire.Metadata = append(wire.Metadata, record.Value.Meta)
		wire.Contents = append(wire.Contents, record.Value.Content)
		wire.IDs = append(wire.IDs, record.Value.ID)
	}
	for _, record := range input.Tensor {
		wire.References = append(wire.References, record.Reference)
		wire.TensorSpaces = append(wire.TensorSpaces, record.Value.Space)
		wire.Tokens = append(wire.Tokens, record.Value.Tensor)
		wire.Metadata = append(wire.Metadata, record.Value.Meta)
		wire.Contents = append(wire.Contents, record.Value.Content)
		wire.IDs = append(wire.IDs, record.Value.ID)
	}
	for _, record := range input.Lexical {
		wire.References = append(wire.References, record.Reference)
		wire.Metadata = append(wire.Metadata, record.Document.Meta)
		wire.Contents = append(wire.Contents, record.Document.Content)
		wire.IDs = append(wire.IDs, record.Document.ID)
	}
	for _, record := range input.Graph.Nodes {
		wire.References = append(wire.References, record.Reference)
		wire.Metadata = append(wire.Metadata, record.Value.Meta)
		wire.Contents = append(wire.Contents, record.Value.Content)
		wire.IDs = append(wire.IDs, record.Value.ID)
		wire.GraphLabels = append(wire.GraphLabels, record.Value.Labels)
	}
	for _, record := range input.Graph.Edges {
		wire.GraphEdges = append(
			wire.GraphEdges,
			edgeFingerprint{
				Reference: record.Reference,
				ID:        record.Value.ID,
				Source:    record.Value.SourceID,
				Target:    record.Value.TargetID,
				Relation:  record.Value.Type,
				Meta:      record.Value.Meta,
			},
		)
	}
	data, _ := json.Marshal(wire)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func validateBatch(manifest lifecycle.Manifest, input batch) error {
	for _, edge := range input.Graph.Edges {
		if edge.Reference.Validate() != nil || edge.Value.Validate() != nil ||
			edge.Reference.Representation != "graph-edge" ||
			edge.Reference.Artifact != edge.Value.ID {
			return ragy.ErrInvalidArgument
		}
	}
	for _, record := range input.Lexical {
		if record.Document.ScoreState != retrieval.ScoreAbsent || record.Document.Score != 0 ||
			record.Document.ScoreSemantics != "" ||
			record.Document.Rank != 0 ||
			len(record.Document.ScoreHistory) != 0 {
			return ragy.ErrInvalidArgument
		}
	}

	if manifest.Payload != fingerprint(input) {
		return ragy.ErrInvalidArgument
	}
	return nil
}

func sourceBatch(sourceID, revision string, artifacts []string) batch {
	out := batch{Dense: nil, Tensor: nil, Lexical: nil, Graph: graphmanaged.Payload[meta]{Nodes: nil, Edges: nil}}
	for i, artifact := range artifacts {
		reference := source.Reference{
			Namespace:         "fixture-a",
			Source:            sourceID,
			Revision:          revision,
			Transformation:    "ingest",
			AccessFingerprint: "acl",
			Artifact:          artifact,
			Representation:    "dense-vector",
		}
		metadata := meta{Tenant: "a", Visibility: "public", Source: sourceID, Revision: revision, Artifact: artifact}
		vector := []float32{1, 0}
		if i > 0 {
			vector = []float32{0, 1}
		}
		content := "needle " + sourceID + " " + revision + " " + artifact
		out.Dense = append(
			out.Dense,
			densefs.Record[meta]{
				Reference: reference,
				Value: dense.Record[meta]{
					Space:   denseSpace(),
					ID:      artifact,
					Content: content,
					Meta:    metadata,
					Vector:  vector,
				},
			},
		)
		reference.Representation = "token-matrix"
		out.Tensor = append(
			out.Tensor,
			tensorfs.Record[meta]{
				Reference: reference,
				Value: tensor.Record[meta]{
					ID:      artifact,
					Content: content,
					Meta:    metadata,
					Tensor:  tensor.Tensor{slices.Clone(vector)},
					Space:   tensorSpace(),
				},
			},
		)
		reference.Representation = "graph-node"
		out.Graph.Nodes = append(
			out.Graph.Nodes,
			graphmanaged.Node[meta]{
				Reference: reference,
				Value: graph.Node[meta]{
					ID:      artifact,
					Labels:  []string{"Document"},
					Content: content,
					Meta:    metadata,
				},
			},
		)
		reference.Representation = "utf8"
		out.Lexical = append(
			out.Lexical,
			lexicalmanaged.Record[meta]{
				Reference: reference,
				Document:  retrieval.Document[meta]{ID: artifact, Content: content, Meta: metadata},
			},
		)
	}
	return out
}
func plan(id, expected, target string, input batch) lifecycle.Manifest {
	reference := input.Dense[0].Reference
	denseArtifacts := make([]lifecycle.Artifact, 0, len(input.Dense))
	secondaryArtifacts := make([]lifecycle.Artifact, 0, len(input.Dense))
	for i, record := range input.Dense {
		denseArtifacts = append(
			denseArtifacts,
			lifecycle.Artifact{Reference: record.Reference, Supports: []source.Reference{input.Lexical[i].Reference}},
		)
		ref := input.Lexical[i].Reference
		switch target {
		case "tensor":
			ref = input.Tensor[i].Reference
		case "graph":
			continue
		}
		secondaryArtifacts = append(
			secondaryArtifacts,
			lifecycle.Artifact{Reference: ref, Supports: []source.Reference{input.Lexical[i].Reference}},
		)
	}
	if target == "graph" {
		for i, record := range input.Graph.Nodes {
			secondaryArtifacts = append(
				secondaryArtifacts,
				lifecycle.Artifact{
					Reference: record.Reference,
					Supports:  []source.Reference{input.Lexical[min(i, len(input.Lexical)-1)].Reference},
				},
			)
		}
		for _, record := range input.Graph.Edges {
			secondaryArtifacts = append(
				secondaryArtifacts,
				lifecycle.Artifact{
					Reference: record.Reference,
					Supports:  []source.Reference{input.Lexical[0].Reference},
				},
			)
		}
	}
	return lifecycle.Manifest{
		ID:                  id,
		Key:                 id,
		Payload:             fingerprint(input),
		ExpectedPublication: expected,
		State:               lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      reference.Namespace,
			Source:         reference.Source,
			Revision:       reference.Revision,
			Content:        reference.Source + reference.Revision,
			Transformation: reference.Transformation,
			Access:         reference.AccessFingerprint,
		},
		Targets: []lifecycle.Target{
			{Name: "dense", Required: true, State: lifecycle.TargetPending, Artifacts: denseArtifacts},
			{Name: target, Required: true, State: lifecycle.TargetPending, Artifacts: secondaryArtifacts},
		},
	}
}
func (f *fixture) ingest(t *testing.T, manifest lifecycle.Manifest, input batch) {
	t.Helper()
	ctx := context.Background()
	if _, err := f.executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{"dense", f.target} {
		if _, err := f.executor.Stage(ctx, "fixture-a", manifest.ID, target, input); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := f.executor.Publish(ctx, "fixture-a", manifest.ID); err != nil {
		t.Fatal(err)
	}
}
func (f *fixture) pin(t *testing.T) access.Binding {
	t.Helper()
	publication, err := lifecycle.CapturePublication(
		context.Background(),
		f.store,
		"fixture-a",
		[]string{"dense", f.target},
	)
	if err != nil {
		t.Fatal(err)
	}
	return f.bind(t, publication)
}

func (f *fixture) bind(t *testing.T, publication access.Publication) access.Binding {
	t.Helper()
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
				ExpiresAt:   f.now.Add(time.Minute),
			},
			Mandatory:   mandatory,
			Schema:      f.schema,
			Publication: publication,
			Now:         func() time.Time { return f.now },
			Authority:   access.AuthorityFunc(func(context.Context, access.Snapshot) error { return nil }),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return read
}

func (f *fixture) read(
	t *testing.T,
	binding access.Binding,
	batches ...batch,
) ([]retrieval.Document[meta], []retrieval.Document[meta]) {
	t.Helper()
	denseResult, err := f.dense.Retrieve(
		context.Background(),
		retrieval.Query[densefs.Intent]{
			Read:    binding,
			Intent:  densefs.Intent{Embedding: dense.Embedding{Space: denseSpace(), Vector: []float32{1, 0}}},
			Options: retrieval.RetrieveOptions{TopK: 10},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if f.target == "graph" {
		return denseResult.Documents(), f.graphDocs(t, binding, batches)
	}
	if f.target == "lexical" {
		result, readErr := f.secondary.lexical.Retrieve(
			context.Background(),
			retrieval.Query[struct{}]{Read: binding, Text: "needle", Options: retrieval.RetrieveOptions{TopK: 10}},
		)
		if readErr != nil {
			t.Fatal(readErr)
		}
		return denseResult.Documents(), result.Documents()
	}
	var refs []source.Reference
	for _, input := range batches {
		for _, record := range input.Tensor {
			refs = append(refs, record.Reference)
		}
	}
	result, err := f.secondary.tensor.Query(
		context.Background(),
		retrieval.Query[tensorquery.Intent]{
			Read: binding,
			Intent: tensorquery.Intent{
				Embedding:       tensor.Embedding{Space: tensorSpace(), Tokens: tensor.Tensor{{1, 0}}},
				Candidates:      refs,
				CandidateBudget: 100,
			},
			Options: retrieval.RetrieveOptions{TopK: 10},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return denseResult.Documents(), result.Documents.Documents()
}
func assertRevision(t *testing.T, docs []retrieval.Document[meta], policyRevision string, policyCount int) {
	t.Helper()
	count := 0
	faq := 0
	for _, doc := range docs {
		if doc.Meta.Source == "faq" {
			faq++
			continue
		}
		if doc.Meta.Source != "policy" || doc.Meta.Revision != policyRevision {
			t.Fatal("mixed or unexpected source revision", doc.Meta)
		}
		count++
	}
	if count != policyCount || faq != 1 {
		t.Fatal("source inventory changed", count, faq)
	}
}

func TestActualJointPublicationUnknownStageAndRetainedSnapshot(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		t.Run("dense+"+target, func(t *testing.T) { jointCase(t, target) })
	}
}
func jointCase(t *testing.T, target string) {
	t.Helper()
	// Arrange: both actual targets publish policy p1/p2 and unrelated faq f1.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	captured := f.pin(t)
	next := sourceBatch("policy", "r2", []string{"p3"})
	manifest := plan("policy-r2", "policy-r1", target, next)
	ctx := context.Background()
	if _, err := f.executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Stage(ctx, "fixture-a", manifest.ID, "dense", next); err != nil {
		t.Fatal(err)
	}
	f.secondary.failAfter = true
	before := f.secondary.calls
	// Act: secondary really commits, then its response is lost.
	if _, err := f.executor.Stage(
		ctx,
		"fixture-a",
		manifest.ID,
		target,
		next,
	); !errors.Is(
		err,
		lifecycle.ErrOutcomeUnknown,
	) {
		t.Fatal("write timeout was treated as confirmed", err)
	}
	if _, err := f.executor.Publish(ctx, "fixture-a", manifest.ID); err == nil {
		t.Fatal("unknown required target published")
	}
	current := f.pin(t)
	if current.Publication().Reference() != captured.Publication().Reference() {
		t.Fatal("staging changed logical publication")
	}
	denseDocs, secondaryDocs := f.read(t, current, old, next, faq)
	assertRevision(t, denseDocs, "r1", 2)
	assertRevision(t, secondaryDocs, "r1", 2)
	if _, err := f.executor.Reconcile(ctx, "fixture-a", manifest.ID, target); err != nil {
		t.Fatal(err)
	}
	if f.secondary.calls != before+1 {
		t.Fatal("reconciliation blindly repeated write")
	}
	if _, err := f.executor.Publish(ctx, "fixture-a", manifest.ID); err != nil {
		t.Fatal(err)
	}
	// Assert: one logical swap selects r2 in both leaves, captured read remains r1.
	denseDocs, secondaryDocs = f.read(t, f.pin(t), old, next, faq)
	assertRevision(t, denseDocs, "r2", 1)
	assertRevision(t, secondaryDocs, "r2", 1)
	denseDocs, secondaryDocs = f.read(t, captured, old, next, faq)
	assertRevision(t, denseDocs, "r1", 2)
	assertRevision(t, secondaryDocs, "r1", 2)
}

func TestActualJointStaleSourceWriterCannotPublish(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		t.Run("dense+"+target, func(t *testing.T) { staleCase(t, target) })
	}
}

func prepareStages(t *testing.T, f *fixture, manifest lifecycle.Manifest, input batch) {
	t.Helper()
	ctx := context.Background()
	if _, err := f.executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{"dense", f.target} {
		if _, err := f.executor.Stage(ctx, "fixture-a", manifest.ID, target, input); err != nil {
			t.Fatal(err)
		}
	}
}

func staleCase(t *testing.T, target string) {
	t.Helper()
	// Arrange: two fully staged writers expect the same published source revision.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	second := sourceBatch("policy", "r2", []string{"p3"})
	third := sourceBatch("policy", "r3", []string{"p4"})
	prepareStages(t, f, plan("policy-r2", "policy-r1", target, second), second)
	prepareStages(t, f, plan("policy-r3", "policy-r1", target, third), third)
	ctx := context.Background()
	// Act: the later writer wins source CAS, the stale writer must not replace it.
	if _, err := f.executor.Publish(ctx, "fixture-a", "policy-r3"); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(ctx, "fixture-a", "policy-r2"); !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("stale writer replaced source publication", err)
	}
	// Assert: both actual targets select only r3 plus unchanged faq.
	denseDocs, secondaryDocs := f.read(t, f.pin(t), old, second, third, faq)
	assertRevision(t, denseDocs, "r3", 1)
	assertRevision(t, secondaryDocs, "r3", 1)
}

func TestActualJointTombstoneAndCleanupPreserveFAQ(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		t.Run("dense+"+target, func(t *testing.T) { cleanupCase(t, target) })
	}
}

func cleanupCase(t *testing.T, target string) {
	t.Helper()
	// Arrange: shared publication inventory contains policy and unrelated faq.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	captured := f.pin(t)
	deleted := plan("deleted", "policy-r1", target, old)
	deleted.Identity.Revision = "r2"
	deleted.Tombstone, deleted.Targets = true, nil
	ctx := context.Background()
	if _, err := f.executor.Prepare(ctx, deleted); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(ctx, "fixture-a", "deleted"); err != nil {
		t.Fatal(err)
	}
	// Act/Assert: both new reads observe the barrier before physical removal.
	denseDocs, secondaryDocs := f.read(t, f.pin(t), old, faq)
	assertRevision(t, denseDocs, "r1", 0)
	assertRevision(t, secondaryDocs, "r1", 0)
	denseDocs, secondaryDocs = f.read(t, captured, old, faq)
	assertRevision(t, denseDocs, "r1", 2)
	assertRevision(t, secondaryDocs, "r1", 2)
	cleaner := newCleaner(t, f)
	if _, err := cleaner.Begin(ctx, "fixture-a", "deleted"); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"dense", target} {
		if _, err := cleaner.Attempt(ctx, "fixture-a", "deleted", "policy-r1", name, false); err != nil {
			t.Fatal(err)
		}
	}
	denseDocs, secondaryDocs = f.read(t, f.pin(t), old, faq)
	assertRevision(t, denseDocs, "r1", 0)
	assertRevision(t, secondaryDocs, "r1", 0)
	oldDense, err := f.dense.Retrieve(
		ctx,
		retrieval.Query[densefs.Intent]{
			Read:    captured,
			Intent:  densefs.Intent{Embedding: dense.Embedding{Space: denseSpace(), Vector: []float32{1, 0}}},
			Options: retrieval.RetrieveOptions{TopK: 10},
		},
	)
	if !errors.Is(err, ragy.ErrUnavailable) || oldDense.Len() != 0 {
		t.Fatal("cleaned old read was substituted or partial", err)
	}
}

func newCleaner(t *testing.T, f *fixture) *lifecycle.Cleaner {
	t.Helper()
	var secondary lifecycle.CleanupPort = f.secondary.lexical
	switch f.target {
	case "tensor":
		secondary = f.secondary.tensor
	case "graph":
		secondary = f.secondary.graph
	}
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   f.store,
			Now:     func() time.Time { return f.now },
			Targets: []lifecycle.CleanupRegistration{{Name: "dense", Port: f.dense}, {Name: f.target, Port: secondary}},
			Policy: lifecycle.CleanupPolicy{
				Deadline: time.Minute,
				Backoff:  []time.Duration{time.Second, 2 * time.Second, 4 * time.Second},
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return cleaner
}

func (f *fixture) graphDocs(t *testing.T, read access.Binding, batches []batch) []retrieval.Document[meta] {
	t.Helper()
	var ids []string
	for _, input := range batches {
		for _, node := range input.Graph.Nodes {
			ids = append(ids, node.Value.ID)
		}
	}
	result, err := f.secondary.graph.FindByIDs(
		context.Background(),
		graphmanaged.Request{
			Read:      read,
			HostBasis: "",
			Traversal: graph.TraversalRequest{Seeds: ids, Direction: graph.DirectionOutbound, Depth: 1},
			MaxNodes:  50,
			MaxEdges:  100,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	var docs []retrieval.Document[meta]
	for _, node := range result.Snapshot.Nodes {
		docs = append(docs, retrieval.Document[meta]{ID: node.ID, Content: node.Content, Meta: node.Meta})
	}
	return docs
}
