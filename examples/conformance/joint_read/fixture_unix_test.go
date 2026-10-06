//go:build darwin || linux

package joint_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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
	probe     *readProbe
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
	probe := &readProbe{}
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
			PayloadReader:   probe,
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
				MaxRecords: 100,
			},
		)
	default:
		secondary.lexical, err = lexicalmanaged.New(
			lexicalmanaged.Config[meta]{
				Namespace: "fixture-a",
				Target:    target,
				Store:     store,
				Schema:    fields,
				BM25:      lexical.Config[meta]{SearchFields: []string{"content"}},
				CloneMeta: cloneMeta,
			},
		)
	}
	if err != nil {
		t.Fatal(err)
	}
	out := &fixture{
		store:     store,
		probe:     probe,
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
			Authority:   access.AuthorityFunc(f.probe.authorize),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return read
}
