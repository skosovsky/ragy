//go:build darwin || linux

package joint_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"maps"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorfs "github.com/skosovsky/ragy/tensor/persistent"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

func TestActualPersistentCandidateTensorImmutableRecorder(t *testing.T) {
	// Arrange: actual saved dense candidates and persistent MaxSim query.
	p, stages, controls := recordedTensorStages(t)
	candidateIDs := controls.CandidateIDs
	admitted, allowed := tensorEvidencePolicyInputs(p.tensorRead, stages, p.originals)
	input := evidence.Input[meta]{
		Schema:         p.tensor.Schema(),
		Codec:          retrieval.NewJSONCodec[meta](p.tensor.Schema()),
		RetrievalID:    "tensor-record",
		RecipeRevision: p.configuration,
		Outcome:        evidence.Complete,
		Reason:         evidence.NoReason,
		Coverage:       retrieval.CompleteReadCoverage(),
		Required:       []evidence.Field{evidence.SourceField, evidence.ScoreField, evidence.PublicationField},
		Stages:         stages,
		SourceAdmission: func(ctx context.Context, read access.Binding, ref source.Reference) error {
			if gateErr := read.Check(ctx); gateErr != nil {
				return gateErr
			}
			if !admitted[ref] {
				return access.Protect(ragy.ErrUnavailable)
			}
			return nil
		},
	}
	allowed[p.configuration] = true
	policy := evidence.Policy{
		AllowIdentifier: func(_ evidence.IdentifierKind, id string) bool { return allowed[id] },
		AllowNumbers:    true,
		AllowLocation:   func(loc source.Locator) bool { return admitted[loc.Reference] },
	}
	// Act: capture the actual candidate and native reranking stages exactly once.
	record, err := evidence.Capture(t.Context(), p.tensorRead, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = evidence.Decode(wire); err != nil {
		t.Fatal(err)
	}
	snapshot, err := record.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	// Assert: native 2/1/-1 and actual candidate order survive strict immutable export.
	for i, want := range []float64{2, 1, -1} {
		hit := snapshot.Stages[2].Hits[i]
		if hit.Score.State != evidence.ScoreNative || hit.Score.Value == nil || *hit.Score.Value != want ||
			hit.Rank.Value == nil ||
			*hit.Rank.Value != float64(i+1) {
			t.Fatal("native tensor evidence changed", hit)
		}
	}
	if len(snapshot.Stages[0].Hits) != len(candidateIDs) {
		t.Fatal("candidate stage lost")
	}
	missing := input
	missing.Stages = slices.Clone(input.Stages)
	missing.Stages[0].Scores = evidence.Unavailable
	missingRecord, missingErr := evidence.Capture(t.Context(), p.tensorRead, missing, policy)
	if !errors.Is(missingErr, ragy.ErrUnavailable) {
		t.Fatal("required missing score capability accepted", missingErr)
	}
	if _, missingErr = missingRecord.MarshalJSON(); missingErr == nil {
		t.Fatal("missing capability returned record")
	}
	t.Logf("TASK12_EVIDENCE %s", wire)
	assertTensorRunEnvelope(t, p, stages, controls, record)
	originalScore := input.Stages[1].Hits[0].Document.Score
	originalSource := input.Stages[1].Hits[0].Sources[0]
	input.Stages[1].Hits[0].Document.Score = 999
	input.Stages[1].Hits[0].Sources[0].Revision = "mutated"
	after, err := json.Marshal(record)
	if err != nil || !bytes.Equal(wire, after) {
		t.Fatal("record aliases tensor observation", err)
	}
	input.Stages[1].Hits[0].Document.Score = originalScore
	input.Stages[1].Hits[0].Sources[0] = originalSource
	// Retiring one actual source after retrieval fails before export, never substitutes a revision.
	delete(admitted, stages[0].Hits[0].Sources[0])
	rejected, err := evidence.Capture(t.Context(), p.tensorRead, input, policy)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal("retired tensor source exported", err)
	}
	if _, err = rejected.MarshalJSON(); err == nil {
		t.Fatal("rejected tensor export returned record")
	}
}

func tensorObservedStage(name string, result retrieval.ResultSet[meta]) evidence.Stage[meta] {
	stage := evidence.Stage[meta]{
		Name:      name,
		Status:    evidence.StageObserved,
		Scores:    evidence.Observed,
		Sources:   evidence.Observed,
		Judgments: evidence.Unavailable,
	}
	for _, doc := range result.Documents() {
		var locs []source.Locator
		for _, loc := range doc.SourceLocations() {
			if loc.Reference.Representation == "utf8" {
				locs = append(locs, loc)
			}
		}
		doc.SourceSupports = locs
		var refs []source.Reference
		for _, loc := range locs {
			refs = append(refs, loc.Reference)
		}
		stage.Hits = append(stage.Hits, evidence.Hit[meta]{Document: doc, Sources: refs, Locations: locs})
	}
	return stage
}

type recorderProfiles struct {
	dense                 *densefs.Adapter[meta]
	tensor                *tensorfs.Adapter[meta]
	denseRead, tensorRead access.Binding
	originals             map[source.Reference]bool
	configuration         string
}

func recordedTensorStages(t *testing.T) (recorderProfiles, []evidence.Stage[meta], tensor.RerankResult) {
	t.Helper()
	// Actual joint ledger and Ready inventory are published before either read.
	f := newFixture(t, "tensor")
	input := sourceBatch("policy", "r1", []string{"t1", "t2", "t3"})
	input.Dense[0].Value.Vector = []float32{0, 1}
	input.Dense[1].Value.Vector = []float32{1, 0}
	input.Dense[2].Value.Vector = []float32{-1, 0}
	input.Tensor[0].Value.Tensor = tensor.Tensor{{1, 0}, {0, 1}}
	input.Tensor[1].Value.Tensor = tensor.Tensor{{1, 0}}
	input.Tensor[2].Value.Tensor = tensor.Tensor{{-1, 0}}
	bindTensorOriginalMappings(t, &input)
	f.ingest(t, plan("joint-record", "", "tensor", input), input)
	read := f.pin(t)
	p := recorderProfiles{
		dense:         f.dense,
		tensor:        f.secondary.tensor,
		denseRead:     read,
		tensorRead:    read,
		originals:     make(map[source.Reference]bool),
		configuration: fingerprint(input),
	}
	for _, r := range input.Lexical {
		p.originals[r.Reference] = true
	}
	candidates, err := p.dense.Retrieve(
		t.Context(),
		retrieval.Query[densefs.Intent]{
			Read:    read,
			Intent:  densefs.Intent{Embedding: dense.Embedding{Space: denseSpace(), Vector: []float32{1, 0}}},
			Options: retrieval.RetrieveOptions{TopK: 100},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	var refs []source.Reference
	for _, doc := range candidates.Documents() {
		for _, r := range input.Tensor {
			if r.Value.Meta.Artifact == doc.Meta.Artifact {
				refs = append(refs, r.Reference)
			}
		}
	}
	ranked, err := p.tensor.Query(
		t.Context(),
		retrieval.Query[tensorquery.Intent]{
			Read: read,
			Intent: tensorquery.Intent{
				Embedding:       tensor.Embedding{Space: tensorSpace(), Tokens: tensor.Tensor{{1, 0}, {0, 1}}},
				Candidates:      refs,
				CandidateBudget: 100,
			},
			Options: retrieval.RetrieveOptions{TopK: 10},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if ranked.Evidence.CandidateBudget != 100 || len(ranked.Evidence.CandidateIDs) != len(refs) {
		t.Fatal("actual candidate evidence lost")
	}
	requested := make(map[string]bool)
	for _, ref := range refs {
		id, idErr := (source.Locator{Reference: ref, Kind: source.DocumentLocation}).Identity()
		if idErr != nil {
			t.Fatal(idErr)
		}
		requested[id] = true
	}
	for _, id := range ranked.Evidence.CandidateIDs {
		if !requested[id] {
			t.Fatal("actual tensor candidate not requested")
		}
		delete(requested, id)
	}
	if len(requested) != 0 {
		t.Fatal("actual tensor candidate universe differs from admitted dense refs")
	}
	universe := tensorCandidateStage(t, ranked)
	stages := []evidence.Stage[meta]{
		tensorObservedStage("dense-candidates", candidates),
		universe,
		tensorObservedStage("maxsim", ranked.Documents),
	}
	return p, stages, ranked.Evidence
}

func tensorEvidencePolicyInputs(
	read access.Binding,
	stages []evidence.Stage[meta],
	approved map[source.Reference]bool,
) (map[source.Reference]bool, map[string]bool) {
	admitted := make(map[source.Reference]bool)
	maps.Copy(admitted, approved)
	allowed := map[string]bool{
		"tensor-record":                true,
		"saved-vectors":                true,
		read.Snapshot().Identity:       true,
		read.Publication().Reference(): true,
	}
	for _, stage := range stages {
		allowed[stage.Name] = true
		for _, hit := range stage.Hits {
			allowed[hit.Document.ID] = true
			allowed[string(hit.Document.ScoreSemantics)] = true
			for _, ref := range hit.Sources {
				for _, id := range []string{ref.Namespace, ref.Source, ref.Revision, ref.Transformation, ref.Artifact, ref.Representation} {
					allowed[id] = true
				}
			}
		}
	}
	return admitted, allowed
}

func tensorCandidateStage(t *testing.T, result tensorquery.Result[meta]) evidence.Stage[meta] {
	t.Helper()
	byID := make(map[string]retrieval.Document[meta])
	for _, doc := range result.Documents.Documents() {
		byID[doc.ID] = doc
	}
	stage := evidence.Stage[meta]{
		Name:      "tensor-candidate-observations",
		Status:    evidence.StageObserved,
		Scores:    evidence.Observed,
		Sources:   evidence.Observed,
		Judgments: evidence.Unavailable,
	}
	for _, id := range result.Evidence.CandidateIDs {
		doc, ok := byID[id]
		if !ok {
			t.Fatal("candidate metadata not observed in full TopK fixture")
		}
		// Preserve the actual post-query score/rank; candidate enumeration is
		// recorded separately and does not claim a pre-rerank score observation.
		projected := tensorObservedStage("", retrieval.NewResultSet([]retrieval.Document[meta]{doc}, nil))
		stage.Hits = append(stage.Hits, projected.Hits[0])
	}
	return stage
}

func bindTensorOriginalMappings(t *testing.T, input *batch) {
	t.Helper()
	for i, original := range input.Lexical {
		mapping, err := source.OriginalText(source.Locator{
			Reference: original.Reference,
			Kind:      source.TextLocation,
			Span:      source.ByteSpan{Start: 0, End: len(original.Document.Content)},
		}, original.Document.Content)
		if err != nil {
			t.Fatal(err)
		}
		input.Dense[i].SourceMapping = mapping
		input.Tensor[i].SourceMapping = mapping
	}
}
