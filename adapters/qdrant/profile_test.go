package qdrant

import (
	"errors"
	"math"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/retrieval"
)

func fixtureSpace() embedding.Space { return contracttest.DenseSpace() }
func TestProfileRejectsIncompatibleVectorsBeforeIO(t *testing.T) {
	for _, mode := range []string{"model", "revision", "configuration", "identity", "dimension", "metric", "nan", "inf", "zero"} {
		t.Run(mode, func(t *testing.T) {
			// Arrange.
			port := &fakeClient{}
			store, err := New[contracttest.StructMeta](
				port,
				Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t), Space: fixtureSpace()},
				contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
			)
			if err != nil {
				t.Fatal(err)
			}
			space := fixtureSpace()
			vector := []float32{1}
			switch mode {
			case "model":
				space.Model = "other"
			case "revision":
				space.ModelRevision = "other"
			case "configuration":
				space.Configuration = "other"
			case "identity":
				space.VectorSpace = "other"
			case "dimension":
				space.Dimension = 2
				vector = []float32{1, 0}
			case "metric":
				space.Metric = embedding.Dot
			case "nan":
				vector[0] = float32(math.NaN())
			case "inf":
				vector[0] = float32(math.Inf(1))
			case "zero":
				vector[0] = 0
			}
			// Act.
			out, readErr := store.Retrieve(
				t.Context(),
				retrieval.Query[struct{}]{
					Read:    retrieval.UnrestrictedRead(),
					Options: retrieval.RetrieveOptions{TopK: 1, Vector: vector, Space: space},
				},
			)
			writeErr := store.Upsert(
				t.Context(),
				[]dense.Record[contracttest.StructMeta]{{ID: "doc", Content: "original", Vector: vector, Space: space}},
			)
			// Assert.
			if !errors.Is(readErr, ragy.ErrInvalidArgument) || !errors.Is(writeErr, ragy.ErrInvalidArgument) ||
				out == nil ||
				!out.IsEmpty() {
				t.Fatal("incompatible vector accepted", readErr, writeErr)
			}
			if port.cond != nil || port.upsertCalls != 0 {
				t.Fatal("invalid profile reached remote store")
			}
		})
	}
}
func TestConstructorRequiresSupportedHostSpace(t *testing.T) {
	for _, space := range []embedding.Space{{}, func() embedding.Space { s := fixtureSpace(); s.Metric = embedding.SquaredL2; return s }()} {
		// Arrange.
		port := &fakeClient{}
		// Act.
		_, err := New[contracttest.StructMeta](
			port,
			Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t), Space: space},
			contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
		)
		// Assert.
		if err == nil {
			t.Fatal("unsupported host profile accepted")
		}
	}
}
func TestNativeDotScorePreservesMetricAndSign(t *testing.T) {
	// Arrange: the host provisions a Dot collection, whose score is already a similarity.
	space := fixtureSpace()
	space.Metric = embedding.Dot
	port := &fakeClient{searchPoints: []Point{{ID: "doc", Content: "original", Score: -7}}}
	store, err := New[contracttest.StructMeta](
		port,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t), Space: space},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act: non-unit vector is valid for raw dot.
	err = store.Upsert(
		t.Context(),
		[]dense.Record[contracttest.StructMeta]{{ID: "doc", Content: "original", Space: space, Vector: []float32{4}}},
	)
	out, readErr := store.Retrieve(
		t.Context(),
		retrieval.Query[struct{}]{
			Read:    retrieval.UnrestrictedRead(),
			Options: retrieval.RetrieveOptions{TopK: 1, Space: space, Vector: []float32{3}},
		},
	)
	// Assert.
	if err != nil || readErr != nil || out.Len() != 1 {
		t.Fatal(err, readErr)
	}
	if out.Documents()[0].Score != -7 || out.Documents()[0].ScoreSemantics != "dense.dot" ||
		port.upsertPoints[0].Vector[0] != 4 ||
		store.Space() != space {
		t.Fatal("native score/profile/vector changed")
	}
}
