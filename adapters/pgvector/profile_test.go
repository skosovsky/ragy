package pgvector

import (
	"errors"
	"math"
	"strings"
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
			port := &fakeDB{}
			store, err := New[contracttest.StructMeta](
				port,
				Config[contracttest.StructMeta]{Table: "docs", Schema: emptySchema(t), Space: fixtureSpace()},
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
			if port.query != "" || port.execCalls != 0 {
				t.Fatal("invalid profile reached remote store")
			}
		})
	}
}
func TestConstructorRequiresSupportedHostSpace(t *testing.T) {
	for _, space := range []embedding.Space{{}, func() embedding.Space { s := fixtureSpace(); s.Metric = embedding.Dot; return s }()} {
		// Arrange.
		port := &fakeDB{}
		// Act.
		_, err := New[contracttest.StructMeta](
			port,
			Config[contracttest.StructMeta]{Table: "docs", Schema: emptySchema(t), Space: space},
			contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
		)
		// Assert.
		if err == nil {
			t.Fatal("unsupported host profile accepted")
		}
	}
}
func TestCosineSQLAndNegativeScore(t *testing.T) {
	// Arrange: SQL fixture returns 1 - cosine distance for an opposing vector.
	port := &fakeDB{queryRows: &fakeRows{rows: []fakeRow{{id: "doc", content: "original", relevance: -1}}}}
	store := newStoreEmptySchema(t, port)
	// Act: cosine permits non-unit vectors without library normalization.
	err := store.Upsert(
		t.Context(),
		[]dense.Record[contracttest.StructMeta]{
			{ID: "doc", Content: "original", Space: fixtureSpace(), Vector: []float32{4}},
		},
	)
	out, readErr := store.Retrieve(
		t.Context(),
		retrieval.Query[struct{}]{
			Read:    retrieval.UnrestrictedRead(),
			Options: retrieval.RetrieveOptions{TopK: 1, Space: fixtureSpace(), Vector: []float32{3}},
		},
	)
	// Assert.
	if err != nil || readErr != nil || out.Len() != 1 {
		t.Fatal(err, readErr)
	}
	if out.Documents()[0].Score != -1 || out.Documents()[0].ScoreSemantics != "dense.cosine" ||
		port.execArgs[3].([]float32)[0] != 4 ||
		store.Space() != fixtureSpace() {
		t.Fatal("cosine score/profile/vector changed")
	}
	if !strings.Contains(port.query, "1 - (vector <=> $1) AS relevance") ||
		!strings.Contains(port.query, "ORDER BY vector <=> $1") {
		t.Fatal("SQL does not implement cosine similarity", port.query)
	}
}
