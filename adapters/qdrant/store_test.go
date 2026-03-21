package qdrant

import (
	"context"
	"testing"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

func TestStore_Search_SparseOnlyNotSupported(t *testing.T) {
	s := New(nil)
	_, err := s.Search(context.Background(), ragy.SearchRequest{})
	require.NoError(t, err)
	_, err = s.Search(context.Background(), ragy.SearchRequest{SparseVector: map[uint32]float32{1: 1}})
	require.Error(t, err)
	require.ErrorIs(t, err, ragy.ErrSparseVectorNotSupported)
}

func TestStore_Stream_SparseOnlyNotSupported(t *testing.T) {
	s := New(nil)
	n := 0
	for _, err := range s.Stream(context.Background(), ragy.SearchRequest{SparseVector: map[uint32]float32{1: 1}}) {
		require.Error(t, err)
		require.ErrorIs(t, err, ragy.ErrSparseVectorNotSupported)
		n++
	}
	assert.Equal(t, 1, n)
}

func TestScoredPointToDocument(t *testing.T) {
	sp := &qdrant.ScoredPoint{
		Id: qdrant.NewIDNum(42),
		Payload: map[string]*qdrant.Value{
			"content": {Kind: &qdrant.Value_StringValue{StringValue: "hello"}},
			ragyIDPayloadKey: {
				Kind: &qdrant.Value_StringValue{StringValue: "business-id-1"},
			},
			"tenant": {Kind: &qdrant.Value_StringValue{StringValue: "t1"}},
		},
		Score: 0.5,
	}
	doc := scoredPointToDocument(sp)
	assert.Equal(t, "business-id-1", doc.ID)
	assert.Equal(t, "hello", doc.Content)
	assert.InDelta(t, float32(0.5), doc.Score, 1e-6)
	assert.InDelta(t, logisticConfidence(0.5), doc.Confidence, 1e-9)
	assert.GreaterOrEqual(t, doc.Confidence, 0.0)
	assert.LessOrEqual(t, doc.Confidence, 1.0)
	assert.Equal(t, "t1", doc.Metadata["tenant"])
}

func TestLogisticConfidence_Bounds(t *testing.T) {
	assert.InEpsilon(t, 1.0, logisticConfidence(logisticClamp), 1e-8)
	low := logisticConfidence(-logisticClamp)
	assert.Less(t, low, 1e-7)
	assert.GreaterOrEqual(t, logisticConfidence(0), 0.0)
	assert.LessOrEqual(t, logisticConfidence(0), 1.0)
}

func TestBuildQdrantFilter_AST(t *testing.T) {
	t.Run("Eq", func(t *testing.T) {
		f := buildQdrantFilter(filter.Equal("tenant", "a"))
		require.NotNil(t, f)
		assert.NotEmpty(t, f.GetMust())
	})
	t.Run("And", func(t *testing.T) {
		f := buildQdrantFilter(filter.All(
			filter.Equal("a", "1"),
			filter.Equal("b", "2"),
		))
		require.NotNil(t, f)
		assert.NotEmpty(t, f.GetMust())
	})
	t.Run("Or", func(t *testing.T) {
		f := buildQdrantFilter(filter.Any(
			filter.Equal("x", "1"),
			filter.Equal("y", "2"),
		))
		require.NotNil(t, f)
		assert.NotEmpty(t, f.GetShould())
	})
	t.Run("Not", func(t *testing.T) {
		f := buildQdrantFilter(filter.Inverse(filter.Equal("z", "bad")))
		require.NotNil(t, f)
		assert.NotEmpty(t, f.GetMustNot())
	})
	t.Run("In", func(t *testing.T) {
		f := buildQdrantFilter(filter.OneOf("role", "a", "b"))
		require.NotNil(t, f)
		assert.NotEmpty(t, f.GetShould())
	})
}

func TestBuildQdrantFilter_Nil(t *testing.T) {
	assert.Nil(t, buildQdrantFilter(nil))
}

func TestValueToAny_Nil(t *testing.T) {
	assert.Nil(t, valueToAny(nil))
}

func TestAnyToFloat64Ptr(t *testing.T) {
	x := 3.14
	p := anyToFloat64Ptr(x)
	require.NotNil(t, p)
	assert.InDelta(t, 3.14, *p, 1e-9)
}

func TestFilterValueToString(t *testing.T) {
	assert.Equal(t, "x", filterValueToString("x"))
	assert.Equal(t, "42", filterValueToString(42))
}

func TestIdToUint64_Deterministic(t *testing.T) {
	a := idToUint64("same")
	b := idToUint64("same")
	assert.Equal(t, a, b)
	assert.NotEqual(t, idToUint64("other"), a)
}
