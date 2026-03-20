package jina

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmbedTensors_EmptyInput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("should not call API for empty input")
	}))
	defer srv.Close()
	emb := New("key", WithBaseURL(srv.URL))
	out, err := emb.EmbedTensors(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, out)
	out, err = emb.EmbedTensors(context.Background(), []string{})
	require.NoError(t, err)
	assert.Nil(t, out)
}

func TestEmbedTensors_SingleAndBatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []string `json:"input"`
		}
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		data := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			data[i] = map[string]any{
				"index":     i,
				"embedding": [][]float32{{0.1, 0.2}, {0.3, 0.4}},
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer srv.Close()
	emb := New("key", WithBaseURL(srv.URL), WithBatchSize(2))

	out, err := emb.EmbedTensors(context.Background(), []string{"hello"})
	require.NoError(t, err)
	require.Len(t, out, 1)
	require.Len(t, out[0], 2)
	assert.Equal(t, []float32{0.1, 0.2}, out[0][0])
	assert.Equal(t, []float32{0.3, 0.4}, out[0][1])

	out, err = emb.EmbedTensors(context.Background(), []string{"a", "b"})
	require.NoError(t, err)
	require.Len(t, out, 2)
	assert.Len(t, out[0], 2)
	assert.Len(t, out[1], 2)
}

func TestEmbedTensors_429NoAdapterRetry(t *testing.T) {
	var attempt int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempt++
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()
	emb := New("key", WithBaseURL(srv.URL))
	out, err := emb.EmbedTensors(context.Background(), []string{"x"})
	require.Error(t, err)
	assert.Nil(t, out)
	assert.Equal(t, 1, attempt)
}

func TestDenseEmbed_EmptyInput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("should not call API for empty input")
	}))
	defer srv.Close()
	d := NewDense("key", DenseWithBaseURL(srv.URL))
	out, err := d.Embed(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, out)
	out, err = d.Embed(context.Background(), []string{})
	require.NoError(t, err)
	assert.Nil(t, out)
}

func TestDenseEmbed_SingleAndBatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []string `json:"input"`
		}
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		data := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			data[i] = map[string]any{
				"index":     i,
				"embedding": []float32{0.1, 0.2, 0.3},
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer srv.Close()
	d := NewDense("key", DenseWithBaseURL(srv.URL), DenseWithBatchSize(2))

	out, err := d.Embed(context.Background(), []string{"hello"})
	require.NoError(t, err)
	require.Len(t, out, 1)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, out[0])

	out, err = d.Embed(context.Background(), []string{"a", "b"})
	require.NoError(t, err)
	require.Len(t, out, 2)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, out[0])
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, out[1])
}
