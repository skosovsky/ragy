package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	openaiapi "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmbed_EmptyInput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("should not call API for empty input")
	}))
	defer srv.Close()
	cfg := openaiapi.DefaultConfig("test")
	cfg.BaseURL = srv.URL + "/v1"
	cfg.HTTPClient = srv.Client()
	client := openaiapi.NewClientWithConfig(cfg)
	emb := NewWithClient(client, WithBatchSize(2))
	out, err := emb.Embed(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, out)
	out, err = emb.Embed(context.Background(), []string{})
	require.NoError(t, err)
	assert.Nil(t, out)
}

func TestEmbed_SingleAndBatch(t *testing.T) {
	var callCount int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		var body struct {
			Input []string `json:"input"`
		}
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		data := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			data[i] = map[string]any{
				"object":    "embedding",
				"embedding": []float32{0.1, 0.2},
				"index":     i,
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data":   data,
			"model":  "text-embedding-3-small",
			"usage":  map[string]any{},
		})
	}))
	defer srv.Close()
	cfg := openaiapi.DefaultConfig("test")
	cfg.BaseURL = srv.URL + "/v1"
	cfg.HTTPClient = srv.Client()
	client := openaiapi.NewClientWithConfig(cfg)
	emb := NewWithClient(client, WithBatchSize(2))

	// Single item
	out, err := emb.Embed(context.Background(), []string{"hello"})
	require.NoError(t, err)
	require.Len(t, out, 1)
	assert.Equal(t, []float32{0.1, 0.2}, out[0])
	assert.Equal(t, 1, callCount)

	// Exactly batch size
	callCount = 0
	out, err = emb.Embed(context.Background(), []string{"a", "b"})
	require.NoError(t, err)
	require.Len(t, out, 2)
	assert.Equal(t, []float32{0.1, 0.2}, out[0])
	assert.Equal(t, []float32{0.1, 0.2}, out[1])
	assert.Equal(t, 1, callCount)

	// More than one batch
	callCount = 0
	out, err = emb.Embed(context.Background(), []string{"1", "2", "3", "4", "5"})
	require.NoError(t, err)
	require.Len(t, out, 5)
	assert.GreaterOrEqual(t, callCount, 2)
}
