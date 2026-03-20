package gemini

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmbed_EmptyInput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("should not call API for empty input")
	}))
	defer srv.Close()
	emb := New("key", WithBaseURL(srv.URL), WithBatchSize(2))
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
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		}
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		vec := []float32{0.1, 0.2}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embedding": map[string]any{"values": vec},
		})
	}))
	defer srv.Close()
	emb := New("key", WithBaseURL(srv.URL), WithBatchSize(2))

	out, err := emb.Embed(context.Background(), []string{"hello"})
	require.NoError(t, err)
	require.Len(t, out, 1)
	assert.Equal(t, []float32{0.1, 0.2}, out[0])
	assert.Equal(t, 1, callCount)

	callCount = 0
	out, err = emb.Embed(context.Background(), []string{"a", "b"})
	require.NoError(t, err)
	require.Len(t, out, 2)
	assert.Equal(t, 2, callCount)
}

func TestEmbed_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()
	emb := New("key", WithBaseURL(srv.URL), WithBatchesPerMinute(1000000))
	_, err := emb.Embed(context.Background(), []string{"x"})
	require.Error(t, err)
}

func TestEmbedMultimodal_EmptyInput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("should not call API for empty input")
	}))
	defer srv.Close()
	emb := New("key", WithBaseURL(srv.URL))
	out, err := emb.EmbedMultimodal(context.Background(), nil, nil)
	require.NoError(t, err)
	assert.Nil(t, out)
}

func TestEmbedMultimodal_LengthMismatch(t *testing.T) {
	emb := New("key", WithBaseURL("http://localhost"))
	_, err := emb.EmbedMultimodal(context.Background(), []string{"a"}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "len(media)")
}

func TestEmbedMultimodal_TextOnly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		}
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		assert.Len(t, body.Content.Parts, 1)
		assert.Equal(t, "hello", body.Content.Parts[0].Text)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embedding": map[string]any{"values": []float32{0.5}},
		})
	}))
	defer srv.Close()
	emb := New("key", WithBaseURL(srv.URL))
	media := [][]ragy.Media{nil}
	out, err := emb.EmbedMultimodal(context.Background(), []string{"hello"}, media)
	require.NoError(t, err)
	require.Len(t, out, 1)
	assert.Equal(t, []float32{0.5}, out[0])
}

func TestEmbedMultimodal_WithImage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Content struct {
				Parts []struct {
					Text       string `json:"text,omitempty"`
					InlineData *struct {
						MimeType string `json:"mimeType"`
						Data     string `json:"data"`
					} `json:"inlineData,omitempty"`
				} `json:"parts"`
			} `json:"content"`
		}
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		assert.Len(t, body.Content.Parts, 2)
		assert.Equal(t, "symptom", body.Content.Parts[0].Text)
		assert.NotNil(t, body.Content.Parts[1].InlineData)
		assert.Equal(t, "image/jpeg", body.Content.Parts[1].InlineData.MimeType)
		assert.NotEmpty(t, body.Content.Parts[1].InlineData.Data)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embedding": map[string]any{"values": []float32{0.1, 0.2, 0.3}},
		})
	}))
	defer srv.Close()
	emb := New("key", WithBaseURL(srv.URL))
	media := [][]ragy.Media{{
		{MimeType: "image/jpeg", Data: []byte("fake-jpeg-bytes")},
	}}
	out, err := emb.EmbedMultimodal(context.Background(), []string{"symptom"}, media)
	require.NoError(t, err)
	require.Len(t, out, 1)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, out[0])
}
