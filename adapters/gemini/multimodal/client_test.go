package multimodal

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/gemini/internal/wire"
	"github.com/skosovsky/ragy/embedding"
	root "github.com/skosovsky/ragy/multimodal"
)

func profile() embedding.Space {
	return embedding.Space{
		Model:         "gemini-embedding-2",
		ModelRevision: "host-pinned",
		Configuration: "gemini-inline-image-v1",
		VectorSpace:   "test",
		Dimension:     2,
		Metric:        embedding.Cosine,
	}
}

// Official multimodal batch fixture: https://ai.google.dev/gemini-api/docs/embeddings
// verified 2026-10-06. Inline parts are aggregated into one embedding per input.
func TestMultimodalBatchProtocol(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gemini-embedding-2:batchEmbedContents" || r.URL.RawQuery != "" ||
			r.Header.Get("X-Goog-Api-Key") != "key" {
			t.Errorf("request=%s", r.URL.Path)
		}
		var body wire.Batch
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if len(body.Requests) != 2 {
			t.Errorf("items=%d", len(body.Requests))
			return
		}
		item := body.Requests[0]
		if item.Model != "models/gemini-embedding-2" || item.TaskType != "" || item.Dimension != 2 ||
			len(item.Content.Parts) != 2 {
			t.Errorf("item=%+v", item)
			return
		}
		if item.Content.Parts[0].Text != "caption" || item.Content.Parts[1].InlineData == nil ||
			item.Content.Parts[1].InlineData.MIME != "image/png" ||
			string(item.Content.Parts[1].InlineData.Data) != "image" {
			t.Errorf("parts=%+v", item.Content.Parts)
		}
		if body.Requests[1].Content.Parts[0].Text != "title: none | text: text-only" {
			t.Errorf("text-only purpose encoding=%+v", body.Requests[1])
		}
		_, _ = w.Write(
			[]byte(`{"embeddings":[{"values":[1,2]},{"values":[3,4]}],"usageMetadata":{"promptTokenCount":11}}`),
		)
	}))
	defer server.Close()
	client, err := New(Config{APIKey: "key", Space: profile(), BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Embed(
		context.Background(),
		root.Request{
			Purpose: embedding.Document,
			Inputs: []root.Input{
				{
					Parts: []root.Part{
						{Kind: root.PartText, Text: "caption"},
						{Kind: root.PartBytes, MIME: "image/png", Bytes: []byte("image")},
					},
				},
				{Parts: []root.Part{{Kind: root.PartText, Text: "text-only"}}},
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 || result.Embeddings[1].Vector[0] != 3 ||
		result.Embeddings[0].Space != client.Space() ||
		!result.Usage.InputTokensKnown ||
		result.Usage.InputTokens != 11 {
		t.Fatalf("result=%+v", result)
	}
}
func TestMultimodalLocalRejection(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) { calls++ }))
	defer server.Close()
	client, err := New(
		Config{APIKey: "key", Space: profile(), BaseURL: server.URL, Limits: embedding.Limits{MaxInputBytes: 64}},
	)
	if err != nil {
		t.Fatal(err)
	}
	cases := []root.Request{
		{
			Purpose:                 embedding.Query,
			RequireRemoteTokenBound: true,
			Inputs:                  []root.Input{{Parts: []root.Part{{Kind: root.PartText, Text: "hi"}}}},
		},
		{
			Purpose: embedding.Query,
			Inputs:  []root.Input{{Parts: []root.Part{{Kind: root.PartURL, URL: "https://example.com/image"}}}},
		},
		{
			Purpose: embedding.Query,
			Inputs: []root.Input{
				{Parts: []root.Part{{Kind: root.PartBytes, MIME: "audio/mpeg", Bytes: []byte("audio")}}},
			},
		},
		{
			Purpose: embedding.Query,
			Inputs: []root.Input{
				{Parts: []root.Part{{Kind: root.PartBytes, MIME: "image/png", Bytes: make([]byte, 65)}}},
			},
		},
		{
			Purpose: embedding.Query,
			Inputs:  []root.Input{{Parts: []root.Part{{Kind: root.PartText, Text: string([]byte{255})}}}},
		},
		{Purpose: embedding.Query, Inputs: []root.Input{{}}},
	}
	for _, request := range cases {
		if _, err := client.Embed(context.Background(), request); err == nil {
			t.Fatal("expected local rejection")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := client.Embed(ctx, cases[0]); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("calls=%d", calls)
	}
	space := profile()
	space.Model = "gemini-embedding-001"
	if _, err := New(Config{APIKey: "key", Space: space}); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal(err)
	}
}
func TestMultimodalImageLimit(t *testing.T) {
	client, err := New(Config{APIKey: "key", Space: profile()})
	if err != nil {
		t.Fatal(err)
	}
	parts := make([]root.Part, 7)
	for i := range parts {
		parts[i] = root.Part{Kind: root.PartBytes, MIME: "image/jpeg", Bytes: []byte("image")}
	}
	_, err = client.Embed(
		context.Background(),
		root.Request{Purpose: embedding.Document, Inputs: []root.Input{{Parts: parts}}},
	)
	if !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal(err)
	}
}

func TestUntrustedPartKindIsBoundedAndSanitizedBeforeDispatch(t *testing.T) {
	// Arrange.
	calls := 0
	server := httptest.NewServer(
		http.HandlerFunc(
			func(w http.ResponseWriter, _ *http.Request) { calls++; w.WriteHeader(http.StatusInternalServerError) },
		),
	)
	defer server.Close()
	client, err := New(
		Config{APIKey: "secret", Space: profile(), BaseURL: server.URL, Limits: embedding.Limits{MaxInputBytes: 16}},
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []root.PartKind{"PRIVATE_RAW_INPUT_MARKER", root.PartKind(strings.Repeat("PRIVATE_RAW_INPUT_MARKER", 1<<16))} {
		request := root.Request{
			Purpose: embedding.Document,
			Inputs:  []root.Input{{Parts: []root.Part{{Kind: kind, Text: "hello"}}}},
		}
		// Act.
		_, err = client.Embed(context.Background(), request)
		// Assert.
		if !errors.Is(err, ragy.ErrInvalidArgument) || err.Error() != ragy.ErrInvalidArgument.Error() {
			t.Fatalf("untrusted kind error leaked or grew: %v", err)
		}
	}
	if calls != 0 {
		t.Fatalf("calls=%d, want none", calls)
	}
}
