package multimodal_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"

	"google.golang.org/genai"

	"github.com/skosovsky/ragy/adapters/gemini/multimodal"
	"github.com/skosovsky/ragy/embedding"
	root "github.com/skosovsky/ragy/multimodal"
)

func ExampleClient_Embed() {
	// Inline bytes are synthetic offline fixture data, not a live image certification.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"embeddings":[{"values":[1,2]}]}`)
	}))
	defer server.Close()
	httpClient := &http.Client{
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	attempts := int32(1)
	sdk, err := genai.NewClient(
		context.Background(),
		&genai.ClientConfig{
			APIKey:     "fixture",
			Backend:    genai.BackendGeminiAPI,
			HTTPClient: httpClient,
			HTTPOptions: genai.HTTPOptions{
				BaseURL:      server.URL,
				APIVersion:   "v1beta",
				RetryOptions: &genai.HTTPRetryOptions{Attempts: &attempts},
			},
		},
	)
	if err != nil {
		panic(err)
	}
	space := embedding.Space{
		Model:         "gemini-embedding-2",
		ModelRevision: "host-pinned",
		Configuration: "inline-v1",
		VectorSpace:   "host-pair",
		Dimension:     2,
		Metric:        embedding.Dot,
	}
	client, err := multimodal.New(multimodal.Config{SDK: sdk, Space: space})
	if err != nil {
		panic(err)
	}
	result, err := client.Embed(
		context.Background(),
		root.Request{
			Inputs: []root.Input{
				{
					Parts: []root.Part{
						{Kind: root.PartText, Text: "caption"},
						{Kind: root.PartBytes, MIME: "image/png", Bytes: []byte("fixture")},
					},
				},
			},
			Purpose: embedding.Query,
		},
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(result.Embeddings), result.Usage.InputTokensKnown)
	// Output: 1 false
}
