package rerank_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"

	cohereclient "github.com/cohere-ai/cohere-go/v2/client"

	"github.com/skosovsky/ragy/adapters/cohere/rerank"
	"github.com/skosovsky/ragy/retrieval"
)

func ExampleClient_RerankWithUsage() {
	// Offline provider fixture; no paid service dispatch.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(
			w,
			`{"results":[{"index":0,"relevance_score":0.9}],"meta":{"billed_units":{"search_units":1}}}`,
		)
	}))
	defer server.Close()
	httpClient := &http.Client{
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	sdk := cohereclient.NewClient(
		cohereclient.WithToken("fixture"),
		cohereclient.WithBaseURL(server.URL),
		cohereclient.WithHTTPClient(httpClient),
		cohereclient.WithMaxAttempts(1),
	)
	ranker, err := rerank.New[struct{}](rerank.Config{SDK: sdk, Model: "fixture-model"})
	if err != nil {
		panic(err)
	}
	docs := retrieval.NewResultSet(
		[]retrieval.Document[struct{}]{{ID: "a", Content: "document"}},
		retrieval.DocumentIDResolver[struct{}]{},
	)
	result, err := ranker.RerankWithUsage(context.Background(), retrieval.UnrestrictedRead(), "query", docs)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Documents.Len(), result.Usage.BilledUnits, result.Usage.BilledUnitsKnown)
	// Output: 1 1 true
}
