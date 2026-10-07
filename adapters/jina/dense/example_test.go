package dense_test

import (
	"context"
	"errors"
	"fmt"

	client "github.com/skosovsky/ragy/adapters/jina/dense"
	root "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

func ExampleClient_Embed() {
	encoder, err := client.New(client.Config{
		APIKey: "host-credential",
		Space: embedding.Space{
			Model:         "jina-embeddings-v3",
			ModelRevision: "host-pinned",
			Configuration: "host-preprocessing-v1",
			VectorSpace:   "query-document-pair",
			Dimension:     2,
			Metric:        embedding.Dot,
		},
		Limits: embedding.Limits{MaxInputs: 2},
	})
	if err != nil {
		panic(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Demonstrates error/accounting handling without a provider dispatch.
	out, err := encoder.Embed(ctx, root.Request{Inputs: []string{"question"}, Purpose: embedding.Query})
	if err != nil {
		// Observed usage may be known after rejected materialization; never consume vectors on error.
		fmt.Println(errors.Is(err, context.Canceled), out.Usage.InputTokensKnown, len(out.Embeddings))
	}
	// Output: true false 0
}
