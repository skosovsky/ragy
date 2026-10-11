package dense_test

import (
	"context"
	"errors"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	client "github.com/skosovsky/ragy/adapters/openai/dense"
	root "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

func ExampleClient_Embed() {
	sdk := openai.NewClient(option.WithAPIKey("host-credential"), option.WithMaxRetries(0))
	encoder, err := client.New(client.Config{SDK: &sdk, Space: embedding.Space{
		Model:         "text-embedding-3-small",
		ModelRevision: "host-pinned",
		Configuration: "host-preprocessing-v1",
		VectorSpace:   "query-document-pair",
		Dimension:     2,
		Metric:        embedding.Dot,
	}, Limits: embedding.Limits{MaxInputs: 2}})
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
