// Command local-bm25 demonstrates a complete local typed retrieval setup.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

const resultLimit = 3

type docMeta struct {
	Tenant string `json:"tenant"`
}

func main() {
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	schemaBuilder := filter.NewSchema()
	tenant, err := schemaBuilder.String("tenant")
	if err != nil {
		return fmt.Errorf("declare tenant: %w", err)
	}

	schema, err := schemaBuilder.Build()
	if err != nil {
		return fmt.Errorf("finalize schema: %w", err)
	}

	index, err := lexical.NewBM25Index[docMeta](schema, lexical.Config[docMeta]{
		SearchFields: []string{"content"},
	}, nil, nil)
	if err != nil {
		return fmt.Errorf("create BM25 index: %w", err)
	}

	if err = index.Index([]retrieval.Document[docMeta]{
		{ID: "reset", Content: "Reset your password from account settings.", Meta: docMeta{Tenant: "acme"}},
		{ID: "other", Content: "Reset your password using the admin portal.", Meta: docMeta{Tenant: "other"}},
	}); err != nil {
		return fmt.Errorf("index documents: %w", err)
	}

	builder, err := filter.NewBuilder(schema)
	if err != nil {
		return fmt.Errorf("create filter builder: %w", err)
	}

	condition, err := filter.Eq(builder, tenant, "acme").Build()
	if err != nil {
		return fmt.Errorf("build tenant filter: %w", err)
	}

	pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, docMeta, retrieval.NoExecutionMeta]().
		WithRoot(retrieval.BackendNode[struct{}, docMeta, retrieval.NoExecutionMeta]{Backend: index}).
		Build()
	if err != nil {
		return fmt.Errorf("build pipeline: %w", err)
	}

	result, retrieveErr := pipeline.Execute(ctx, retrieval.Query[struct{}]{
		// This explicit live read is suitable for local onboarding. The tenant
		// filter is search selection, not an authorization policy or snapshot.
		Read: retrieval.UnrestrictedRead(),
		Text: "reset password",
		Options: retrieval.RetrieveOptions{
			TopK: resultLimit, Filters: condition,
		},
	})

	return display(result, retrieveErr)
}

func display(result retrieval.RetrievalResult[docMeta, retrieval.NoExecutionMeta], err error) error {
	// Protection wins even when joined with another error: never show payload.
	if access.IsProtectionFailure(err) {
		return errors.New("read protection failed; no documents displayed")
	}

	if err != nil {
		_, partial := retrieval.AsPartialFailure[docMeta](err)
		if !partial && result.Len() == 0 {
			return errors.New("retrieval failed; no documents displayed")
		}

		fmt.Println("Partial retrieval: inspect the error before accepting these documents.")
	}

	if result.Coverage.IsPartial() {
		fmt.Println("Partial read coverage: documents do not represent the complete publication.")
	}

	// Use only the returned result. An error-carried ResultSet is not authority.
	for _, document := range result.Documents() {
		fmt.Printf("%s [%s]: %s\n", document.ID, document.Meta.Tenant, document.Content)
	}

	return nil
}
