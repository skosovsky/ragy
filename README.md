# ragy

Ragy is a typed, capability-specific retrieval toolkit for host-defined business types. It provides filters, dense/lexical/tensor/graph retrieval, explicit composition and bounded source-aware recipes. Provider/storage adapters are separate modules. Hosts own IAM, original source retention, models/prompts, native drivers, billing, agent/tool loops and scheduling.

## Install and run locally

The core module requires Go **1.27.1 or later** (`go.mod`). CI compiler and linter versions are pinned in `.github/workflows/ci.yml`; record the actual compiler/profile for your deployment. The core has no external runtime dependency. Choose a reviewed module version when reproducibility is required.

```sh
mkdir ragy-demo
cd ragy-demo
go mod init example.com/ragy-demo
go get github.com/skosovsky/ragy
# Save the complete Go example below as main.go:
GOWORK=off go run .
```

From a repository checkout, run `GOWORK=off go run ./examples/local-bm25`. It prints `reset [acme]: Reset your password from account settings.` No external service or provider credential is needed.

This example defines typed metadata, finalizes a filter schema, indexes local documents and selects an explicit Read binding. The tenant predicate selects search results; production authority must come from a host-approved scoped binding. Unrestricted live reads do not establish IAM or snapshot consistency.

```go
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
```

Inspect protection first and use only the returned ResultSet for partial delivery. Read coverage and error both matter. Do not recover documents from an error-carried diagnostic set. See [errors and recovery](docs/errors-and-recovery.md).

## Public contracts

| Topic | Guide |
|---|---|
| Composition, planning, routing, scores/fusion, artifact handoff | [Integration](docs/integration.md) |
| Typed metadata, borrowed/owned captures, cooperative callbacks | [Ownership](docs/ownership.md) |
| Calls/tokens/rows/bytes/candidates, defaults and admission boundaries | [Limits](docs/limits.md) |
| Protection, partial delivery, unknown outcomes and replay | [Errors and recovery](docs/errors-and-recovery.md) |
| Lifecycle capture, generation/publication, pins and cleanup | [Lifecycle](lifecycle/README.md) |
| Retained source authority, layout and chunk projection | [Source](source/README.md), [layout](layout/README.md), [chunking](chunking/README.md) |
| Graph extraction/resolution/materialization and summaries | [Graph ingestion](graphingest/README.md), [graph summary](recipe/graphsummary/README.md) |
| Local BM25, managed snapshots and persistent retrieval | [Lexical](lexical/README.md), [managed lexical](lexical/managed/README.md), [dense](dense/persistent/README.md), [tensor](tensor/README.md) |
| Provider space identity, limits, transport and usage | [Embedding/providers](embedding/README.md) |
| Resource ledger and bounded recipes | [Budget](recipe/budget/README.md), [recipe](recipe/README.md) |
| Payload-free synchronous instrumentation | [Observation](observation/README.md) |
| Implemented contracts versus verified local/wire/live/quality profiles | [Capabilities](docs/capabilities.md) |
| Host/library responsibilities | [Architecture](docs/architecture.md) |
| Exact release source/modules/tags and recovery | [Release runbook](docs/release/runbook.md) |
| License, versioning, contribution and security inventory | [Project policies](docs/project-policies.md) |

Business metadata stays `TMeta`. Private adapter JSON maps and `filter.RawAttributes` are wire representations. A capability declaration must be supported by the adapter/host profile; it is not an authorization sandbox. Native scores need declared comparable scales; dense/lexical fusion belongs in composition, not inside storage adapters.

Historical acceptance percentages and scripted quality results do not establish production readiness. [Current verification scope](docs/capabilities.md) explains the evidence classes and conformance-helper limits. The root license and private security channel remain explicit owner decisions; no license is selected by this remediation.
