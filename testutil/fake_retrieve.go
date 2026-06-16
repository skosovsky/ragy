package testutil

import (
	"fmt"
	"strings"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

func runFakeRetrieve(
	err error,
	query string,
	opts retrieval.RetrieveOptions,
	vectorRequired bool,
	schema filter.Schema,
	docs []retrieval.Document[contracttest.StructMeta],
	resolver retrieval.IdentityResolver[contracttest.StructMeta],
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	empty := func(retErr error) (retrieval.ResultSet[contracttest.StructMeta], error) {
		return retrieval.NewResultSet[contracttest.StructMeta](nil, resolver), retErr
	}
	if err != nil {
		return empty(err)
	}
	if validateErr := opts.Validate(); validateErr != nil {
		return empty(validateErr)
	}
	if vectorRequired {
		if len(opts.Vector) == 0 {
			return empty(fmt.Errorf("%w: retrieve vector", ragy.ErrEmptyVector))
		}
	} else if strings.TrimSpace(query) == "" {
		return empty(fmt.Errorf("%w: retrieve query", ragy.ErrEmptyText))
	}
	if schemaErr := schema.ValidateSchemaIR(opts.Filters.IR()); schemaErr != nil {
		return empty(schemaErr)
	}

	validated := make([]retrieval.Document[contracttest.StructMeta], 0, len(docs))
	for _, doc := range docs {
		if err := retrieval.ValidateDocument(doc); err != nil {
			rs := retrieval.NewResultSet(validated, resolver)
			return retrieval.PreserveResultOnError(
				rs,
				ragy.WrapProjectionError(err, "fake retrieve validate"),
				resolver,
			)
		}
		validated = append(validated, doc)
	}
	return retrieval.NewResultSet(validated, resolver), nil
}
