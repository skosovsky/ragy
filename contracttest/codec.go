package contracttest

import (
	"testing"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

// JSONCodec returns a schema-aware metadata codec for contract tests.
func JSONCodec[TMeta any](t *testing.T, schema filter.Schema) retrieval.MetadataCodec[TMeta] {
	t.Helper()
	return retrieval.NewJSONCodec[TMeta](schema)
}
