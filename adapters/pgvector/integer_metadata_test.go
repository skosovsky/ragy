package pgvector

import (
	"testing"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

func TestStoredMetadataPreservesIntegerIdentity(t *testing.T) {
	t.Parallel()
	// Arrange.
	builder := filter.NewSchema()
	if _, err := builder.Int("tenant"); err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	type meta struct {
		Tenant int64 `json:"tenant"`
	}
	codec := retrieval.NewJSONCodec[meta](schema)
	// Act: exercise the actual stored JSON boundary and the metadata codec.
	attrs, parseErr := attributesFromJSON([]byte(`{"tenant":9007199254740993}`))
	got, decodeErr := codec.Decode(attrs)
	// Assert.
	if parseErr != nil || decodeErr != nil || got.Tenant != 9007199254740993 {
		t.Fatalf("attrs=%v got=%v parse=%v decode=%v", attrs, got, parseErr, decodeErr)
	}
}
