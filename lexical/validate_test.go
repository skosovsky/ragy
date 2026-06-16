package lexical

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

func TestValidateSearchFieldsRejectsDuplicate(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	err = ValidateSearchFields(schema, []string{"content", "content"})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateSearchFields() error = %v, want invalid argument", err)
	}
}
