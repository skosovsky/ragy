package lexical

import (
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// ValidateSearchFields checks search field names against a finalized schema.
func ValidateSearchFields(schema filter.Schema, fields []string) error {
	seen := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		if field == "" {
			return fmt.Errorf("%w: lexical search field", ragy.ErrInvalidArgument)
		}
		if _, exists := seen[field]; exists {
			return fmt.Errorf("%w: duplicate lexical search field %q", ragy.ErrInvalidArgument, field)
		}
		seen[field] = struct{}{}
		if field == "content" {
			continue
		}
		if err := filter.ValidateIdentifier(field); err != nil {
			return fmt.Errorf("%w: lexical search field %q: %w", ragy.ErrInvalidArgument, field, err)
		}
		if _, ok := schema.Lookup(field); !ok {
			return fmt.Errorf("%w: unknown lexical search field %q", ragy.ErrInvalidArgument, field)
		}
	}
	return nil
}
