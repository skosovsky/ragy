package pgvector

import (
	"fmt"

	"github.com/jackc/pgx/v5"
)

// sanitizedIdent wraps a single SQL identifier segment validated by [regexp fieldSanitize] before sanitize.
type sanitizedIdent string

// sanitizeIdent returns a safe SQL identifier fragment using pgx.Identifier.Sanitize.
func sanitizeIdent(raw string) (sanitizedIdent, error) {
	if !fieldSanitize.MatchString(raw) {
		return "", fmt.Errorf("pgvector: invalid identifier %q", raw)
	}
	s := pgx.Identifier{raw}.Sanitize()
	return sanitizedIdent(s), nil
}
