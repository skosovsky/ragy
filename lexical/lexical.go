// Package lexical provides text-search contracts.
package lexical

import (
	"context"
	"fmt"
	"strings"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// Request is a typed lexical-search request.
type Request struct {
	Text   string
	Filter filter.IR
	Page   *ragy.Page
}

// Validate checks request invariants.
func (r Request) Validate() error {
	if strings.TrimSpace(r.Text) == "" {
		return fmt.Errorf("%w: lexical request text", ragy.ErrEmptyText)
	}

	if err := filter.ValidateIR(r.Filter); err != nil {
		return err
	}

	return r.Page.Validate()
}

// Searcher executes lexical search.
type Searcher interface {
	Search(ctx context.Context, req Request) ([]ragy.Document, error)
	Schema() filter.Schema
}
