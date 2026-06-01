// Package lexical provides typed lexical retrieval capability contracts.
package lexical

import (
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

// Backend is a typed lexical retrieval backend that ranks documents by query text.
type Backend[TMeta any] interface {
	retrieval.Backend[TMeta]
	Schema() filter.Schema
	LexicalBackend()
}
