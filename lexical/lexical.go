// Package lexical provides typed lexical retrieval capability contracts.
package lexical

import (
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

// Backend is a typed lexical retrieval backend that ranks documents by query text.
type Backend[TMeta any] interface {
	retrieval.Backend[struct{}, TMeta]
	Schema() filter.Schema
	LexicalBackend()
}

// LexicalRetriever is the typed lexical retrieval backend alias.
//
//nolint:revive // intentional API alias matching task9 naming.
type LexicalRetriever[TMeta any] = Backend[TMeta]
