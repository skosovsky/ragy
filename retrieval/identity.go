package retrieval

import "github.com/skosovsky/ragy/internal/nilvalue"

// Identity describes document identity for result fusion. DocumentID is the storage key;
// MergeKey is the deduplication key and may differ (for example URI from TMeta).
type Identity struct {
	DocumentID string
	MergeKey   string
}

// IdentityResolver maps a document to its identity.
type IdentityResolver[TMeta any] interface {
	Resolve(doc Document[TMeta]) Identity
}

// DocumentIDResolver uses Document.ID as both document and merge key.
type DocumentIDResolver[TMeta any] struct{}

// Resolve implements IdentityResolver.
func (DocumentIDResolver[TMeta]) Resolve(doc Document[TMeta]) Identity {
	return Identity{DocumentID: doc.ID, MergeKey: doc.ID}
}

// DefaultResolver returns resolver or DocumentIDResolver when resolver is nil or typed nil.
func DefaultResolver[TMeta any](resolver IdentityResolver[TMeta]) IdentityResolver[TMeta] {
	if nilvalue.IsNil(resolver) {
		return DocumentIDResolver[TMeta]{}
	}
	return resolver
}
