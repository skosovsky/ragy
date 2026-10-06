package lexical

import (
	"context"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/readfailure"
	"github.com/skosovsky/ragy/retrieval"
)

// snapshotCodec is temporary during owned index construction. It is never cached
// or retained in a published snapshot and does not capture authority decisions.
type snapshotCodec[TMeta any] struct {
	ctx   context.Context
	read  access.Binding
	codec retrieval.MetadataCodec[TMeta]
}

func (c snapshotCodec[TMeta]) Encode(meta TMeta) (filter.RawAttributes, error) {
	if err := c.read.Check(c.ctx); err != nil {
		return nil, err
	}
	attrs, err := c.codec.Encode(meta)
	if gateErr := c.read.Check(c.ctx); gateErr != nil {
		return nil, readfailure.Join(gateErr, err)
	}
	return attrs, err
}

func (c snapshotCodec[TMeta]) Decode(attrs filter.RawAttributes) (TMeta, error) {
	var zero TMeta
	if err := c.read.Check(c.ctx); err != nil {
		return zero, err
	}
	meta, err := c.codec.Decode(attrs)
	if gateErr := c.read.Check(c.ctx); gateErr != nil {
		return zero, readfailure.Join(gateErr, err)
	}
	return meta, err
}
