// Package dense implements Gemini text embeddings over batchEmbedContents.
package dense

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/gemini/internal/wire"
	rootdense "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/internal/providerhttp"
)

const DefaultBaseURL = wire.DefaultBaseURL

type Doer = providerhttp.Doer

// Config declares authoritative host space, optional equal Model and finite limits.
type Config = wire.Config

// Client supports concurrent calls when the host Doer does.
type Client struct{ wire *wire.Client }

// New validates credentials, declared identity and the supported wire profile.
func New(cfg Config) (*Client, error) {
	client, err := wire.New(cfg)
	if err != nil {
		return nil, err
	}
	return &Client{wire: client}, nil
}

// Space returns the host attestation without asserting remote revision.
func (c *Client) Space() embedding.Space { return c.wire.Space() }

// Embed borrows immutable input payloads during one exchange; successful vectors
// transfer to caller. Errors suppress vectors and retain independently known usage.
func (c *Client) Embed(ctx context.Context, request rootdense.Request) (rootdense.Result, error) {
	if err := ctx.Err(); err != nil {
		return rootdense.Result{}, err
	}
	if request.RequireRemoteTokenBound {
		return rootdense.Result{}, ragy.ErrUnsupported
	}
	if err := request.Validate(); err != nil {
		return rootdense.Result{}, err
	}
	if err := c.wire.ValidateTexts(ctx, request.Inputs); err != nil {
		return rootdense.Result{}, err
	}
	items := make([]wire.Item, len(request.Inputs))
	for i, text := range request.Inputs {
		if err := ctx.Err(); err != nil {
			return rootdense.Result{}, err
		}
		items[i] = c.wire.Item([]wire.Part{{Text: text, InlineData: nil}}, request.Purpose)
	}
	return c.wire.Embed(ctx, items)
}

var _ rootdense.Embedder = (*Client)(nil)
