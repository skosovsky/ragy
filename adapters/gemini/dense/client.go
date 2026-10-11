// Package dense implements Gemini text embeddings over batchEmbedContents.
package dense

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/gemini/internal/binding"
	rootdense "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

const DefaultBaseURL = binding.DefaultBaseURL

// Config declares authoritative host space, optional equal Model and finite limits.
type Config = binding.Config

// Client supports concurrent calls when the host SDK client does.
type Client struct{ binding *binding.Client }

// New validates the borrowed SDK, declared identity and supported profile.
func New(cfg Config) (*Client, error) {
	client, err := binding.New(cfg)
	if err != nil {
		return nil, err
	}
	return &Client{binding: client}, nil
}

// Space returns the host attestation without asserting remote revision.
func (c *Client) Space() embedding.Space { return c.binding.Space() }

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
	if err := c.binding.ValidateTexts(ctx, request.Inputs); err != nil {
		return rootdense.Result{}, err
	}
	items := make([]binding.Item, len(request.Inputs))
	for i, text := range request.Inputs {
		if err := ctx.Err(); err != nil {
			return rootdense.Result{}, err
		}
		items[i] = c.binding.Item([]binding.Part{{Text: text, InlineData: nil}}, request.Purpose)
	}
	return c.binding.Embed(ctx, items)
}

var _ rootdense.Embedder = (*Client)(nil)
