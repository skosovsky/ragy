// Package multimodal implements Gemini Embedding 2 text and inline image inputs.
package multimodal

import (
	"context"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/gemini/internal/wire"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/internal/providerhttp"
	root "github.com/skosovsky/ragy/multimodal"
)

const DefaultBaseURL = wire.DefaultBaseURL
const maxImages = 6

type Doer = providerhttp.Doer
type Config = wire.Config
type Client struct{ wire *wire.Client }

func New(cfg Config) (*Client, error) {
	if cfg.Space.Model != "gemini-embedding-2" {
		return nil, ragy.ErrUnsupported
	}
	client, err := wire.New(cfg)
	if err != nil {
		return nil, err
	}
	return &Client{wire: client}, nil
}
func (c *Client) Space() embedding.Space { return c.wire.Space() }
func (c *Client) Embed(ctx context.Context, request root.Request) (root.Result, error) {
	if err := ctx.Err(); err != nil {
		return root.Result{}, err
	}
	if request.RequireRemoteTokenBound {
		return root.Result{}, ragy.ErrUnsupported
	}
	if err := request.Validate(); err != nil {
		return root.Result{}, err
	}
	if err := c.validateInputs(ctx, request.Inputs); err != nil {
		return root.Result{}, err
	}
	items := make([]wire.Item, len(request.Inputs))
	for i, input := range request.Inputs {
		parts := make([]wire.Part, len(input.Parts))
		for j, part := range input.Parts {
			if err := ctx.Err(); err != nil {
				return root.Result{}, err
			}
			if part.Kind == root.PartText {
				parts[j] = wire.Part{Text: part.Text, InlineData: nil}
			} else {
				parts[j] = wire.Part{Text: "", InlineData: &wire.InlineData{MIME: part.MIME, Data: part.Bytes}}
			}
		}
		items[i] = c.wire.Item(parts, request.Purpose)
	}
	return c.wire.Embed(ctx, items)
}
func (c *Client) validateInputs(ctx context.Context, inputs []root.Input) error {
	limits := c.wire.Limits()
	if len(inputs) > limits.MaxInputs {
		return ragy.ErrInvalidArgument
	}
	total := 0
	for _, input := range inputs {
		if len(input.Parts) == 0 || len(input.Parts) > limits.MaxInputBytes {
			return ragy.ErrInvalidArgument
		}
		images := 0
		for _, part := range input.Parts {
			size, err := validateBoundedPart(ctx, part, limits.MaxInputBytes-total)
			if err != nil {
				return err
			}
			total += size
			if part.Kind == root.PartBytes {
				images++
			}
			if images > maxImages {
				return ragy.ErrUnsupported
			}
		}
	}
	return ctx.Err()
}

var _ root.Embedder = (*Client)(nil)

func validatePart(part root.Part) error {
	// Reject untrusted kind before core validation can format its arbitrary value.
	switch part.Kind {
	case root.PartText, root.PartBytes, root.PartURL:
	default:
		return ragy.ErrInvalidArgument
	}
	if err := part.Validate(); err != nil {
		return ragy.ErrInvalidArgument
	}
	switch part.Kind {
	case root.PartText:
		if !utf8.ValidString(part.Text) {
			return ragy.ErrInvalidArgument
		}
	case root.PartBytes:
		if part.MIME != "image/png" && part.MIME != "image/jpeg" {
			return ragy.ErrUnsupported
		}
	case root.PartURL:
		return ragy.ErrUnsupported
	default:
		return ragy.ErrUnsupported
	}
	return nil
}

func validateBoundedPart(ctx context.Context, part root.Part, remaining int) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	total := 0
	for _, size := range []int{len(part.Text), len(part.Bytes), len(part.MIME), len(part.URL)} {
		if size > remaining-total {
			return 0, ragy.ErrInvalidArgument
		}
		total += size
	}
	if err := validatePart(part); err != nil {
		return 0, err
	}
	return total, nil
}
