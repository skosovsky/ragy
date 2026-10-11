// Package multimodal implements Gemini Embedding 2 text and inline image inputs.
package multimodal

import (
	"context"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/gemini/internal/binding"
	"github.com/skosovsky/ragy/embedding"
	root "github.com/skosovsky/ragy/multimodal"
)

const DefaultBaseURL = binding.DefaultBaseURL
const maxImages = 6

// Config declares authoritative host space, optional equal Model and finite limits.
type Config = binding.Config

// Client supports concurrent calls when the host SDK client does.
type Client struct{ binding *binding.Client }

// New validates the borrowed SDK, declared identity and supported profile.
func New(cfg Config) (*Client, error) {
	if cfg.Space.Model != "gemini-embedding-2" {
		return nil, ragy.ErrUnsupported
	}
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
	items := make([]binding.Item, len(request.Inputs))
	for i, input := range request.Inputs {
		parts := make([]binding.Part, len(input.Parts))
		for j, part := range input.Parts {
			if err := ctx.Err(); err != nil {
				return root.Result{}, err
			}
			if part.Kind == root.PartText {
				parts[j] = binding.Part{Text: part.Text, InlineData: nil}
			} else {
				parts[j] = binding.Part{Text: "", InlineData: &binding.InlineData{MIME: part.MIME, Data: part.Bytes}}
			}
		}
		items[i] = c.binding.Item(parts, request.Purpose)
	}
	return c.binding.Embed(ctx, items)
}
func (c *Client) validateInputs(ctx context.Context, inputs []root.Input) error {
	limits := c.binding.Limits()
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
