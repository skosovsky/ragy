package dense

import (
	"context"

	"errors"
	"net/http"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/skosovsky/ragy/internal/providerpolicy"

	ragy "github.com/skosovsky/ragy"
	rootdense "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

const smallDimension = 1536
const largeDimension = 3072
const DefaultBaseURL = "https://api.openai.com/v1"

// Config declares the host-owned space and finite local work limits. Model, when
// supplied, must equal Space.Model. No normalization or remote token cap is inferred.
type Config struct {
	SDK    *openai.Client
	Model  string
	Space  embedding.Space
	Limits embedding.Limits
}
type Client struct {
	sdk    *openai.Client
	space  embedding.Space
	limits embedding.Limits
}

// New validates the borrowed SDK client, space, profile and finite local limits.
func New(cfg Config) (*Client, error) {
	if cfg.SDK == nil || (cfg.Model != "" && cfg.Model != cfg.Space.Model) {
		return nil, ragy.ErrInvalidArgument
	}
	if err := cfg.Space.Validate(); err != nil {
		return nil, err
	}
	switch cfg.Space.Model {
	case "text-embedding-3-small":
		if cfg.Space.Dimension > smallDimension {
			return nil, ragy.ErrUnsupported
		}
	case "text-embedding-3-large":
		if cfg.Space.Dimension > largeDimension {
			return nil, ragy.ErrUnsupported
		}
	case "text-embedding-ada-002":
		if cfg.Space.Dimension != smallDimension {
			return nil, ragy.ErrUnsupported
		}
	default:
		return nil, ragy.ErrUnsupported
	}
	limits, err := cfg.Limits.Resolve()
	if err != nil {
		return nil, err
	}
	return &Client{sdk: cfg.SDK, space: cfg.Space, limits: limits}, nil
}

// Space returns the host-declared identity, not a remote revision attestation.
func (c *Client) Space() embedding.Space { return c.space }

// Embed borrows immutable inputs during one exchange and transfers successful
// embeddings to caller. On rejected materialization it returns no embeddings,
// an error and independently valid observed usage.
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
	if err := providerpolicy.ValidateTexts(ctx, request.Inputs, c.limits); err != nil {
		return rootdense.Result{}, err
	}
	//nolint:exhaustruct_v5 // Domain text embeddings omit unrelated SDK options.
	body := openai.EmbeddingNewParams{
		Model: c.space.Model,
		//nolint:exhaustruct_v5 // The SDK union selects ordered text input.
		Input:          openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: request.Inputs},
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	}
	if c.space.Model != "text-embedding-ada-002" {
		body.Dimensions = openai.Int(int64(c.space.Dimension))
	}
	// SDK-L01: public representation cap, not wire retention.
	if err := providerpolicy.Size(body, c.limits.MaxRequestBytes, ragy.ErrInvalidArgument); err != nil {
		return rootdense.Result{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, c.limits.Timeout)
	defer cancel()
	var httpResponse *http.Response
	// Public SDK status observation only; no body capture or parser.
	decoded, err := c.sdk.Embeddings.New(ctx, body, option.WithMaxRetries(0), option.WithResponseInto(&httpResponse))
	if err != nil {
		status := 0
		if httpResponse != nil {
			status = httpResponse.StatusCode
		}
		if apiErr, ok := errors.AsType[*openai.Error](err); ok {
			status = apiErr.StatusCode
		}

		return rootdense.Result{}, providerpolicy.Failure(ctx, err, status)
	}
	if decoded == nil {
		return rootdense.Result{}, ragy.ErrProtocol
	}
	return c.materialize(ctx, len(request.Inputs), *decoded)
}

func (c *Client) materialize(
	ctx context.Context,
	inputCount int,
	decoded openai.CreateEmbeddingResponse,
) (rootdense.Result, error) {
	usage, err := readUsage(decoded)
	if gateErr := ctx.Err(); gateErr != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if err != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, err
	}
	// SDK-L02: use public field metadata; never parse RawJSON to bypass SDK.
	if decoded.JSON.Model.Valid() && decoded.Model != c.space.Model {
		return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	if decoded.JSON.Model.Raw() != "" && !decoded.JSON.Model.Valid() {
		return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	// SDK-L01: public representation cap, not wire retention.
	if err := providerpolicy.Size(decoded, c.limits.MaxResponseBytes, ragy.ErrProtocol); err != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, err
	}
	data := decoded.Data
	if len(data) != inputCount {
		return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	out := rootdense.Result{Embeddings: make([]rootdense.Embedding, inputCount), Usage: usage}
	seen := make([]bool, inputCount)
	for _, item := range data {
		if err := ctx.Err(); err != nil {
			return rootdense.Result{Embeddings: nil, Usage: usage}, err
		}
		if !item.JSON.Embedding.Valid() || !item.JSON.Index.Valid() || item.Index < 0 ||
			item.Index >= int64(len(seen)) ||
			seen[item.Index] {
			return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
		}
		vector := make([]float32, len(item.Embedding))
		for i, v := range item.Embedding {
			vector[i] = float32(v)
		}
		value := rootdense.Embedding{Space: c.space, Vector: vector}
		if err := value.Validate(); err != nil {
			return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
		}
		out.Embeddings[item.Index] = value
		seen[item.Index] = true
	}
	if err := ctx.Err(); err != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, err
	}
	return out, nil
}

var _ rootdense.Embedder = (*Client)(nil)

func readUsage(decoded openai.CreateEmbeddingResponse) (embedding.Usage, error) {
	usage := unknownUsage()
	if (decoded.Usage.JSON.PromptTokens.Raw() != "" && !decoded.Usage.JSON.PromptTokens.Valid()) ||
		(decoded.Usage.JSON.TotalTokens.Raw() != "" && !decoded.Usage.JSON.TotalTokens.Valid()) {
		return embedding.Usage{}, ragy.ErrProtocol
	}
	if decoded.Usage.JSON.TotalTokens.Valid() && decoded.Usage.TotalTokens < 0 {
		return embedding.Usage{}, ragy.ErrProtocol
	}
	if decoded.Usage.JSON.PromptTokens.Valid() {
		usage.InputTokens = decoded.Usage.PromptTokens
		usage.InputTokensKnown = true
	}

	if err := usage.Validate(); err != nil {
		return embedding.Usage{}, err
	}
	return usage, nil
}

func unknownUsage() embedding.Usage {
	return embedding.Usage{InputTokens: 0, InputTokensKnown: false, BilledUnits: 0, BilledUnitsKnown: false}
}
