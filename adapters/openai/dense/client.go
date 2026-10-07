package dense

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	ragy "github.com/skosovsky/ragy"
	rootdense "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/internal/providerhttp"
)

const smallDimension = 1536
const largeDimension = 3072
const DefaultBaseURL = "https://api.openai.com/v1"

// Doer must honor context and perform no hidden retries or redirects.
type Doer = providerhttp.Doer

// Config declares the host-owned space and finite local work limits. Model, when
// supplied, must equal Space.Model. No normalization or remote token cap is inferred.
type Config struct {
	APIKey     string
	Model      string
	Space      embedding.Space
	Limits     embedding.Limits
	BaseURL    string
	HTTPClient Doer
}

// Client captures configuration and supports concurrent calls when its Doer does.
type Client struct {
	apiKey string
	space  embedding.Space
	http   *providerhttp.Client
}

// New validates credentials, space, supported profile and finite local limits.
func New(cfg Config) (*Client, error) {
	if providerhttp.ValidateAPIKey(cfg.APIKey) != nil || (cfg.Model != "" && cfg.Model != cfg.Space.Model) {
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
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	client, err := providerhttp.New(
		providerhttp.Config{BaseURL: cfg.BaseURL, HTTPClient: cfg.HTTPClient, Limits: cfg.Limits},
	)
	if err != nil {
		return nil, err
	}
	return &Client{apiKey: cfg.APIKey, space: cfg.Space, http: client}, nil
}

// Space returns the host-declared identity, not a remote revision attestation.
func (c *Client) Space() embedding.Space { return c.space }

type embedRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	Dimensions     int      `json:"dimensions,omitempty"`
	EncodingFormat string   `json:"encoding_format"`
}
type embedItem struct {
	Index     *int      `json:"index"`
	Embedding []float32 `json:"embedding"`
}
type embedResponse struct {
	Model json.RawMessage `json:"model"`
	Data  json.RawMessage `json:"data"`
	Usage *struct {
		PromptTokens     *int64 `json:"prompt_tokens"`
		TotalTokensCheck *int64 `json:"total_tokens"`
	} `json:"usage"`
}

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
	if err := c.http.ValidateTexts(ctx, request.Inputs); err != nil {
		return rootdense.Result{}, err
	}
	body := embedRequest{Model: c.space.Model, Input: request.Inputs, Dimensions: c.space.Dimension, EncodingFormat: ""}
	body.EncodingFormat = "float"
	if c.space.Model == "text-embedding-ada-002" {
		body.Dimensions = 0
	}
	var decoded embedResponse
	if err := c.http.Post(
		ctx,
		"/embeddings",
		http.Header{"Authorization": {"Bearer " + c.apiKey}},
		body,
		&decoded,
	); err != nil {
		// Post only returns a context error with populated output after complete
		// JSON admission; earlier transport/read failures leave decoded zero.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			if usage, usageErr := readUsage(decoded); usageErr == nil {
				return rootdense.Result{Embeddings: nil, Usage: usage}, err
			}
		}
		return rootdense.Result{}, err
	}
	return c.materialize(ctx, len(request.Inputs), decoded)
}

func (c *Client) materialize(ctx context.Context, inputCount int, decoded embedResponse) (rootdense.Result, error) {
	usage, err := readUsage(decoded)
	if gateErr := ctx.Err(); gateErr != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if err != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, err
	}
	model, present, modelErr := providerhttp.ModelEcho(decoded.Model)
	if gateErr := ctx.Err(); gateErr != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if modelErr != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, modelErr
	}
	if present && model != c.space.Model {
		return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	var data []embedItem
	decodeErr := json.Unmarshal(decoded.Data, &data)
	if gateErr := ctx.Err(); gateErr != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if decodeErr != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	if len(data) != inputCount {
		return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	out := rootdense.Result{Embeddings: make([]rootdense.Embedding, inputCount), Usage: usage}
	seen := make([]bool, inputCount)
	for _, item := range data {
		if err := ctx.Err(); err != nil {
			return rootdense.Result{Embeddings: nil, Usage: usage}, err
		}
		if item.Index == nil || *item.Index < 0 || *item.Index >= len(seen) || seen[*item.Index] {
			return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
		}
		value := rootdense.Embedding{Space: c.space, Vector: item.Embedding}
		if err := value.Validate(); err != nil {
			return rootdense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
		}
		out.Embeddings[*item.Index] = value
		seen[*item.Index] = true
	}
	if err := ctx.Err(); err != nil {
		return rootdense.Result{Embeddings: nil, Usage: usage}, err
	}
	return out, nil
}

var _ rootdense.Embedder = (*Client)(nil)

func readUsage(decoded embedResponse) (embedding.Usage, error) {
	usage := unknownUsage()
	if decoded.Usage != nil {
		if decoded.Usage.TotalTokensCheck != nil && *decoded.Usage.TotalTokensCheck < 0 {
			return embedding.Usage{}, ragy.ErrProtocol
		}
		if decoded.Usage.PromptTokens != nil {
			usage.InputTokens = *decoded.Usage.PromptTokens
			usage.InputTokensKnown = true
		}
	}
	if err := usage.Validate(); err != nil {
		return embedding.Usage{}, err
	}
	return usage, nil
}

func unknownUsage() embedding.Usage {
	return embedding.Usage{InputTokens: 0, InputTokensKnown: false, BilledUnits: 0, BilledUnitsKnown: false}
}
