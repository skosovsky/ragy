package dense

import (
	"context"
	"net/http"
	"strings"

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
type Client struct {
	apiKey string
	space  embedding.Space
	http   *providerhttp.Client
}

func New(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.APIKey) == "" || (cfg.Model != "" && cfg.Model != cfg.Space.Model) {
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
	Model *string     `json:"model"`
	Data  []embedItem `json:"data"`
	Usage *struct {
		PromptTokens     *int64 `json:"prompt_tokens"`
		TotalTokensCheck *int64 `json:"total_tokens"`
	} `json:"usage"`
}

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
		return rootdense.Result{}, err
	}
	return c.materialize(ctx, len(request.Inputs), decoded)
}

func (c *Client) materialize(ctx context.Context, inputCount int, decoded embedResponse) (rootdense.Result, error) {
	if decoded.Model != nil && *decoded.Model != c.space.Model {
		return rootdense.Result{}, ragy.ErrProtocol
	}
	if len(decoded.Data) != inputCount {
		return rootdense.Result{}, ragy.ErrProtocol
	}
	out := rootdense.Result{Embeddings: make([]rootdense.Embedding, inputCount), Usage: unknownUsage()}
	seen := make([]bool, inputCount)
	for _, item := range decoded.Data {
		if err := ctx.Err(); err != nil {
			return rootdense.Result{}, err
		}
		if item.Index == nil || *item.Index < 0 || *item.Index >= len(seen) || seen[*item.Index] {
			return rootdense.Result{}, ragy.ErrProtocol
		}
		value := rootdense.Embedding{Space: c.space, Vector: item.Embedding}
		if err := value.Validate(); err != nil {
			return rootdense.Result{}, ragy.ErrProtocol
		}
		out.Embeddings[*item.Index] = value
		seen[*item.Index] = true
	}
	usage, err := readUsage(decoded)
	if err != nil {
		return rootdense.Result{}, err
	}
	out.Usage = usage
	return out, ctx.Err()
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
