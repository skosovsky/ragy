package tensor

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/internal/providerhttp"
	roottensor "github.com/skosovsky/ragy/tensor"
)

const DefaultBaseURL = "https://api.jina.ai/v1"

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
	if cfg.Space.Model != "jina-colbert-v2" || (cfg.Space.Dimension != 64 && cfg.Space.Dimension != 128) {
		return nil, ragy.ErrUnsupported
	}
	if err := roottensor.ValidateSpace(cfg.Space); err != nil {
		return nil, err
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
	Model         string   `json:"model"`
	Input         []string `json:"input"`
	Dimensions    int      `json:"dimensions,omitempty"`
	InputType     string   `json:"input_type"`
	EmbeddingType string   `json:"embedding_type"`
}
type embedItem struct {
	Index      *int        `json:"index"`
	Embeddings [][]float32 `json:"embeddings"`
}
type embedResponse struct {
	Model json.RawMessage `json:"model"`
	Data  json.RawMessage `json:"data"`
	Usage *struct {
		TotalTokens      *int64 `json:"total_tokens"`
		TotalTokensCheck *int64 `json:"prompt_tokens"`
	} `json:"usage"`
}

// Embed borrows immutable inputs during one exchange and transfers successful
// embeddings to caller. On rejected materialization it returns no embeddings,
// an error and independently valid observed usage.
func (c *Client) Embed(ctx context.Context, request roottensor.Request) (roottensor.Result, error) {
	if err := ctx.Err(); err != nil {
		return roottensor.Result{}, err
	}
	if request.RequireRemoteTokenBound {
		return roottensor.Result{}, ragy.ErrUnsupported
	}
	if err := request.Validate(); err != nil {
		return roottensor.Result{}, err
	}
	if err := c.http.ValidateTexts(ctx, request.Inputs); err != nil {
		return roottensor.Result{}, err
	}
	body := embedRequest{
		Model:         c.space.Model,
		Input:         request.Inputs,
		Dimensions:    c.space.Dimension,
		InputType:     "",
		EmbeddingType: "",
	}
	if request.Purpose == embedding.Similarity {
		return roottensor.Result{}, ragy.ErrUnsupported
	}
	body.InputType = string(request.Purpose)
	body.EmbeddingType = "float"
	var decoded embedResponse
	if err := c.http.Post(
		ctx,
		"/multi-vector",
		http.Header{"Authorization": {"Bearer " + c.apiKey}},
		body,
		&decoded,
	); err != nil {
		// Post only returns a context error with populated output after complete
		// JSON admission; earlier transport/read failures leave decoded zero.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			if usage, usageErr := readUsage(decoded); usageErr == nil {
				return roottensor.Result{Embeddings: nil, Usage: usage}, err
			}
		}
		return roottensor.Result{}, err
	}
	return c.materialize(ctx, len(request.Inputs), decoded)
}

func (c *Client) materialize(ctx context.Context, inputCount int, decoded embedResponse) (roottensor.Result, error) {
	usage, err := readUsage(decoded)
	if gateErr := ctx.Err(); gateErr != nil {
		return roottensor.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if err != nil {
		return roottensor.Result{Embeddings: nil, Usage: usage}, err
	}
	model, present, modelErr := providerhttp.ModelEcho(decoded.Model)
	if gateErr := ctx.Err(); gateErr != nil {
		return roottensor.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if modelErr != nil {
		return roottensor.Result{Embeddings: nil, Usage: usage}, modelErr
	}
	if present && model != c.space.Model {
		return roottensor.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	var data []embedItem
	decodeErr := json.Unmarshal(decoded.Data, &data)
	if gateErr := ctx.Err(); gateErr != nil {
		return roottensor.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if decodeErr != nil {
		return roottensor.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	if len(data) != inputCount {
		return roottensor.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	out := roottensor.Result{Embeddings: make([]roottensor.Embedding, inputCount), Usage: usage}
	seen := make([]bool, inputCount)
	for _, item := range data {
		if err := ctx.Err(); err != nil {
			return roottensor.Result{Embeddings: nil, Usage: usage}, err
		}
		if item.Index == nil || *item.Index < 0 || *item.Index >= len(seen) || seen[*item.Index] {
			return roottensor.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
		}
		if len(item.Embeddings) > c.http.Limits().MaxVectorRows {
			return roottensor.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
		}
		value := roottensor.Embedding{Space: c.space, Tokens: item.Embeddings}
		if err := value.Validate(); err != nil {
			return roottensor.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
		}
		out.Embeddings[*item.Index] = value
		seen[*item.Index] = true
	}
	if err := ctx.Err(); err != nil {
		return roottensor.Result{Embeddings: nil, Usage: usage}, err
	}
	return out, nil
}

var _ roottensor.Embedder = (*Client)(nil)

func readUsage(decoded embedResponse) (embedding.Usage, error) {
	usage := unknownUsage()
	if decoded.Usage != nil {
		if decoded.Usage.TotalTokensCheck != nil && *decoded.Usage.TotalTokensCheck < 0 {
			return embedding.Usage{}, ragy.ErrProtocol
		}
		if decoded.Usage.TotalTokens != nil {
			usage.InputTokens = *decoded.Usage.TotalTokens
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
