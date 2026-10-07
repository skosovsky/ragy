// Package wire implements the documented Gemini embedding content protocol.
package wire

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/internal/providerhttp"
)

const maxDimension = 3072
const DefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

type Config struct {
	APIKey     string
	Model      string
	Space      embedding.Space
	BaseURL    string
	HTTPClient providerhttp.Doer
	Limits     embedding.Limits
}
type Client struct {
	http  *providerhttp.Client
	key   string
	space embedding.Space
}

func New(cfg Config) (*Client, error) {
	if providerhttp.ValidateAPIKey(cfg.APIKey) != nil {
		return nil, ragy.ErrInvalidArgument
	}
	if cfg.Model != "" && cfg.Model != cfg.Space.Model {
		return nil, ragy.ErrInvalidArgument
	}
	if err := cfg.Space.Validate(); err != nil {
		return nil, err
	}
	switch cfg.Space.Model {
	case "gemini-embedding-001", "gemini-embedding-2":
	default:
		return nil, ragy.ErrUnsupported
	}
	if cfg.Space.Dimension > maxDimension {
		return nil, ragy.ErrUnsupported
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	transport, err := providerhttp.New(
		providerhttp.Config{BaseURL: cfg.BaseURL, HTTPClient: cfg.HTTPClient, Limits: cfg.Limits},
	)
	if err != nil {
		return nil, err
	}
	return &Client{http: transport, key: cfg.APIKey, space: cfg.Space}, nil
}
func (c *Client) Space() embedding.Space   { return c.space }
func (c *Client) Limits() embedding.Limits { return c.http.Limits() }
func (c *Client) ValidateTexts(ctx context.Context, texts []string) error {
	return c.http.ValidateTexts(ctx, texts)
}

type InlineData struct {
	MIME string `json:"mimeType"`
	Data []byte `json:"data"`
}
type Part struct {
	Text       string      `json:"text,omitempty"`
	InlineData *InlineData `json:"inlineData,omitempty"`
}
type Content struct {
	Parts []Part `json:"parts"`
}
type Item struct {
	Model     string  `json:"model"`
	Content   Content `json:"content"`
	TaskType  string  `json:"taskType,omitempty"`
	Dimension int     `json:"outputDimensionality"`
}
type Batch struct {
	Requests []Item `json:"requests"`
}

func (c *Client) Item(parts []Part, purpose embedding.Purpose) Item {
	item := Item{
		Model:     "models/" + c.space.Model,
		Content:   Content{Parts: parts},
		Dimension: c.space.Dimension,
		TaskType:  "",
	}
	if c.space.Model == "gemini-embedding-001" {
		switch purpose {
		case embedding.Query:
			item.TaskType = "RETRIEVAL_QUERY"
		case embedding.Document:
			item.TaskType = "RETRIEVAL_DOCUMENT"
		case embedding.Similarity:
			item.TaskType = "SEMANTIC_SIMILARITY"
		}
		return item
	}
	onlyText := true
	for _, part := range parts {
		if part.InlineData != nil {
			onlyText = false
		}
	}
	if onlyText {
		prefix := ""
		switch purpose {
		case embedding.Query:
			prefix = "task: search result | query: "
		case embedding.Document:
			prefix = "title: none | text: "
		case embedding.Similarity:
			prefix = "task: sentence similarity | query: "
		}
		item.Content.Parts = append([]Part(nil), parts...)
		item.Content.Parts[0].Text = prefix + item.Content.Parts[0].Text
	}
	return item
}

type Response struct {
	Model      json.RawMessage `json:"model"`
	Embeddings json.RawMessage `json:"embeddings"`
	Usage      *struct {
		Tokens *int64 `json:"promptTokenCount"`
	} `json:"usageMetadata"`
}

func (c *Client) Embed(ctx context.Context, items []Item) (dense.Result, error) {
	var response Response
	if err := c.http.Post(
		ctx,
		"/models/"+c.space.Model+":batchEmbedContents",
		http.Header{"X-Goog-Api-Key": []string{c.key}},
		Batch{Requests: items},
		&response,
	); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			if usage, usageErr := response.usage(); usageErr == nil {
				return dense.Result{Embeddings: nil, Usage: usage}, err
			}
		}
		return dense.Result{}, err
	}
	return c.materialize(ctx, len(items), response)
}

func (c *Client) materialize(ctx context.Context, inputCount int, response Response) (dense.Result, error) {
	usage, err := response.usage()
	if gateErr := ctx.Err(); gateErr != nil {
		return dense.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if err != nil {
		return dense.Result{}, err
	}
	model, present, modelErr := providerhttp.ModelEcho(response.Model)
	if gateErr := ctx.Err(); gateErr != nil {
		return dense.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if modelErr != nil {
		return dense.Result{Embeddings: nil, Usage: usage}, modelErr
	}

	var vectors []struct {
		Values []float32 `json:"values"`
		Shape  []int     `json:"shape"`
	}
	decodeErr := json.Unmarshal(response.Embeddings, &vectors)
	if gateErr := ctx.Err(); gateErr != nil {
		return dense.Result{Embeddings: nil, Usage: usage}, gateErr
	}
	if decodeErr != nil {
		return dense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	if len(vectors) != inputCount ||
		(present && model != "" && model != c.space.Model && model != "models/"+c.space.Model) {
		return dense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
	}
	result := dense.Result{
		Embeddings: make([]dense.Embedding, inputCount),
		Usage:      usage,
	}
	for i, item := range vectors {
		if err := ctx.Err(); err != nil {
			return dense.Result{Embeddings: nil, Usage: usage}, err
		}
		if len(item.Shape) > 0 {
			return dense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
		}
		if err := c.space.ValidateVector(item.Values); err != nil {
			return dense.Result{Embeddings: nil, Usage: usage}, ragy.ErrProtocol
		}
		result.Embeddings[i] = dense.Embedding{Space: c.space, Vector: item.Values}
	}
	if err := ctx.Err(); err != nil {
		return dense.Result{Embeddings: nil, Usage: usage}, err
	}
	return result, nil
}

func (r Response) usage() (embedding.Usage, error) {
	var usage embedding.Usage
	if r.Usage != nil && r.Usage.Tokens != nil {
		usage.InputTokensKnown = true
		usage.InputTokens = *r.Usage.Tokens
	}
	if err := usage.Validate(); err != nil {
		return embedding.Usage{}, err
	}
	return usage, nil
}
