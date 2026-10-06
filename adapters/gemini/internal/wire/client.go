// Package wire implements the documented Gemini embedding content protocol.
package wire

import (
	"context"
	"net/http"
	"strings"

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
	if strings.TrimSpace(cfg.APIKey) == "" || strings.ContainsAny(cfg.APIKey, "\r\n") {
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
	Model      string `json:"model"`
	Embeddings []struct {
		Values []float32 `json:"values"`
		Shape  []int     `json:"shape"`
	} `json:"embeddings"`
	Usage *struct {
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
		return dense.Result{}, err
	}
	if len(response.Embeddings) != len(items) ||
		(response.Model != "" && response.Model != c.space.Model && response.Model != "models/"+c.space.Model) {
		return dense.Result{}, ragy.ErrProtocol
	}
	result := dense.Result{
		Embeddings: make([]dense.Embedding, len(items)),
		Usage:      embedding.Usage{InputTokens: 0, InputTokensKnown: false, BilledUnits: 0, BilledUnitsKnown: false},
	}
	for i, item := range response.Embeddings {
		if err := ctx.Err(); err != nil {
			return dense.Result{}, err
		}
		if len(item.Shape) > 0 {
			return dense.Result{}, ragy.ErrProtocol
		}
		if err := c.space.ValidateVector(item.Values); err != nil {
			return dense.Result{}, ragy.ErrProtocol
		}
		result.Embeddings[i] = dense.Embedding{Space: c.space, Vector: item.Values}
	}
	if response.Usage != nil && response.Usage.Tokens != nil {
		result.Usage.InputTokensKnown = true
		result.Usage.InputTokens = *response.Usage.Tokens
	}
	if err := result.Usage.Validate(); err != nil {
		return dense.Result{}, err
	}
	return result, ctx.Err()
}
