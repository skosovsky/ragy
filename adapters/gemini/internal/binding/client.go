// Package binding maps domain inputs onto the official Gemini SDK.
package binding

import (
	"context"
	"errors"
	"net/url"

	"google.golang.org/genai"

	"github.com/skosovsky/ragy/internal/providerpolicy"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

const maxDimension = 3072
const DefaultBaseURL = "https://generativelanguage.googleapis.com/"

type Config struct {
	SDK    *genai.Client
	Model  string
	Space  embedding.Space
	Limits embedding.Limits
}
type Client struct {
	sdk    *genai.Client
	space  embedding.Space
	limits embedding.Limits
}

func New(cfg Config) (*Client, error) {
	if cfg.SDK == nil || cfg.SDK.Models == nil || (cfg.Model != "" && cfg.Model != cfg.Space.Model) {
		return nil, ragy.ErrInvalidArgument
	}
	if cfg.SDK.ClientConfig().Backend != genai.BackendGeminiAPI {
		return nil, ragy.ErrUnsupported
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
	limits, err := cfg.Limits.Resolve()
	if err != nil {
		return nil, err
	}
	return &Client{sdk: cfg.SDK, space: cfg.Space, limits: limits}, nil
}
func (c *Client) Space() embedding.Space   { return c.space }
func (c *Client) Limits() embedding.Limits { return c.limits }
func (c *Client) ValidateTexts(ctx context.Context, texts []string) error {
	return providerpolicy.ValidateTexts(ctx, texts, c.limits)
}

type InlineData struct {
	MIME string
	Data []byte
}
type Part struct {
	Text       string
	InlineData *InlineData
}
type Content struct {
	Parts []Part
}
type Item struct {
	Content  Content
	TaskType string
}

func (c *Client) Item(parts []Part, purpose embedding.Purpose) Item {
	item := Item{
		Content:  Content{Parts: parts},
		TaskType: "",
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

// Embed maps domain parts into the public SDK batch API, with no rebatching.
func (c *Client) Embed(ctx context.Context, items []Item) (dense.Result, error) {
	contents := make([]*genai.Content, len(items))
	for i, item := range items {
		parts := make([]*genai.Part, len(item.Content.Parts))
		for j, part := range item.Content.Parts {
			//nolint:exhaustruct_v5 // Only bounded text and inline-media SDK fields are admitted.
			parts[j] = &genai.Part{Text: part.Text}
			if part.InlineData != nil {
				//nolint:exhaustruct_v5 // No host display name is inferred.
				parts[j].InlineData = &genai.Blob{MIMEType: part.InlineData.MIME, Data: part.InlineData.Data}
			}
		}
		//nolint:exhaustruct_v5 // Embedding content has no chat role.
		contents[i] = &genai.Content{Parts: parts}
	}
	attempts := int32(1)
	//nolint:gosec // Constructor validates positive dimension <= 3072.
	dimension := int32(c.space.Dimension)
	//nolint:exhaustruct_v5 // Vertex-only options are outside this profile.
	cfg := &genai.EmbedContentConfig{
		TaskType:             items[0].TaskType,
		OutputDimensionality: &dimension,
		//nolint:exhaustruct_v5 // Only public retry override; host owns client options.
		HTTPOptions: &genai.HTTPOptions{RetryOptions: &genai.HTTPRetryOptions{Attempts: &attempts}},
	}
	// SDK-L01: representation bound, not exact SDK request wire size.
	if err := providerpolicy.Size(contents, c.limits.MaxRequestBytes, ragy.ErrInvalidArgument); err != nil {
		return dense.Result{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, c.limits.Timeout)
	defer cancel()
	response, err := c.sdk.Models.EmbedContent(ctx, c.space.Model, contents, cfg)
	if err != nil {
		if apiErr, ok := errors.AsType[genai.APIError](err); ok {
			return dense.Result{}, providerpolicy.Failure(ctx, err, apiErr.Code)
		}
		// http.Client wraps dispatch failures in the public url.Error type.
		// Preserve this cause without inventing an HTTP response status.
		if _, ok := errors.AsType[*url.Error](err); ok {
			return dense.Result{}, providerpolicy.Failure(ctx, err, 0)
		}
		class := ragy.ErrProtocol // SDK response conversion/decoding failed.
		if ctxErr := ctx.Err(); ctxErr != nil {
			class = ctxErr
		}
		return dense.Result{}, &providerpolicy.SDKError{Class: class, Cause: err}
	}
	if response == nil {
		return dense.Result{}, ragy.ErrProtocol
	}
	return c.materialize(ctx, len(items), *response)
}

func (c *Client) materialize(
	ctx context.Context,
	inputCount int,
	response genai.EmbedContentResponse,
) (dense.Result, error) {
	// SDK-L03: Gemini API response converter discards usageMetadata/model/shape.
	// No counter inference or alternate envelope parser is permitted.
	if err := ctx.Err(); err != nil {
		return dense.Result{}, err
	}
	if err := providerpolicy.Size(response, c.limits.MaxResponseBytes, ragy.ErrProtocol); err != nil {
		return dense.Result{}, err
	}
	if len(response.Embeddings) != inputCount {
		return dense.Result{}, ragy.ErrProtocol
	}
	result := dense.Result{
		Embeddings: make([]dense.Embedding, inputCount),
		Usage:      embedding.Usage{InputTokens: 0, InputTokensKnown: false, BilledUnits: 0, BilledUnitsKnown: false},
	}
	for i, item := range response.Embeddings {
		if err := ctx.Err(); err != nil {
			return dense.Result{}, err
		}
		if item == nil || c.space.ValidateVector(item.Values) != nil {
			return dense.Result{}, ragy.ErrProtocol
		}
		result.Embeddings[i] = dense.Embedding{Space: c.space, Vector: item.Values}
	}
	if err := ctx.Err(); err != nil {
		return dense.Result{}, err
	}
	return result, nil
}
