// Package structured provides structured completions through a borrowed official SDK.
// Domain schemas, validation, tokenization and prices belong to the host.
package structured

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"math"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/skosovsky/ragy/internal/providerpolicy"

	ragy "github.com/skosovsky/ragy"

	"github.com/skosovsky/ragy/recipe/budget"
)

var (
	ErrRefused    = errors.New("structured completion refused")
	ErrIncomplete = errors.New("structured completion incomplete")
)

// Config uses a host-supplied executable schema validator and exact tokenizer.
// Callbacks must be pure, bounded and safe for concurrent use. CountTokens counts
// the complete serialized provider request, including instructions and schema.
type Config struct {
	SDK              *openai.Client
	Model            string
	Instructions     string
	SchemaName       string
	Schema           json.RawMessage
	MaxRequestBytes  int
	MaxResponseBytes int
	Duration         time.Duration
	Validate         func(json.RawMessage) error
	CountTokens      func([]byte) (uint64, error)
}

type Client[T any] struct {
	config Config
	sdk    *openai.Client
}

type Limits struct {
	InputTokens  uint64
	OutputTokens uint64
}
type Usage struct {
	InputTokens  uint64
	OutputTokens uint64
	Known        bool
}

func New[T any](cfg Config) (*Client[T], error) {
	if cfg.SDK == nil ||
		strings.TrimSpace(cfg.Model) == "" || cfg.Instructions == "" || !validName(cfg.SchemaName) ||
		cfg.MaxRequestBytes <= 0 || cfg.MaxResponseBytes <= 0 || cfg.MaxResponseBytes == math.MaxInt || cfg.Duration <= 0 ||
		cfg.Validate == nil || cfg.CountTokens == nil {
		return nil, ragy.ErrInvalidArgument
	}
	if !validSchema(cfg.Schema, cfg.MaxRequestBytes) {
		return nil, ragy.ErrInvalidArgument
	}
	cfg.Schema = bytes.Clone(cfg.Schema)
	return &Client[T]{config: cfg, sdk: cfg.SDK}, nil
}

func validSchema(schema json.RawMessage, limit int) bool {
	trimmed := bytes.TrimSpace(schema)
	return len(schema) <= limit && len(trimmed) > 0 && trimmed[0] == '{' && validJSON(schema) == nil
}

func validName(value string) bool {
	if len(value) == 0 || len(value) > 64 {
		return false
	}
	for _, char := range value {
		if (char < 'a' || char > 'z') && (char < 'A' || char > 'Z') &&
			(char < '0' || char > '9') && char != '_' && char != '-' {
			return false
		}
	}
	return true
}

func (c *Client[T]) params(input any, outputTokens uint64) (openai.ChatCompletionNewParams, error) {
	if c == nil || outputTokens == 0 || outputTokens > math.MaxInt64 {
		return openai.ChatCompletionNewParams{}, ragy.ErrInvalidArgument
	}
	content, err := json.Marshal(input)
	if err != nil || len(content) > c.config.MaxRequestBytes {
		return openai.ChatCompletionNewParams{}, ragy.ErrInvalidArgument
	}
	//nolint:exhaustruct_v5 // SDK serializer supplies the discriminator; description is host optional.
	format := shared.ResponseFormatJSONSchemaParam{
		//nolint:exhaustruct_v5 // Only the supplied executable schema is sent.
		JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:   c.config.SchemaName,
			Strict: openai.Bool(true),
			Schema: c.config.Schema,
		},
	}
	//nolint:exhaustruct_v5 // Structured port intentionally excludes optional tools/sampling settings.
	return openai.ChatCompletionNewParams{
		Model: c.config.Model,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(c.config.Instructions),
			openai.UserMessage(string(content)),
		},
		//nolint:exhaustruct_v5 // SDK union selects strict JSON schema.
		ResponseFormat:      openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONSchema: &format},
		MaxCompletionTokens: openai.Int(int64(outputTokens)),
		Store:               openai.Bool(false),
	}, nil
}

func (c *Client[T]) body(input any, outputTokens uint64) ([]byte, error) {
	params, err := c.params(input, outputTokens)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(params)
	if err != nil || len(body) > c.config.MaxRequestBytes {
		return nil, ragy.ErrInvalidArgument
	}
	return body, nil
}

// CountInputTokens performs no I/O. The host tokenizer must account for the exact
// model's message framing and schema, rather than approximating by string length.
func (c *Client[T]) CountInputTokens(input any, outputTokens uint64) (uint64, error) {
	body, err := c.body(input, outputTokens)
	if err != nil {
		return 0, err
	}
	tokens, err := c.config.CountTokens(body)
	if err != nil || tokens == 0 {
		return 0, ragy.ErrInvalidArgument
	}
	return tokens, nil
}

// Call performs one SDK operation with SDK retries disabled. Usage remains available on refusal, truncation, validation
// failure and token overrun. Error strings suppress private payloads; SDK causes are private diagnostics.
func (c *Client[T]) Call(ctx context.Context, input any, limits Limits) (T, Usage, error) {
	return c.call(ctx, input, limits, nil)
}

// call admits immediately before transport, after the host token counter. A
// policy change inside that callback must prevent sending admitted source text.
func (c *Client[T]) call(
	ctx context.Context,
	input any,
	limits Limits,
	admit func(context.Context) error,
) (T, Usage, error) {
	var empty T
	if ctx == nil || limits.InputTokens == 0 || c == nil {
		return empty, Usage{}, ragy.ErrInvalidArgument
	}
	ctx, cancel := context.WithTimeout(ctx, c.config.Duration)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return empty, Usage{}, err
	}
	params, err := c.params(input, limits.OutputTokens)
	if err != nil {
		return empty, Usage{}, err
	}
	body, err := json.Marshal(params)
	if err == nil && len(body) > c.config.MaxRequestBytes {
		err = ragy.ErrInvalidArgument
	}
	if err != nil {
		return empty, Usage{}, err
	}
	tokens, err := c.config.CountTokens(bytes.Clone(body))
	if err != nil {
		return empty, Usage{}, ragy.ErrInvalidArgument
	}
	if tokens == 0 || tokens > limits.InputTokens {
		return empty, Usage{}, budget.ErrExhausted
	}
	if err = ctx.Err(); err != nil {
		return empty, Usage{}, err
	}
	if err = checkAdmission(ctx, admit); err != nil {
		return empty, Usage{}, err
	}
	// SDK-L06/L07: SDK owns cancellation response cleanup and endpoint resolution.
	var httpResponse *http.Response
	// Public SDK status observation only; no body capture or parser.
	decoded, err := c.sdk.Chat.Completions.New(
		ctx,
		params,
		option.WithMaxRetries(0),
		option.WithResponseInto(&httpResponse),
	)
	if err != nil {
		status := 0
		if httpResponse != nil {
			status = httpResponse.StatusCode
		}
		if apiErr, ok := errors.AsType[*openai.Error](err); ok {
			status = apiErr.StatusCode
		}

		return empty, Usage{}, providerpolicy.Failure(ctx, err, status)
	}
	if decoded == nil {
		return empty, Usage{}, ragy.ErrProtocol
	}
	output, usage, err := c.decode(*decoded, limits)
	if gateErr := ctx.Err(); gateErr != nil {
		return empty, usage, errors.Join(gateErr, err)
	}
	return output, usage, err
}

func checkAdmission(ctx context.Context, admit func(context.Context) error) error {
	if admit == nil {
		return nil
	}
	return admit(ctx)
}
