package structured

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"unicode/utf8"

	"github.com/openai/openai-go/v3"

	"github.com/skosovsky/ragy/internal/providerpolicy"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/recipe/budget"
)

const maxJSONDepth = 64

func (c *Client[T]) decode(wire openai.ChatCompletion, limits Limits) (T, Usage, error) {
	var empty T
	usage := readCompletionUsage(wire.Usage)
	// SDK-L02: envelope normalization belongs to SDK; validate domain content below.
	// SDK-L01: public representation cap, not wire retention.
	if err := providerpolicy.Size(wire, int64(c.config.MaxResponseBytes), ragy.ErrProtocol); err != nil {
		return empty, usage, err
	}
	if !usage.Known {
		return empty, usage, ragy.ErrProtocol
	}
	if usage.InputTokens > limits.InputTokens || usage.OutputTokens > limits.OutputTokens {
		return empty, usage, budget.ErrUsageExceeded
	}
	if len(wire.Choices) != 1 || !wire.Choices[0].JSON.Index.Valid() || wire.Choices[0].Index != 0 {
		return empty, usage, ragy.ErrProtocol
	}
	choice := wire.Choices[0]
	if choice.Message.JSON.Refusal.Valid() {
		return empty, usage, ErrRefused
	}
	if choice.FinishReason == "length" {
		return empty, usage, ErrIncomplete
	}
	if choice.FinishReason != "stop" || choice.Message.Role != "assistant" || !choice.Message.JSON.Content.Valid() {
		return empty, usage, ragy.ErrProtocol
	}
	content := []byte(choice.Message.Content)
	if err := validJSON(content); err != nil {
		return empty, usage, ragy.ErrProtocol
	}
	if err := c.config.Validate(bytes.Clone(content)); err != nil {
		return empty, usage, ragy.ErrProtocol
	}
	decoder := json.NewDecoder(bytes.NewReader(content))
	decoder.UseNumber()
	decoder.DisallowUnknownFields()
	var output T
	if err := decoder.Decode(&output); err != nil {
		return empty, usage, ragy.ErrProtocol
	}
	return output, usage, nil
}

func readCompletionUsage(u openai.CompletionUsage) Usage {
	if !u.JSON.PromptTokens.Valid() || !u.JSON.CompletionTokens.Valid() || !u.JSON.TotalTokens.Valid() ||
		u.PromptTokens < 0 ||
		u.CompletionTokens < 0 ||
		u.TotalTokens < 0 ||
		u.PromptTokens > math.MaxInt64-u.CompletionTokens ||
		u.TotalTokens != u.PromptTokens+u.CompletionTokens {
		return Usage{}
	}
	return Usage{InputTokens: uint64(u.PromptTokens), OutputTokens: uint64(u.CompletionTokens), Known: true}
}

// validJSON rejects duplicate members, invalid UTF-8, trailing data and excessive
// nesting before JSON decoding can silently discard or normalize them.
func validJSON(value []byte) error {
	if !utf8.Valid(value) {
		return ragy.ErrProtocol
	}
	decoder := json.NewDecoder(bytes.NewReader(value))
	decoder.UseNumber()
	if err := jsonValue(decoder, 0); err != nil {
		return err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return ragy.ErrProtocol
	}
	return nil
}

func jsonValue(decoder *json.Decoder, depth int) error {
	if depth > maxJSONDepth {
		return ragy.ErrProtocol
	}
	token, err := decoder.Token()
	if err != nil {
		return ragy.ErrProtocol
	}
	delim, isDelim := token.(json.Delim)
	if !isDelim {
		return nil
	}
	switch delim {
	case '{':
		if err = jsonMembers(decoder, depth); err != nil {
			return err
		}
	case '[':
		for decoder.More() {
			if err = jsonValue(decoder, depth+1); err != nil {
				return err
			}
		}
	default:
		return ragy.ErrProtocol
	}
	end, err := decoder.Token()
	if err != nil || (delim == '{' && end != json.Delim('}')) || (delim == '[' && end != json.Delim(']')) {
		return ragy.ErrProtocol
	}
	return nil
}

func jsonMembers(decoder *json.Decoder, depth int) error {
	seen := make(map[string]bool)
	for decoder.More() {
		key, err := decoder.Token()
		name, ok := key.(string)
		if err != nil || !ok || seen[name] {
			return ragy.ErrProtocol
		}
		seen[name] = true
		if err = jsonValue(decoder, depth+1); err != nil {
			return err
		}
	}
	return nil
}
