package structured

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/recipe/budget"
)

type response struct {
	Choices []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role    string  `json:"role"`
			Content *string `json:"content"`
			Refusal *string `json:"refusal"`
		} `json:"message"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     *uint64 `json:"prompt_tokens"`
		CompletionTokens *uint64 `json:"completion_tokens"`
		TotalTokens      *uint64 `json:"total_tokens"`
	} `json:"usage"`
}

const maxJSONDepth = 64

func (c *Client[T]) decode(payload []byte, limits Limits) (T, Usage, error) {
	var empty T
	if err := validJSON(payload); err != nil {
		return empty, Usage{}, ragy.ErrProtocol
	}
	var wire response
	if err := json.Unmarshal(payload, &wire); err != nil {
		return empty, Usage{}, ragy.ErrProtocol
	}
	usage := wire.usage()
	if !usage.Known {
		return empty, usage, ragy.ErrProtocol
	}
	if usage.InputTokens > limits.InputTokens || usage.OutputTokens > limits.OutputTokens {
		return empty, usage, budget.ErrUsageExceeded
	}
	if len(wire.Choices) != 1 || wire.Choices[0].Index != 0 {
		return empty, usage, ragy.ErrProtocol
	}
	choice := wire.Choices[0]
	if choice.Message.Refusal != nil {
		return empty, usage, ErrRefused
	}
	if choice.FinishReason == "length" {
		return empty, usage, ErrIncomplete
	}
	if choice.FinishReason != "stop" || choice.Message.Role != "assistant" || choice.Message.Content == nil {
		return empty, usage, ragy.ErrProtocol
	}
	content := []byte(*choice.Message.Content)
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

func (r response) usage() Usage {
	u := r.Usage
	if u == nil || u.PromptTokens == nil || u.CompletionTokens == nil || u.TotalTokens == nil ||
		*u.PromptTokens > math.MaxUint64-*u.CompletionTokens || *u.TotalTokens != *u.PromptTokens+*u.CompletionTokens {
		return Usage{}
	}
	return Usage{InputTokens: *u.PromptTokens, OutputTokens: *u.CompletionTokens, Known: true}
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
