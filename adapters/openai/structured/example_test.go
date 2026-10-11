package structured_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/openai/structured"
)

func ExampleClient_Call() {
	// Offline fixture: fixed token counter is fixture accounting, not a live tokenizer.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(
			w,
			`{"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"{\"value\":1}"}}],"usage":{"prompt_tokens":40,"completion_tokens":5,"total_tokens":45}}`,
		)
	}))
	defer server.Close()
	httpClient := &http.Client{
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	sdk := openai.NewClient(
		option.WithAPIKey("fixture"),
		option.WithBaseURL(server.URL),
		option.WithHTTPClient(httpClient),
		option.WithMaxRetries(0),
	)
	cfg := structured.Config{
		SDK:          &sdk,
		Model:        "fixture-model",
		Instructions: "Extract value",
		SchemaName:   "fixture",
		Schema: json.RawMessage(
			`{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}`,
		),
		MaxRequestBytes:  8192,
		MaxResponseBytes: 8192,
		Duration:         time.Second,
		Validate: func(raw json.RawMessage) error {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &fields); err != nil {
				return err
			}
			if len(fields) != 1 || fields["value"] == nil {
				return ragy.ErrProtocol
			}
			var value uint64
			return json.Unmarshal(fields["value"], &value)
		}, CountTokens: func([]byte) (uint64, error) { return 40, nil },
	}
	client, err := structured.New[payload](cfg)
	if err != nil {
		panic(err)
	}
	result, usage, err := client.Call(
		context.Background(),
		"source",
		structured.Limits{InputTokens: 100, OutputTokens: 10},
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Value, usage.Known)
	// Output: 1 true
}
