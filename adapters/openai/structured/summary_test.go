package structured_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/openai/structured"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

func summaryConfig(endpoint string) structured.Config {
	cfg := config(endpoint)
	cfg.Schema = json.RawMessage(
		`{"type":"object","properties":{"text":{"type":"string"},"selected":{"type":"array","items":{"type":"integer"}}},"required":["text","selected"],"additionalProperties":false}`,
	)
	cfg.Validate = func(raw json.RawMessage) error {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return err
		}
		if len(fields) != 2 || fields["text"] == nil || fields["selected"] == nil {
			return ragy.ErrProtocol
		}
		var output graphsummary.ModelOutput
		if err := json.Unmarshal(raw, &output); err != nil {
			return err
		}
		if output.Text == "" || len(output.Selected) == 0 {
			return ragy.ErrProtocol
		}
		return nil
	}
	return cfg
}

func TestHTTPCommunityAndReduceBinding(t *testing.T) {
	// Arrange.
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		var wire struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err = json.Unmarshal(body, &wire); err != nil {
			t.Error(err)
		}
		var input graphsummary.ModelInput
		if len(wire.Messages) != 2 {
			t.Fatal("wrong model request")
		}
		if err = json.Unmarshal([]byte(wire.Messages[1].Content), &input); err != nil {
			t.Error(err)
		}
		output := graphsummary.ModelOutput{Text: input.Snippets[0].Text, Selected: []int{0}}
		if input.Stage == graphsummary.Reduce {
			output.Text += " " + input.Snippets[1].Text
			output.Selected = []int{0, 1}
		}
		encoded, err := json.Marshal(output)
		if err != nil {
			t.Error(err)
		}
		_, _ = io.WriteString(w, envelope(string(encoded), "stop"))
	}))
	t.Cleanup(server.Close)
	client, err := structured.NewSummarizer(
		summaryConfig(server.URL),
		func(usage structured.Usage) (uint64, error) { return usage.InputTokens + usage.OutputTokens, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	input := graphsummary.ModelInput{
		Stage:           graphsummary.Map,
		Question:        "Dependencies?",
		Snippets:        []graphsummary.ModelSnippet{{Index: 0, Text: "Billing uses LedgerDB."}},
		MaxInputTokens:  40,
		MaxOutputTokens: 10,
	}
	// Act.
	first, usage, err := client.Model(context.Background(), input)
	if err != nil || !usage.Known || usage.Value.Cost != 45 {
		t.Fatal(first, usage, err)
	}
	input.Stage = graphsummary.Reduce
	input.Snippets = []graphsummary.ModelSnippet{{Index: 0, Text: first.Text}, {Index: 1, Text: "Search uses IndexDB."}}
	count, countErr := client.CountInputTokens(input)
	reduced, usage, err := client.Model(context.Background(), input)
	// Assert.
	if countErr != nil || count != 40 || err != nil || reduced.Text != "Billing uses LedgerDB. Search uses IndexDB." ||
		len(reduced.Selected) != 2 ||
		!usage.Known ||
		usage.Value.Cost != 45 ||
		calls.Load() != 2 {
		t.Fatal(reduced, usage, err, countErr, calls.Load())
	}
}

func TestHTTPSummaryIncompletePreservesUsageWithoutRetry(t *testing.T) {
	// Arrange.
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, envelope(`{"text":"partial","selected":[0]}`, "length"))
	}))
	t.Cleanup(server.Close)
	client, err := structured.NewSummarizer(
		summaryConfig(server.URL),
		func(structured.Usage) (uint64, error) { return 30, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	out, usage, err := client.Model(
		context.Background(),
		graphsummary.ModelInput{
			Stage:           graphsummary.Map,
			Question:        "Q",
			Snippets:        []graphsummary.ModelSnippet{{Index: 0, Text: "S"}},
			MaxInputTokens:  40,
			MaxOutputTokens: 10,
		},
	)
	// Assert.
	if !errors.Is(err, structured.ErrIncomplete) || out.Text != "" || !usage.Known || usage.Value.Cost != 30 ||
		calls.Load() != 1 {
		t.Fatal(out, usage, err, calls.Load())
	}
}
