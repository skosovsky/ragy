package codexcall

import (
	"errors"
	"testing"
)

func TestObservedTraceRequiresUsageAndRejectsTools(t *testing.T) {
	// Arrange: actual event vocabulary, one structured model response.
	positive := []byte(
		"{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"{\\\"ok\\\":true}\"}}\n{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":19,\"output_tokens\":2,\"cached_input_tokens\":4}}\n",
	)
	var result Result
	// Act.
	err := parseTrace(positive, &result)
	// Assert: whole input remains counted; cached input is not subtracted.
	if err != nil || result.Usage == nil || result.Usage.Input != 19 || result.Usage.Output != 2 ||
		result.Usage.CachedInput != 4 {
		t.Fatal(result, err)
	}
	for _, trace := range [][]byte{
		[]byte("{\"type\":\"turn.completed\",\"usage\":{\"output_tokens\":2}}\n"),
		[]byte("{\"type\":\"turn.completed\",\"usage\":null}\n"),
		append(append([]byte{}, positive...), positive...),
		append([]byte("{\"type\":\"item.started\",\"item\":{\"type\":\"command_execution\"}}\n"), positive...),
	} {
		var rejected Result
		err = parseTrace(trace, &rejected)
		if err == nil && !rejected.ToolActivity {
			t.Fatal("invalid trace accepted", string(trace))
		}
	}
}
func TestConsumerPortDecodeRejectsUnknownAndTrailing(t *testing.T) {
	for _, data := range []string{`{"ok":true,"extra":1}`, `{"ok":true} {"ok":false}`} {
		// Arrange.
		var output struct {
			OK bool `json:"ok"`
		}
		// Act.
		err := Decode([]byte(data), &output)
		// Assert.
		if !errors.Is(err, ErrExecution) {
			t.Fatal("non-port output accepted", err)
		}
	}
}
