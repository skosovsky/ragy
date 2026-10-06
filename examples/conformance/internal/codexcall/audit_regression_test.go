package codexcall

import (
	"errors"
	"testing"
)

func TestAuditFailedTraceRetainsReturnedUsageAndEvents(t *testing.T) {
	// Arrange: executor reports an execution error, then emits full available settlement.
	data := []byte(
		"{\"type\":\"error\",\"message\":\"temporary failure\"}\n{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"{\\\"ok\\\":true}\"}}\n{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":5327,\"output_tokens\":22,\"cached_input_tokens\":1}}\n",
	)
	var got Result
	// Act.
	err := parseTrace(data, &got)
	// Assert: execution failure remains failed, but already returned tokens/events are not hidden.
	if !errors.Is(err, ErrExecution) {
		t.Fatal("execution error ignored", err)
	}
	if got.Usage == nil || got.Usage.Input != 5327 || got.Usage.Output != 22 || len(got.Events) != 3 {
		t.Fatalf("returned executor evidence lost: usage=%v events=%d", got.Usage, len(got.Events))
	}
}
func TestAuditConsumerPortDecodeRejectsDuplicateFields(t *testing.T) {
	// Arrange.
	var got struct {
		OK bool `json:"ok"`
	}
	// Act.
	err := Decode([]byte(`{"ok":true,"ok":false}`), &got)
	// Assert.
	if err == nil {
		t.Fatalf("ambiguous model port accepted: %+v", got)
	}
}

func TestTraceRetainsToolsAndEverySettlementAfterFailure(t *testing.T) {
	// Arrange: a failed invocation returned two settlements and a tool event.
	data := []byte(
		"{\"type\":\"error\"}\n{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}\n{\"type\":\"item.started\",\"item\":{\"type\":\"command_execution\"}}\n{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":20,\"output_tokens\":3}}\n",
	)
	var got Result
	// Act.
	err := parseTrace(data, &got)
	// Assert: all reported settlements are summed; failure and tool activity remain explicit.
	if err == nil || (got.Usage == nil || got.Usage.Input != 30 || got.Usage.Output != 5) || len(got.Events) != 4 ||
		!got.ToolActivity ||
		got.Trace != string(data) {
		t.Fatal(err, got)
	}
}
func TestUniqueJSONRejectsNestedAndCaseAliasDuplicates(t *testing.T) {
	for _, data := range []string{`{"nested":{"ok":1,"ok":2}}`, `{"array":[{"ok":1,"OK":2}]}`, `{"sufficient":true,"\u017Fufficient":false}`} {
		// Arrange.
		var got any
		// Act.
		err := Decode([]byte(data), &got)
		// Assert.
		if err == nil {
			t.Fatal("ambiguous fields accepted", data)
		}
	}
}

func TestUnknownUsageRemainsUnavailableAfterFurtherSettlements(t *testing.T) {
	for _, first := range []string{`{}`, `{"input_tokens":18446744073709551615,"output_tokens":1}`} {
		// Arrange: unknown or overflowing earlier settlement, followed by known usage.
		data := []byte(
			"{\"type\":\"turn.completed\",\"usage\":" + first + "}\n{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}\n{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}\n",
		)
		var result Result
		// Act.
		err := parseTrace(data, &result)
		// Assert: later known evidence cannot restore full aggregate knowledge.
		if err == nil || result.Usage != nil || !result.UsageUnavailable || len(result.Events) != 3 {
			t.Fatal(first, err, result)
		}
	}
}
