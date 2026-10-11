package dense

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3"
)

func TestSDKL02NumericPrecision(t *testing.T) {
	// Arrange: exact integer outside float64's consecutive-integer range.
	var dto openai.CreateEmbeddingResponse
	// Act: official public decoder.
	err := json.Unmarshal([]byte(`{"data":[],"usage":{"prompt_tokens":9007199254740993}}`), &dto)
	// Assert: characterize dependency rather than reparsing its raw envelope.
	if err != nil || dto.Usage.PromptTokens != 9007199254740993 {
		t.Fatalf("SDK-L02 numeric behavior: %d %v", dto.Usage.PromptTokens, err)
	}
}
