package providerhttp

import "testing"

func TestEnvelopeNormalizationPolicy(t *testing.T) {
	// Arrange: unknown fields, exact duplicates, case alias and unpaired surrogate.
	wire := []byte(`{"value":"first","value":"second","VALUE":"\ud800","future":{"accepted":true}}`)
	var output struct {
		Value string `json:"value"`
	}
	// Act.
	err := decodeObject(wire, &output)
	// Assert: standard evolution/normalization is explicit, independent of unknown-field tolerance.
	if err != nil || output.Value != "\ufffd" {
		t.Fatalf("envelope policy: %q %v", output.Value, err)
	}
}
