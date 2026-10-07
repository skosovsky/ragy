package providerhttp

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestCredentialAdmission(t *testing.T) {
	for _, key := range []string{"", " ", "secret\r", "secret\n", "secret\t", "sec\x00ret", "secret\x7f", "\xff"} {
		// Arrange.
		t.Run("invalid", func(t *testing.T) {
			// Act.
			err := ValidateAPIKey(key)
			// Assert.
			if !errors.Is(err, ragy.ErrInvalidArgument) {
				t.Fatal("invalid credential admitted")
			}
		})
	}
	// Arrange / Act / Assert: admission preserves ordinary opaque keys.
	if err := ValidateAPIKey(" opaque-key "); err != nil {
		t.Fatal(err)
	}
}
