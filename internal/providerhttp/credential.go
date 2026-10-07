package providerhttp

import (
	"strings"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
)

// ValidateAPIKey admits an unchanged nonblank UTF-8 header value without ASCII
// control bytes or DEL. It does not validate remote credentials or key formats.
func ValidateAPIKey(key string) error {
	if strings.TrimSpace(key) == "" || !utf8.ValidString(key) {
		return ragy.ErrInvalidArgument
	}
	for i := range len(key) {
		if key[i] < 0x20 || key[i] == 0x7f {
			return ragy.ErrInvalidArgument
		}
	}
	return nil
}
