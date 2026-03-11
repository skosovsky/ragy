package elasticsearch

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSanitizeSearchFields(t *testing.T) {
	t.Run("valid fields preserved", func(t *testing.T) {
		got := sanitizeSearchFields([]string{"content", "title", "field_1"})
		assert.Equal(t, []string{"content", "title", "field_1"}, got)
	})
	t.Run("invalid fields filtered", func(t *testing.T) {
		got := sanitizeSearchFields([]string{"content", "invalid.field", "also-bad", "valid_name"})
		assert.Equal(t, []string{"content", "valid_name"}, got)
	})
	t.Run("empty and all invalid returns default", func(t *testing.T) {
		got := sanitizeSearchFields(nil)
		assert.Equal(t, []string{"content"}, got)
		got = sanitizeSearchFields([]string{})
		assert.Equal(t, []string{"content"}, got)
		got = sanitizeSearchFields([]string{"bad.field", "x y"})
		assert.Equal(t, []string{"content"}, got)
	})
}
