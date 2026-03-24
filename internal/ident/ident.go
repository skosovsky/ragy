package ident

import "regexp"

var fieldPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// IsField reports whether a name satisfies the shared field/property-name policy.
func IsField(name string) bool {
	return fieldPattern.MatchString(name)
}
