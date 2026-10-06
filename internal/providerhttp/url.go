package providerhttp

import (
	"net/url"
	"strings"

	ragy "github.com/skosovsky/ragy"
)

// ParseBaseURL admits absolute HTTP(S) provider bases without credentials, query,
// fragment or opaque payload. Host network policy remains the caller's concern.
func ParseBaseURL(raw string) (*url.URL, error) {
	base, err := url.Parse(raw)
	if err != nil || base.Hostname() == "" || (base.Scheme != "http" && base.Scheme != "https") || base.User != nil ||
		base.RawQuery != "" ||
		base.ForceQuery ||
		strings.Contains(raw, "#") || base.Fragment != "" ||
		base.Opaque != "" {
		return nil, ragy.ErrInvalidArgument
	}
	return base, nil
}

// Endpoint appends an admitted relative endpoint to the parsed base path without
// cleaning or interpreting the endpoint as query text. Escaped base paths survive.
func Endpoint(raw, path string) (string, error) {
	base, err := ParseBaseURL(raw)
	if err != nil || !validPath(path) {
		return "", ragy.ErrInvalidArgument
	}
	endpoint, _ := url.Parse(path) // validPath has already admitted this URL.
	escaped := strings.TrimRight(base.EscapedPath(), "/") + endpoint.EscapedPath()
	base.Path, _ = url.PathUnescape(escaped)
	base.RawPath = escaped
	return base.String(), nil
}
