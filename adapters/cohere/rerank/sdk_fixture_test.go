package rerank

import (
	"net/http"

	"github.com/cohere-ai/cohere-go/v2/client"
	"github.com/cohere-ai/cohere-go/v2/core"

	"github.com/skosovsky/ragy/internal/providerhttp"
)

//nolint:unparam // Shared fixture signature permits host transport probes.
func newSDK(key, base string, transport providerhttp.Doer) *client.Client {
	if providerhttp.ValidateAPIKey(key) != nil {
		return nil
	}
	opts := []core.RequestOption{client.WithToken(key), client.WithMaxAttempts(1)}
	if base != "" {
		if _, err := providerhttp.ParseBaseURL(base); err != nil {
			return nil
		}
		opts = append(opts, client.WithBaseURL(base))
	}
	if transport == nil {
		transport = &http.Client{
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		}
	}
	opts = append(opts, client.WithHTTPClient(transport))
	return client.NewClient(opts...)
}
