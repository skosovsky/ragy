package binding

import (
	"context"
	"net/http"

	"google.golang.org/genai"

	"github.com/skosovsky/ragy/internal/providerhttp"
)

func newSDK(key, base string, transport *http.Client) *genai.Client {
	if providerhttp.ValidateAPIKey(key) != nil {
		return nil
	}
	if base != "" {
		if _, err := providerhttp.ParseBaseURL(base); err != nil {
			return nil
		}
	}
	if transport == nil {
		transport = &http.Client{
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		}
	}
	attempts := int32(1)
	sdk, err := genai.NewClient(
		context.Background(),
		&genai.ClientConfig{
			APIKey:     key,
			Backend:    genai.BackendGeminiAPI,
			HTTPClient: transport,
			HTTPOptions: genai.HTTPOptions{
				BaseURL:      base,
				APIVersion:   "v1beta",
				RetryOptions: &genai.HTTPRetryOptions{Attempts: &attempts},
			},
		},
	)
	if err != nil {
		panic(err)
	}
	return sdk
}
