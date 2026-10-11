package structured_test

import (
	"net/http"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/skosovsky/ragy/internal/providerhttp"
)

func newSDK(key, base string, transport providerhttp.Doer) *openai.Client {
	if providerhttp.ValidateAPIKey(key) != nil {
		return nil
	}
	opts := []option.RequestOption{option.WithAPIKey(key), option.WithMaxRetries(0)}
	if base != "" {
		if _, err := providerhttp.ParseBaseURL(base); err != nil {
			return nil
		}
		opts = append(opts, option.WithBaseURL(base))
	}
	if transport == nil {
		transport = &http.Client{
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		}
	}
	opts = append(opts, option.WithHTTPClient(transport))
	sdk := openai.NewClient(opts...)
	return &sdk
}
