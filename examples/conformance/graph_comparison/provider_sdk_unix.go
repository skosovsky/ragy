//go:build darwin || linux

package main

import (
	"net/http"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type hostProviderConfig struct {
	APIKey     string
	Model      string
	BaseURL    string
	HTTPClient *http.Client
}

func newProviderSDK(key, base string, client *http.Client) openai.Client {
	opts := []option.RequestOption{option.WithAPIKey(key), option.WithMaxRetries(0)}
	if base != "" {
		opts = append(opts, option.WithBaseURL(base))
	}
	if client != nil {
		opts = append(opts, option.WithHTTPClient(client))
	}
	return openai.NewClient(opts...)
}
