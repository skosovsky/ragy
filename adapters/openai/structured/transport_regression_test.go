package structured_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/adapters/openai/structured"
	"github.com/skosovsky/ragy/internal/providerhttp"
)

type transportFunc func(*http.Request) (*http.Response, error)

func (f transportFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type regressionBody struct {
	read   func([]byte) (int, error)
	closed bool
}

func (b *regressionBody) Read(p []byte) (int, error) { return b.read(p) }
func (b *regressionBody) Close() error               { b.closed = true; return nil }

func TestStructuredContextTransport(t *testing.T) {
	for _, name := range []string{"body_deadline", "body_cancel", "body_io", "transport_deadline", "transport_cancel", "transport_io", "headers_cancel", "body_wrapped_deadline", "body_wrapped_cancel", "transport_wrapped_deadline", "transport_wrapped_cancel"} {
		t.Run(name, func(t *testing.T) { checkStructuredContextTransport(t, name) })
	}
}
func checkStructuredContextTransport(t *testing.T, name string) {
	t.Helper()
	// Arrange.
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	cfg := config("https://provider.example/v1")
	cfg.Duration = time.Second
	calls := 0
	body := &regressionBody{}
	cfg.HTTPClient = &http.Client{Transport: transportFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		if strings.HasPrefix(name, "transport_") {
			return nil, transportFailure(req.Context(), cancel, name)
		}
		if name == "headers_cancel" {
			cancel()
		}
		body.read = func(p []byte) (int, error) {
			if name == "body_io" {
				return copy(p, envelope(`{"value":1}`, "stop")), transportFailure(req.Context(), cancel, name)
			}
			return 0, transportFailure(req.Context(), cancel, name)
		}
		return &http.Response{StatusCode: http.StatusOK, Body: body, Header: make(http.Header)}, nil
	})}
	client, err := structured.New[payload](cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	output, usage, err := client.Call(ctx, struct{}{}, structured.Limits{InputTokens: 100, OutputTokens: 10})
	// Assert.
	want := ragy.ErrProtocol
	if name == "transport_io" {
		want = ragy.ErrUnavailable
	}
	if strings.HasSuffix(name, "deadline") {
		want = context.DeadlineExceeded
	}
	if strings.HasSuffix(name, "cancel") {
		want = context.Canceled
	}
	if !errors.Is(err, want) || output.Value != "" || usage != (structured.Usage{}) || calls != 1 {
		t.Fatal(output, usage, err, calls)
	}
	if !strings.HasPrefix(name, "transport_") && !body.closed {
		t.Fatal("body not closed")
	}
	if strings.Contains(err.Error(), "private") {
		t.Fatal("private transport error leaked", err)
	}
}
func transportFailure(ctx context.Context, cancel context.CancelFunc, name string) error {
	if strings.Contains(name, "wrapped") {
		cause := context.Canceled
		if strings.HasSuffix(name, "deadline") {
			cause = context.DeadlineExceeded
		}
		return fmt.Errorf("private wrapped transport: %w", cause)
	}
	if strings.HasSuffix(name, "deadline") {
		<-ctx.Done()
		return ctx.Err()
	}
	if strings.HasSuffix(name, "cancel") {
		cancel()
		return ctx.Err()
	}
	return errors.New("private body or transport detail")
}

func TestProviderBaseURLPolicy(t *testing.T) {
	for _, base := range []string{"https://provider.example/v1?", "https://provider.example/v1?secret=key", "https://provider.example/v1#secret", "https://provider.example/v1#", "https://secret@provider.example/v1", "https:opaque", "ftp://provider.example/v1", "/v1", "https://:443/v1"} {
		t.Run(base, func(t *testing.T) {
			// Arrange.
			calls := 0
			cfg := config(base)
			cfg.HTTPClient = &http.Client{
				Transport: transportFunc(
					func(*http.Request) (*http.Response, error) { calls++; return nil, errors.New("unexpected network") },
				),
			}
			// Act.
			client, err := structured.New[payload](cfg)
			shared, sharedErr := providerhttp.New(providerhttp.Config{BaseURL: base})
			// Assert.
			if client != nil || shared != nil || !errors.Is(err, ragy.ErrInvalidArgument) ||
				!errors.Is(sharedErr, ragy.ErrInvalidArgument) ||
				calls != 0 {
				t.Fatal(client, shared, err, sharedErr, calls)
			}
		})
	}
	for _, test := range []struct{ base, path string }{
		{"https://provider.example/v1", "/v1/chat/completions"},
		{"https://provider.example/v1/", "/v1/chat/completions"},
		{"https://provider.example", "/chat/completions"},
		{"https://provider.example/custom/prefix///", "/custom/prefix/chat/completions"},
		{"https://provider.example/custom%2F", "/custom%2F/chat/completions"},
	} {
		t.Run(test.base, func(t *testing.T) { checkProviderEndpoint(t, test.base, test.path) })
	}
}
func checkProviderEndpoint(t *testing.T, base, path string) {
	t.Helper()
	// Arrange.
	calls := 0
	cfg := config(base)
	cfg.HTTPClient = &http.Client{Transport: transportFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		if req.URL.EscapedPath() != path || req.URL.RawQuery != "" || req.URL.ForceQuery || req.URL.Fragment != "" {
			t.Error("wrong endpoint", req.URL)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(envelope(`{"value":1}`, "stop"))),
		}, nil
	})}
	client, err := structured.New[payload](cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	output, usage, err := client.Call(t.Context(), struct{}{}, structured.Limits{InputTokens: 100, OutputTokens: 10})
	shared, sharedErr := providerhttp.Endpoint(base, "/chat/completions")
	// Assert.
	if err != nil || output.Value != "1" || !usage.Known || calls != 1 || sharedErr != nil ||
		shared != "https://provider.example"+path {
		t.Fatal(output, usage, err, calls, shared, sharedErr)
	}
}
