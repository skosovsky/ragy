package providerhttp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
)

type doerFunc func(*http.Request) (*http.Response, error)

func (fn doerFunc) Do(req *http.Request) (*http.Response, error) { return fn(req) }

type envelope struct {
	Value int `json:"value"`
}

func testClient(t *testing.T, limits embedding.Limits, transport Doer) *Client {
	t.Helper()
	client, err := New(Config{BaseURL: "https://provider.example/v1", HTTPClient: transport, Limits: limits})
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func TestConfiguration(t *testing.T) {
	for _, base := range []string{"", "ftp://provider.example", "https://secret@provider.example", "https://provider.example?key=secret", "https://provider.example?", "https://provider.example#secret"} {
		t.Run(base, func(t *testing.T) {
			// Arrange / Act.
			_, err := New(Config{BaseURL: base})
			// Assert.
			if !errors.Is(err, ragy.ErrInvalidArgument) || strings.Contains(err.Error(), "secret") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
	// Arrange / Act.
	client, err := New(Config{BaseURL: "https://provider.example"})
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	limits := client.Limits()
	if limits.MaxInputs <= 0 || limits.MaxInputBytes <= 0 || limits.MaxRequestBytes <= 0 ||
		limits.MaxResponseBytes <= 0 ||
		limits.MaxOutputTokens <= 0 ||
		limits.Timeout <= 0 {
		t.Fatalf("nonfinite defaults: %+v", limits)
	}
}

func TestValidateTexts(t *testing.T) {
	client := testClient(t, embedding.Limits{MaxInputs: 2, MaxInputBytes: 5}, nil)
	for _, test := range []struct {
		name   string
		inputs []string
		want   error
	}{
		{"valid", []string{"ab", "cde"}, nil},
		{"aggregate-overflow", []string{"abc", "def"}, ragy.ErrInvalidArgument},
		{"too-many", []string{"a", "b", "c"}, ragy.ErrInvalidArgument},
		{"empty", nil, ragy.ErrInvalidArgument},
		{"blank", []string{" \n "}, ragy.ErrInvalidArgument},
		{"invalid-utf8", []string{string([]byte{0xff})}, ragy.ErrInvalidArgument},
	} {
		t.Run(test.name, func(t *testing.T) {
			// Act.
			err := client.ValidateTexts(t.Context(), test.inputs)
			// Assert.
			if !errors.Is(err, test.want) {
				t.Fatalf("got %v want %v", err, test.want)
			}
		})
	}
	// Arrange.
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	// Act / Assert.
	if err := client.ValidateTexts(ctx, []string{"abc"}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestPostEnvelope(t *testing.T) {
	for _, test := range []struct {
		name, body string
		limit      int64
		want       error
	}{
		{"valid-unknown", `{"value":7,"new_field":true}`, 100, nil},
		{"trailing-space", "{\"value\":7} \n", 100, nil},
		{"trailing-object", `{"value":7} {}`, 100, ragy.ErrProtocol},
		{"trailing-junk", `{"value":7} secret`, 100, ragy.ErrProtocol},
		{"array", `[]`, 100, ragy.ErrProtocol},
		{"null", `null`, 100, ragy.ErrProtocol},
		{"invalid", `{"value":"secret"}`, 100, ragy.ErrProtocol},
		{"oversized", `{"value":7}`, 5, ragy.ErrProtocol},
	} {
		t.Run(test.name, func(t *testing.T) {
			// Arrange.
			calls := 0
			closed := false
			client := testClient(
				t,
				embedding.Limits{MaxResponseBytes: test.limit},
				doerFunc(func(req *http.Request) (*http.Response, error) {
					calls++
					if req.URL.String() != "https://provider.example/v1/embed" || req.Method != http.MethodPost ||
						req.Header.Get("Authorization") != "Bearer secret" {
						t.Fatalf("request: %+v", req)
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       &closeReader{Reader: strings.NewReader(test.body), closed: &closed},
					}, nil
				}),
			)
			var output envelope
			// Act.
			err := client.Post(
				t.Context(),
				"/embed",
				http.Header{"Authorization": []string{"Bearer secret"}},
				map[string]string{"input": "hello"},
				&output,
			)
			// Assert.
			if !errors.Is(err, test.want) || calls != 1 || !closed {
				t.Fatalf("err=%v calls=%d closed=%t", err, calls, closed)
			}
			if err != nil && strings.Contains(err.Error(), "secret") {
				t.Fatalf("leaked error: %v", err)
			}
			if err == nil && output.Value != 7 {
				t.Fatalf("output: %+v", output)
			}
		})
	}
}

type closeReader struct {
	io.Reader

	closed *bool
}

func (r *closeReader) Close() error { *r.closed = true; return nil }

func TestPostPreflight(t *testing.T) {
	// Arrange.
	calls := 0
	client := testClient(
		t,
		embedding.Limits{MaxRequestBytes: 2},
		doerFunc(func(*http.Request) (*http.Response, error) { calls++; return nil, errors.New("must not execute") }),
	)
	for _, path := range []string{"/embed", "//evil.example/embed", "https://evil.example", "/embed?key=secret", "/embed#secret", "/embed?"} {
		// Act.
		err := client.Post(t.Context(), path, nil, map[string]string{"secret": "large"}, &envelope{})
		// Assert.
		if !errors.Is(err, ragy.ErrInvalidArgument) || calls != 0 {
			t.Fatalf("path=%q err=%v calls=%d", path, err, calls)
		}
	}
	// Arrange.
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	// Act / Assert.
	if err := client.Post(ctx, "/embed", nil, nil, &envelope{}); !errors.Is(err, context.Canceled) || calls != 0 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

func TestPostSanitizesErrors(t *testing.T) {
	for _, test := range []struct {
		name         string
		response     *http.Response
		transportErr error
		want         error
	}{
		{"transport", nil, errors.New("secret in request URL"), ragy.ErrUnavailable},
		{"transport-canceled", nil, fmt.Errorf("secret: %w", context.Canceled), context.Canceled},
		{"transport-timeout", nil, fmt.Errorf("secret: %w", context.DeadlineExceeded), context.DeadlineExceeded},
		{"nil-response", nil, nil, ragy.ErrProtocol},
		{"nil-body", &http.Response{StatusCode: http.StatusOK}, nil, ragy.ErrProtocol},
		{"server-error", &http.Response{StatusCode: http.StatusInternalServerError, Body: io.NopCloser(strings.NewReader("secret"))}, nil, ragy.ErrUnavailable},
		{"redirect", &http.Response{StatusCode: http.StatusFound, Body: io.NopCloser(strings.NewReader("secret"))}, nil, ragy.ErrProtocol},
		{"read-error", &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(errorReader{})}, nil, ragy.ErrProtocol},
	} {
		t.Run(test.name, func(t *testing.T) {
			// Arrange.
			client := testClient(
				t,
				embedding.Limits{},
				doerFunc(func(*http.Request) (*http.Response, error) { return test.response, test.transportErr }),
			)
			// Act.
			err := client.Post(t.Context(), "/embed", nil, nil, &envelope{})
			// Assert.
			if !errors.Is(err, test.want) || strings.Contains(err.Error(), "secret") {
				t.Fatalf("got %v want %v", err, test.want)
			}
		})
	}
}

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) { return 0, errors.New("secret response error") }

func TestPostContextTimeout(t *testing.T) {
	// Arrange.
	const timeout = time.Millisecond * 10
	client := testClient(
		t,
		embedding.Limits{Timeout: timeout},
		doerFunc(func(req *http.Request) (*http.Response, error) {
			<-req.Context().Done()
			return nil, req.Context().Err()
		}),
	)
	// Act.
	err := client.Post(t.Context(), "/embed", nil, nil, &envelope{})
	// Assert.
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
}

func TestPostParentCancellationAfterTransport(t *testing.T) {
	// Arrange.
	ctx, cancel := context.WithCancel(t.Context())
	closed := false
	client := testClient(t, embedding.Limits{}, doerFunc(func(*http.Request) (*http.Response, error) {
		cancel()
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       &closeReader{Reader: strings.NewReader(`{"value":7}`), closed: &closed},
		}, nil
	}))
	// Act.
	err := client.Post(ctx, "/embed", nil, nil, &envelope{})
	// Assert.
	if !errors.Is(err, context.Canceled) || !closed {
		t.Fatalf("err=%v closed=%t", err, closed)
	}
}

func TestStandardClientRedirectsRejected(t *testing.T) {
	// Arrange.
	var calls atomic.Int32
	target := httptest.NewServer(
		http.HandlerFunc(
			func(w http.ResponseWriter, _ *http.Request) { calls.Add(1); _, _ = w.Write([]byte(`{"value":7}`)) },
		),
	)
	defer target.Close()
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer origin.Close()
	provided := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return nil }}
	client, err := New(Config{BaseURL: origin.URL, HTTPClient: provided})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	err = client.Post(t.Context(), "/embed", nil, nil, &envelope{})
	// Assert.
	if !errors.Is(err, ragy.ErrProtocol) || calls.Load() != 0 {
		t.Fatalf("err=%v targetcalls=%d", err, calls.Load())
	}
	if err := provided.CheckRedirect(nil, nil); err != nil {
		t.Fatal("caller client mutated")
	}
}
