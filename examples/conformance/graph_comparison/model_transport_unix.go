//go:build darwin || linux

package main

import (
	"net/http"
	"sync/atomic"
)

type observedModelTransport struct {
	next  http.RoundTripper
	calls atomic.Uint64
}

func (t *observedModelTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	t.calls.Add(1)
	return t.next.RoundTrip(request)
}
func observeModelClient(client *http.Client) (*http.Client, func() uint64) {
	if client == nil {
		client = &http.Client{Timeout: attemptDuration}
	}
	owned := *client
	next := owned.Transport
	if next == nil {
		next = http.DefaultTransport
	}
	transport := &observedModelTransport{next: next}
	owned.Transport = transport
	return &owned, transport.calls.Load
}
