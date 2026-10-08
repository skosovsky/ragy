package main

import (
	"context"

	"github.com/skosovsky/ragy/examples/conformance/internal/modelcounter"
)

const (
	maxCounterRequest  = modelcounter.MaxRequest
	maxCounterResponse = modelcounter.MaxResponse
	counterTimeout     = modelcounter.Timeout
)

var errCounter = modelcounter.ErrUnavailable

type counterReceipt = modelcounter.Receipt

// hostCounter supplies the trusted executable configuration for this consumer.
type hostCounter struct {
	program  string
	args     []string
	model    string
	identity string
}

func (c hostCounter) count(ctx context.Context, request []byte) (uint64, error) {
	return (modelcounter.Counter{Program: c.program, Args: c.args, Model: c.model, Identity: c.identity}).Count(
		ctx,
		request,
	)
}
