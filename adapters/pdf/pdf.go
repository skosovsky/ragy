// Package pdf adapts an optional external Python PDF/layout parser to ragy.
// Core has no parser dependency. Input bytes must already have been authorized
// and associated with an exact retained revision by ingestion/source host code.
package pdf

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"time"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/layout"
)

//go:embed engine.py
var engine string

// Limits bounds input/output bytes, pages and per-page elements. No hidden retry
// or worker is started. Timeout preserves earlier caller deadlines.
type Limits struct {
	InputBytes  int
	OutputBytes int
	Pages       int
	Words       int
	Cells       int
	Images      int
	Timeout     time.Duration
}

// Config supplies the optional external engine and declared transformation identity.
// Python must have pdfplumber and pypdf installed; dependencies remain outside core.
type Config struct {
	Python         string
	Transformation string
	Limits         Limits
}

// Parser captures its configuration and runs the external engine once per Parse.
type Parser struct{ config Config }

// New constructs a bounded parser; no process is started during construction.
func New(config Config) (*Parser, error) {
	limits := config.Limits
	if config.Python == "" || config.Transformation == "" || limits.InputBytes <= 0 || limits.OutputBytes <= 0 ||
		limits.Pages <= 0 || limits.Words <= 0 || limits.Cells <= 0 || limits.Images <= 0 || limits.Timeout <= 0 {
		return nil, fmt.Errorf("%w: PDF parser configuration", ragy.ErrInvalidArgument)
	}
	return &Parser{config: config}, nil
}

// Parse snapshots authorized input, validates native output and returns no document
// on any failure/cancellation. Retention, source access and OCR remain host concerns.
func (p *Parser) Parse(ctx context.Context, input layout.Input) (layout.Document, error) {
	if p == nil {
		return layout.Document{}, ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return layout.Document{}, err
	}
	if err := input.Reference.Validate(); err != nil {
		return layout.Document{}, err
	}
	if input.Reference.Transformation != p.config.Transformation || input.Reference.Representation != "pdf-binary" ||
		len(input.Data) == 0 || len(input.Data) > p.config.Limits.InputBytes {
		return layout.Document{}, fmt.Errorf("%w: PDF parser input", ragy.ErrInvalidArgument)
	}
	captured := layout.Input{Reference: input.Reference, Data: bytes.Clone(input.Data)}
	bounded, cancel := context.WithTimeout(ctx, p.config.Limits.Timeout)
	defer cancel()
	output, err := p.run(bounded, captured.Data)
	if err != nil {
		return layout.Document{}, err
	}
	document, err := normalize(output, captured.Reference)
	if gateErr := bounded.Err(); gateErr != nil {
		return layout.Document{}, gateErr
	}
	if err != nil {
		return layout.Document{}, err
	}
	if err := document.Validate(); err != nil {
		return layout.Document{}, err
	}
	if gateErr := bounded.Err(); gateErr != nil {
		return layout.Document{}, gateErr
	}
	return document, nil
}

func (p *Parser) run(ctx context.Context, data []byte) (engineOutput, error) {
	limits := p.config.Limits
	//nolint:gosec // Executable is trusted host configuration; script is embedded, arguments are bounded integers, and source bytes never enter a shell.
	command := exec.CommandContext(
		ctx,
		p.config.Python,
		"-c",
		engine,
		strconv.Itoa(limits.Pages),
		strconv.Itoa(limits.Words),
		strconv.Itoa(limits.Cells),
		strconv.Itoa(limits.Images),
	)
	command.Stdin = bytes.NewReader(data)
	command.Stderr = io.Discard
	output := limitedOutput{buffer: bytes.Buffer{}, limit: limits.OutputBytes, exceeded: false}
	command.Stdout = &output
	if err := command.Run(); err != nil {
		if ctx.Err() != nil {
			return engineOutput{}, ctx.Err()
		}
		if output.exceeded {
			return engineOutput{}, ragy.ErrInvalidArgument
		}
		return engineOutput{}, fmt.Errorf("%w: external PDF parser failed", ragy.ErrUnavailable)
	}
	if !utf8.Valid(output.buffer.Bytes()) {
		return engineOutput{}, ragy.ErrProtocol
	}
	var parsed engineOutput
	decoder := json.NewDecoder(&output.buffer)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&parsed); err != nil {
		return engineOutput{}, ragy.ErrProtocol
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return engineOutput{}, ragy.ErrProtocol
	}
	if len(parsed.Pages) > limits.Pages {
		return engineOutput{}, ragy.ErrProtocol
	}
	for _, page := range parsed.Pages {
		if len(page.Words) > limits.Words || len(page.Cells) > limits.Cells || len(page.Images) > limits.Images {
			return engineOutput{}, ragy.ErrProtocol
		}
	}
	return parsed, nil
}

type limitedOutput struct {
	buffer   bytes.Buffer
	limit    int
	exceeded bool
}

func (w *limitedOutput) Write(data []byte) (int, error) {
	if len(data) > w.limit-w.buffer.Len() {
		w.exceeded = true
		return 0, fmt.Errorf("%w: PDF output limit", ragy.ErrInvalidArgument)
	}
	written, err := w.buffer.Write(data)
	if err != nil {
		return written, fmt.Errorf("%w: PDF output buffering", ragy.ErrProtocol)
	}
	return written, nil
}

var _ layout.Parser = (*Parser)(nil)
