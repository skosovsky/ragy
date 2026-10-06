package modelcounter

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os/exec"
	"path/filepath"
	"time"
)

const (
	MaxRequest  = 32 << 10
	MaxResponse = 512
	Timeout     = 2 * time.Second
	WaitDelay   = 100 * time.Millisecond
)

var ErrUnavailable = errors.New("host tokenizer unavailable or incompatible")

// Counter executes a trusted local tokenizer without a shell or retries.
// The host must qualify it for complete provider request framing/schema overhead.
// Matching identity receipts alone do not establish tokenizer correctness.
type Counter struct {
	Program  string
	Args     []string
	Model    string
	Identity string
}
type Receipt struct {
	Model    string `json:"model"`
	Identity string `json:"tokenizer_identity"`
	Tokens   uint64 `json:"input_tokens"`
}

func (c Counter) Count(ctx context.Context, request []byte) (uint64, error) {
	if ctx == nil {
		return 0, ErrUnavailable
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if !filepath.IsAbs(c.Program) || c.Model == "" || c.Identity == "" || len(request) == 0 ||
		len(request) > MaxRequest {
		return 0, ErrUnavailable
	}
	// Reject incompatible request models before running the host executable.
	var envelope struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(request, &envelope) != nil || envelope.Model != c.Model {
		return 0, ErrUnavailable
	}
	child, cancel := context.WithTimeout(ctx, Timeout)
	defer cancel()
	//nolint:gosec // Program is explicit trusted host configuration with an absolute path; no shell is used, request bytes enter stdin only, and duration/output/environment are bounded.
	command := exec.CommandContext(child, c.Program, c.Args...)
	command.WaitDelay = WaitDelay
	command.Stdin = bytes.NewReader(request)
	// No inherited credentials or unrelated host process environment.
	command.Env = []string{"PATH=/usr/bin:/bin", "LANG=C.UTF-8"}
	output := limitedCounterOutput{limit: MaxResponse}
	command.Stdout = &output
	command.Stderr = io.Discard
	if err := command.Run(); err != nil {
		if contextErr := child.Err(); contextErr != nil {
			return 0, contextErr
		}
		return 0, ErrUnavailable
	}
	if err := child.Err(); err != nil {
		return 0, err
	}
	var receipt Receipt
	if err := decodeStrict(
		output.buffer.Bytes(),
		&receipt,
	); err != nil || receipt.Model != c.Model || receipt.Identity != c.Identity ||
		receipt.Tokens == 0 {
		return 0, ErrUnavailable
	}
	return receipt.Tokens, nil
}

type limitedCounterOutput struct {
	buffer bytes.Buffer
	limit  int
}

func (w *limitedCounterOutput) Write(data []byte) (int, error) {
	if len(data) > w.limit-w.buffer.Len() {
		return 0, ErrUnavailable
	}
	return w.buffer.Write(data)
}

func decodeStrict(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return ErrUnavailable
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		return ErrUnavailable
	}
	return nil
}
