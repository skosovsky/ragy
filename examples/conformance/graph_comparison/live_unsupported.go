//go:build !darwin && !linux

package main

import (
	"context"
	"errors"
)

func liveCaptureFile(context.Context, string, string, string, string) error {
	return errors.New("the supplied persistent graph capture profile requires a supported filesystem host")
}

func calibrateCodex(context.Context, string, string, string) error           { return errInvalid }
func captureCodexFile(context.Context, string, string, string, string) error { return errInvalid }
