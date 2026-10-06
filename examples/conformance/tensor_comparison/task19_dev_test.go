//go:build darwin || linux

package main

import (
	"context"
	"testing"
	"time"

	"example.com/ragyconsumer/internal/task19"
)

func TestTask19DevelopmentExecutionPolicy(t *testing.T) {
	// Arrange: only development corpus/query input, never holdout labels.
	c, s, err := task19.Read("../datasets/task19/corpus.json", "../datasets/task19/dev.json")
	if err != nil {
		t.Fatal(err)
	}
	cleanup, _, err := task19.PrepareDenseTensor(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	ctx, cancel := context.WithTimeout(context.Background(), task19.Deadline)
	defer cancel()
	started := time.Now()
	// Act.
	row, err := task19Run(ctx, c, s.Queries[0], "tensor-candidate-maxsim", 1)
	row.Nanos = time.Since(started).Nanoseconds()
	task19.Audit(c, s.Queries[0], &row)
	// Assert actual dispatch/enforcement, including original citations and scope.
	if err != nil {
		t.Fatalf("execution after %s: %v; row %+v", time.Since(started), err, row)
	}
	if !task19.Compliant(row) {
		t.Fatalf("noncompliant actual execution: %+v", row)
	}
}
