//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"

	"github.com/skosovsky/ragy/examples/conformance/internal/task19"

	"github.com/skosovsky/ragy/retrieval"
)

func task19Command(corpus, split, output string) error {
	if corpus == "" || split == "" || output == "" {
		return errors.New("task19 requires -task19-corpus, -task19-split and -task19-output")
	}
	data, err := os.ReadFile(corpus)
	if err != nil {
		return err
	}
	var c task19.Corpus
	if err = json.Unmarshal(data, &c); err != nil {
		return err
	}
	cleanup, setup, err := task19.PrepareDenseTensor(context.Background(), c)
	if err != nil {
		return err
	}
	defer cleanup()
	_ = setup // Execute's configuration records the shared explicit setup observation.
	return task19.Execute(corpus, split, output, []string{"baseline", "tensor-candidate-maxsim"}, task19Run)
}

func task19Run(ctx context.Context, c task19.Corpus, q task19.Query, strategy string, _ int) (task19.Row, error) {
	row := task19.Row{UsageKnown: true, BillingState: "no-provider-invoked"}
	if strategy == "baseline" {
		binding, index, err := task19.BM25(ctx, c, q)
		if err != nil {
			return row, err
		}
		row.RetrievalCalls++
		result, err := index.Retrieve(
			ctx,
			retrieval.Query[struct{}]{
				Read:    binding,
				Text:    q.Text,
				Options: retrieval.RetrieveOptions{TopK: task19.CandidateLimit},
			},
		)
		docs := result.Documents()
		row.DispatchCandidates = append(row.DispatchCandidates, len(docs))
		if err != nil {
			return row, err
		}
		for _, doc := range docs {
			row.CandidateIDs = append(row.CandidateIDs, doc.ID)
		}
		err = task19.Pack(ctx, c, binding, docs[:min(task19.TopK, len(docs))], &row)
		return row, err
	}
	// Both host hash encoding and native MaxSim run locally with no model dispatch.
	// Count the actual dense candidate and tensor query dispatches separately.

	binding, docs, err := task19.TensorRetrieveObserved(ctx, c, q, &row)
	if err != nil {
		return row, err
	}
	err = task19.Pack(ctx, c, binding, docs, &row)
	return row, err
}
