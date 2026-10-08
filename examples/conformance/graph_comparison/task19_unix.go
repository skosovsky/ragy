//go:build darwin || linux

package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/skosovsky/ragy/examples/conformance/internal/task19"

	"github.com/skosovsky/ragy/retrieval"
)

func runTask19Graph(ctx context.Context, c task19.Corpus, q task19.Query, strategy string, _ int) (task19.Row, error) {
	return runPreparedTask19Graph(ctx, c, q, strategy, map[string]task19PreparedGraph{})
}

func runPreparedTask19Graph(
	ctx context.Context,
	c task19.Corpus,
	q task19.Query,
	strategy string,
	preparedGraphs map[string]task19PreparedGraph,
) (task19.Row, error) {
	row := task19.Row{UsageKnown: true, BillingState: "no-provider-invoked"}
	read, index, err := task19.BM25(ctx, c, q)
	if err != nil {
		return row, err
	}
	row.RetrievalCalls++
	result, err := index.Retrieve(
		ctx,
		retrieval.Query[struct{}]{
			Read:    read,
			Text:    q.Text,
			Options: retrieval.RetrieveOptions{TopK: task19.CandidateLimit},
		},
	)
	if err != nil {
		return row, err
	}
	candidates := result.Documents()
	row.DispatchCandidates = append(row.DispatchCandidates, len(candidates))
	for _, candidate := range candidates {
		row.CandidateIDs = append(row.CandidateIDs, candidate.ID)
	}
	if strategy == "baseline" {
		err = task19.Pack(ctx, c, read, candidates[:min(task19.TopK, len(candidates))], &row)
		return row, err
	}
	if strategy != "graph-expansion" {
		return row, errInvalid
	}
	seeds := []string{}
	for _, d := range candidates[:min(1, len(candidates))] {
		seeds = append(seeds, d.ID)
	}
	docs := task19GraphDocuments(c, q)
	root, err := os.MkdirTemp("", "ragy-task19-graph-")
	if err != nil {
		return row, err
	}
	defer os.RemoveAll(root)
	expanded, err := task19Expand(
		ctx,
		preparedGraphs,
		filepath.Join(root, "ledger"),
		c.DatasetID,
		q.Scope,
		docs,
		seeds,
		task19.RetrievalSlots-row.RetrievalCalls,
		task19.Deadline,
	)
	row.RetrievalCalls += expanded.Calls
	if expanded.Calls > 0 {
		row.DispatchCandidates = append(row.DispatchCandidates, len(expanded.IDs))
	}
	for _, id := range expanded.IDs {
		if !slices.Contains(row.CandidateIDs, id) {
			row.CandidateIDs = append(row.CandidateIDs, id)
		}
	}
	if err != nil {
		return row, err
	}
	ordered := task19GraphOrder(c, candidates, expanded.IDs, seeds)

	err = task19.Pack(ctx, c, read, ordered[:min(task19.TopK, len(ordered))], &row)
	row.GraphPublication = expanded.Publication
	row.GraphContributors = expanded.References
	return row, err
}

func executeTask19Graph(corpus, split, output string) error {
	c, s, err := task19.Read(corpus, split)
	if err != nil {
		return err
	}
	if err = task19.Validate(c, s); err != nil {
		return err
	}
	root, err := os.MkdirTemp("", "ragy-task19-graph-preparation-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(root)
	preparedGraphs := map[string]task19PreparedGraph{}
	started := time.Now()
	scopes := map[string]bool{}
	for _, q := range s.Queries {
		if scopes[q.Scope] {
			continue
		}
		scopes[q.Scope] = true
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		_, err = task19Expand(
			ctx,
			preparedGraphs,
			filepath.Join(root, task19.Digest([]byte(q.Scope))),
			c.DatasetID,
			q.Scope,
			task19GraphDocuments(c, q),
			nil,
			task19.RetrievalSlots,
			time.Minute,
		)
		cancel()
		if err != nil {
			return err
		}
	}
	publications := []string{}
	for _, prepared := range preparedGraphs {
		publications = append(publications, prepared.publication.Reference())
	}
	slices.Sort(publications)
	task19.RegisterSetup(
		"graph_index_setup",
		map[string]any{
			"elapsed_nanos": time.Since(started).Nanoseconds(),
			"scopes":        len(scopes), "documents": len(c.Documents), "deadline_nanos": int64(time.Minute),
			"model_calls": 0, "source_publications": publications,
			"profile": "authored-relations-managed-publication/v1",
		},
	)
	return task19.Execute(
		corpus,
		split,
		output,
		[]string{"baseline", "graph-expansion"},
		func(ctx context.Context, c task19.Corpus, q task19.Query, strategy string, _ int) (task19.Row, error) {
			return runPreparedTask19Graph(ctx, c, q, strategy, preparedGraphs)
		},
	)
}

func task19GraphOrder(
	c task19.Corpus,
	candidates []retrieval.Document[task19.Meta],
	ids, seeds []string,
) []retrieval.Document[task19.Meta] {
	// Reachable neighbors precede the seed, with lexical order providing the
	// explicit host tie-break. Graph evidence carries no similarity score.
	ordered := []retrieval.Document[task19.Meta]{}
	ids = slices.Clone(ids)
	slices.Sort(ids)
	originals := task19.Documents(c)
	for _, id := range ids {
		if slices.Contains(seeds, id) {
			continue
		}
		for _, d := range originals {
			if d.ID == id {
				ordered = append(ordered, d)
				break
			}
		}
	}
	for _, d := range candidates {
		if slices.Contains(ids, d.ID) && slices.Contains(seeds, d.ID) {
			ordered = append(ordered, d)
		}
	}
	// A disconnected graph still retains the original lexical candidates.
	for _, d := range candidates {
		found := false
		for _, o := range ordered {
			if o.ID == d.ID {
				found = true
				break
			}
		}
		if !found {
			ordered = append(ordered, d)
		}
	}
	return ordered
}

func task19GraphDocuments(c task19.Corpus, q task19.Query) []task19GraphDocument {
	docs := []task19GraphDocument{}
	for _, d := range task19.Eligible(c, q) {
		relations := []task19GraphRelation{}
		for _, r := range d.Relations {
			relations = append(relations, task19GraphRelation{TargetID: r.TargetID, Type: r.Type})
		}
		docs = append(
			docs,
			task19GraphDocument{
				ID:        d.ID,
				Scope:     d.Scope,
				Text:      d.Text,
				Reference: task19.Reference(c, d),
				Relations: relations,
			},
		)
	}

	return docs
}
