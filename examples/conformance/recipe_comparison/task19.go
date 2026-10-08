package main

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"time"
	"unicode"

	"github.com/skosovsky/ragy/examples/conformance/internal/task19"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

const maxRewriteTerms = 6
const rerankerOutputOverhead = 64

// task19Backend reuses the existing capture's call-counting pattern with BYOT metadata.
type task19Backend struct {
	index    *lexical.BM25Snapshot[task19.Meta]
	calls    uint64
	dispatch []int
}

func (b *task19Backend) Schema() filter.Schema                 { return b.index.Schema() }
func (b *task19Backend) ReadCapabilities() access.Capabilities { return b.index.ReadCapabilities() }

func (b *task19Backend) Retrieve(
	ctx context.Context,
	q retrieval.Query[struct{}],
) (retrieval.ResultSet[task19.Meta], error) {
	b.calls++
	if b.calls > task19.RetrievalSlots {
		return retrieval.NewResultSet[task19.Meta](nil, nil), errors.New("retrieval slots exceeded")
	}
	result, err := b.index.Retrieve(ctx, q)
	b.dispatch = append(b.dispatch, len(result.Documents()))
	return result, err
}
func task19Terms(text string) []string {
	stop := map[string]bool{
		"the":   true,
		"is":    true,
		"what":  true,
		"which": true,
		"and":   true,
		"for":   true,
		"to":    true,
		"a":     true,
		"of":    true,
		"in":    true,
		"how":   true,
		"does":  true,
		"its":   true,
		"do":    true,
		"this":  true,
	}
	parts := strings.FieldsFunc(
		strings.ToLower(text),
		func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) },
	)
	out := []string{}
	for _, p := range parts {
		if !stop[p] && !slices.Contains(out, p) {
			out = append(out, p)
		}
	}
	return out
}
func task19Overlap(q, text string) int {
	score := 0
	terms := task19Terms(text)
	for _, t := range task19Terms(q) {
		if slices.Contains(terms, t) {
			score++
		}
	}
	return score
}

// Plans depend only on query text; never category, answerable flags or qrels.
func task19Plan(text, profile string) []string {
	terms := task19Terms(text)
	switch profile {
	case rewriteProfile:
		return []string{strings.Join(terms[:min(maxRewriteTerms, len(terms))], " ")}
	case multiProfile:
		return []string{strings.Join(terms, " "), strings.ReplaceAll(strings.ToLower(text), "roll back", "rollback")}
	case decompositionProfile:
		parts := strings.SplitN(text, " and ", 2)
		if len(parts) == 1 {
			parts = strings.SplitN(text, ",", 2)
		}
		return parts
	}
	return nil
}

func task19Run(ctx context.Context, c task19.Corpus, q task19.Query, profile string, _ int) (task19.Row, error) {
	row := task19.Row{UsageKnown: true, BillingState: "no-provider-invoked; monetary-price-unavailable"}
	read, index, e := task19.BM25(ctx, c, q)
	if e != nil {
		return row, e
	}
	backend := &task19Backend{index: index}
	request := retrieval.Query[struct{}]{
		Read:    read,
		Text:    q.Text,
		Options: retrieval.RetrieveOptions{TopK: task19.CandidateLimit},
	}
	if profile == baselineProfile {
		result, err := backend.Retrieve(ctx, request)
		row.RetrievalCalls = backend.calls
		row.DispatchCandidates = slices.Clone(backend.dispatch)
		if err != nil {
			return row, err
		}
		docs := result.Documents()
		task19RememberCandidates(&row, docs)
		err = task19.Pack(ctx, c, read, docs[:min(task19.TopK, len(docs))], &row)
		return row, err
	}
	if profile == "hybrid-rerank" {
		return task19Hybrid(ctx, c, q, read, backend, request, row)
	}
	config := recipe.Config[struct{}, retrieval.NoRequestMeta, task19.Meta]{
		Strategy:         recipe.Strategy(profile),
		Revision:         "task19-local-text/v1",
		BackendModelFree: true,
		Backend:          backend,
		Identity:         retrieval.DocumentIDResolver[task19.Meta]{},
		Admission: func(ctx context.Context, req retrieval.Query[struct{}]) (retrieval.ReadCoverage, error) {
			_, err := retrieval.PrepareRead(ctx, req, backend)
			return retrieval.CompleteReadCoverage(), err
		},
		CloneIntent:      func(v struct{}) (struct{}, error) { return v, nil },
		CloneRequestMeta: func(v retrieval.NoRequestMeta) (retrieval.NoRequestMeta, error) { return v, nil },
		CloneMeta:        task19.Clone,
		Supports:         task19.Supports(c),
		Limits: budget.Limits{
			RetrievalCalls: task19.RetrievalSlots,
			ModelCalls:     task19.ModelSlots,
			Usage: budget.Usage{
				InputTokens:  task19.InputBytes,
				OutputTokens: task19.OutputBytes,
				Cost:         task19.ModelSlots,
			},
		},
		RequireKnownCost: true,
		Duration:         task19.Deadline,
		Now:              time.Now,
		MaxQueries:       2,
		MaxDocuments:     task19.CandidateLimit,
		FusionK:          fusionK,
		Pricing: func(_ context.Context, op recipe.Operation) (recipe.Quote, error) {
			if op == recipe.Retrieve {
				return recipe.Quote{CostKnown: true}, nil
			}
			return recipe.Quote{
				CostKnown: true,
				Usage: budget.Usage{
					InputTokens:  task19.InputBytes / task19.ModelSlots,
					OutputTokens: task19.OutputBytes / task19.ModelSlots,
					Cost:         1,
				},
			}, nil
		},
		Planner:  task19Planner(profile, &row),
		Assessor: task19Assessor(&row),
	}
	if profile == rewriteProfile {
		config.MaxQueries = 1
	}
	instance, e := recipe.New(config)
	if e != nil {
		return row, e
	}
	result, e := instance.RunOwnObserved(ctx, request)
	row.RetrievalCalls = backend.calls
	row.DispatchCandidates = slices.Clone(backend.dispatch)
	row.UsageKnown = result.Budget.UnknownUsage == 0 && !result.Budget.UnknownCost
	if e != nil {
		return row, e
	}
	row.RecipeOutcome = string(result.Outcome)
	row.StopReason = string(result.Stop)
	row.Sufficiency = result.Sufficiency
	for _, sub := range result.Queries {
		task19RememberCandidates(&row, sub.Documents)
	}
	docs := []retrieval.Document[task19.Meta]{}
	for _, v := range result.Selected {
		docs = append(docs, v.Document)
	}
	docs = docs[:min(task19.TopK, len(docs))]
	e = task19.Pack(ctx, c, read, docs, &row)
	return row, e
}

// An explicit host reranker ranks actual RRF candidates by lexical overlap.
func task19Hybrid(
	ctx context.Context,
	c task19.Corpus,
	q task19.Query,
	read access.Binding,
	b *task19Backend,
	request retrieval.Query[struct{}],
	row task19.Row,
) (task19.Row, error) {
	lexicalResult, e := b.Retrieve(ctx, request)
	row.RetrievalCalls = b.calls
	row.DispatchCandidates = slices.Clone(b.dispatch)
	if e != nil {
		return row, e
	}
	denseRead, denseDocs, e := task19.DenseRetrieveObserved(ctx, c, q, &row)
	_ = denseRead
	if e != nil {
		return row, e
	}
	merger, e := retrieval.NewReciprocalRankFusion[task19.Meta](fusionK, retrieval.DocumentIDResolver[task19.Meta]{})
	if e != nil {
		return row, e
	}
	fused, e := merger.Merge(
		ctx,
		lexicalResult,
		retrieval.NewResultSet(denseDocs, retrieval.DocumentIDResolver[task19.Meta]{}),
	)
	if e != nil {
		return row, e
	}
	ranked := fused.Documents()
	for _, d := range ranked {
		if !slices.Contains(row.CandidateIDs, d.ID) {
			row.CandidateIDs = append(row.CandidateIDs, d.ID)
		}
	}
	ranked = ranked[:min(task19.CandidateLimit, len(ranked))]
	wire, err := json.Marshal(struct {
		Query      string                            `json:"query"`
		Candidates []retrieval.Document[task19.Meta] `json:"candidates"`
	}{q.Text, ranked})
	if err != nil {
		return row, err
	}
	outputBound := uint64(0)
	for _, d := range ranked {
		outputBound += uint64(len(d.ID) + rerankerOutputOverhead)
	}
	if row.ModelCalls+row.EncoderCalls+row.RerankerCalls >= task19.ModelSlots ||
		row.LocalInputUnits+uint64(len(wire)) > task19.InputBytes ||
		row.LocalOutputUnits+outputBound > task19.OutputBytes {
		return row, errors.New("host reranker serialized budget exhausted")
	}
	row.RerankerCalls++
	row.LocalInputUnits += uint64(len(wire))
	slices.SortStableFunc(ranked, func(a, b retrieval.Document[task19.Meta]) int {
		return task19Overlap(q.Text, b.Content) - task19Overlap(q.Text, a.Content)
	})
	scored := []struct {
		ID    string `json:"id"`
		Score int    `json:"score"`
	}{}
	for _, d := range ranked {
		scored = append(scored, struct {
			ID    string `json:"id"`
			Score int    `json:"score"`
		}{d.ID, task19Overlap(q.Text, d.Content)})
	}
	outputWire, _ := json.Marshal(scored)
	row.LocalOutputUnits += uint64(len(outputWire))
	ranked = ranked[:min(task19.TopK, len(ranked))]
	e = task19.Pack(ctx, c, read, ranked, &row)
	return row, e
}

func task19Command(corpusPath, splitPath, output string) error {
	c, _, err := task19.Read(corpusPath, splitPath)
	if err != nil {
		return err
	}
	cleanup, _, err := task19.PrepareDenseTensor(context.Background(), c)
	if err != nil {
		return err
	}
	defer cleanup()
	return task19.Execute(
		corpusPath,
		splitPath,
		output,
		[]string{baselineProfile, "hybrid-rerank", rewriteProfile, multiProfile, decompositionProfile},
		task19Run,
	)
}

func task19Planner(
	profile string,
	row *task19.Row,
) func(context.Context, retrieval.Query[struct{}], recipe.ModelLimits) (recipe.Planning, error) {
	return func(ctx context.Context, req retrieval.Query[struct{}], limits recipe.ModelLimits) (recipe.Planning, error) {
		plans := slices.Compact(task19Plan(req.Text, profile))
		wire, _ := json.Marshal(struct {
			Text     string `json:"text"`
			Strategy string `json:"strategy"`
		}{req.Text, profile})
		input := uint64(len(wire))
		outputWire, _ := json.Marshal(struct {
			Queries []string `json:"queries"`
		}{plans})
		output := uint64(len(outputWire))
		row.ModelCalls++
		row.LocalInputUnits += input
		row.LocalOutputUnits += output
		if input > limits.InputTokens || output > limits.OutputTokens {
			return recipe.Planning{}, errors.New("local planner bytes exceed reservation")
		}
		return recipe.Planning{
			Queries: plans,
			Usage: recipe.Usage{
				Known: true,
				Value: budget.Usage{InputTokens: input, OutputTokens: output, Cost: 1},
			},
		}, ctx.Err()
	}
}

func task19Assessor(
	row *task19.Row,
) func(context.Context, recipe.AssessmentInput[struct{}, retrieval.NoRequestMeta, task19.Meta], recipe.ModelLimits) (recipe.Assessment, error) {
	return func(ctx context.Context, in recipe.AssessmentInput[struct{}, retrieval.NoRequestMeta, task19.Meta], limits recipe.ModelLimits) (recipe.Assessment, error) {
		selected := []int{}
		wire, _ := json.Marshal(struct {
			Original string                              `json:"original"`
			Queries  []recipe.QueryEvidence[task19.Meta] `json:"queries"`
		}{in.Original.Text, in.Queries})
		input := uint64(len(wire))
		for _, query := range in.Queries {
			admit := false
			for _, d := range query.Documents {
				if task19Overlap(in.Original.Text, d.Content) >= 2 {
					admit = true
				}
			}
			if admit {
				selected = append(selected, query.Index)
			}
		}
		outputWire, _ := json.Marshal(struct {
			Selected   []int `json:"selected"`
			Sufficient bool  `json:"sufficient"`
		}{selected, len(selected) > 0})
		output := uint64(len(outputWire))
		row.ModelCalls++
		row.LocalInputUnits += input
		row.LocalOutputUnits += output
		if input > limits.InputTokens || output > limits.OutputTokens {
			return recipe.Assessment{}, errors.New("local assessor bytes exceed reservation")
		}
		return recipe.Assessment{
			Selected:   selected,
			Sufficient: len(selected) > 0,
			Usage: recipe.Usage{
				Known: true,
				Value: budget.Usage{InputTokens: input, OutputTokens: output, Cost: 1},
			},
		}, ctx.Err()
	}
}

func task19RememberCandidates(row *task19.Row, docs []retrieval.Document[task19.Meta]) {
	for _, d := range docs {
		if !slices.Contains(row.CandidateIDs, d.ID) {
			row.CandidateIDs = append(row.CandidateIDs, d.ID)
		}
	}
}
