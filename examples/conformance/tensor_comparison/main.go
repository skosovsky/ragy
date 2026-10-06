//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

const (
	minimumNDCGGain        = 0.05
	minimumCandidateRecall = 0.95
	experimentTimeout      = 10 * time.Second
	maxFixtureBytes        = 128 << 10
	float32Bytes           = 4
	targetLimit            = 100
	manifestBytes          = 8 << 20
	targetFileBytes        = 1 << 20
	savedModel             = "saved-fixture"
)

type fixtureDoc struct {
	ID     string        `json:"id"`
	Dense  []float32     `json:"dense"`
	Tokens tensor.Tensor `json:"tokens"`
	Grade  *int          `json:"grade"`
}
type fixtureQuery struct {
	ID     string        `json:"id"`
	Dense  []float32     `json:"dense"`
	Tokens tensor.Tensor `json:"tokens"`
}
type fixture struct {
	Description     string       `json:"description"`
	CandidateBudget int          `json:"candidate_budget"`
	TopK            int          `json:"top_k"`
	Repetitions     int          `json:"repetitions"`
	Documents       []fixtureDoc `json:"documents"`
	Query           fixtureQuery `json:"query"`
}
type hit struct {
	Reference source.Reference         `json:"reference"`
	ID        string                   `json:"id"`
	Score     float64                  `json:"score"`
	Semantics retrieval.ScoreSemantics `json:"semantics"`
}
type quality struct {
	Recall *float64 `json:"recall_at_10"`
	NDCG   *float64 `json:"ndcg_at_10"`
}
type observation struct {
	Dense                []hit    `json:"dense"`
	Reranked             []hit    `json:"reranked"`
	Candidates           []string `json:"candidates"`
	DenseNanos           int64    `json:"dense_nanos"`
	CandidateRerankNanos int64    `json:"candidate_rerank_nanos"`
}
type report struct {
	Configuration  executionConfiguration `json:"configuration"`
	ConfigIdentity string                 `json:"config_identity"`
	DenseScope     access.Snapshot        `json:"dense_scope"`
	TensorScope    access.Snapshot        `json:"tensor_scope"`
	NDCGGain       float64                `json:"ndcg_gain"`
	QualityGate    bool                   `json:"quality_gate"`
	LatencyGate    string                 `json:"latency_gate"`

	DenseSpace        dense.Space  `json:"dense_space"`
	TensorSpace       tensor.Space `json:"tensor_space"`
	DensePublication  string       `json:"dense_publication"`
	TensorPublication string       `json:"tensor_publication"`

	Fixture                 fixture       `json:"fixture"`
	Samples                 []observation `json:"samples"`
	Baseline                quality       `json:"baseline"`
	Reranked                quality       `json:"reranked"`
	NegativeRanking         []hit         `json:"negative_ranking"`
	Negative                quality       `json:"negative"`
	CandidateRecall         float64       `json:"candidate_recall"`
	NegativeCandidateRecall float64       `json:"negative_candidate_recall"`
	DenseEmbeddingBytes     int           `json:"dense_embedding_bytes"`
	TensorEmbeddingBytes    int           `json:"tensor_embedding_bytes"`
	DenseIndexBytes         int64         `json:"dense_index_file_bytes"`
	TensorIndexBytes        int64         `json:"tensor_index_file_bytes"`
	ModelCalls              int           `json:"runtime_model_calls"`
	InputTokens             int           `json:"runtime_input_tokens"`
	OutputTokens            int           `json:"runtime_output_tokens"`
	P50                     *int64        `json:"p50_nanos"`
	P95                     *int64        `json:"p95_nanos"`
	RecommendDefault        bool          `json:"recommend_default"`
	Limitation              string        `json:"limitation"`
}

func main() {
	if err := command(); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func command() error {
	fixturePath := flag.String("fixture", "tensor_comparison/fixture.json", "saved synthetic vectors/tensors/qrels")
	outputPath := flag.String("output", "", "JSON report file; empty prints to stdout")
	task19Corpus := flag.String("task19-corpus", "", "TASK19 versioned corpus JSON")
	task19Split := flag.String("task19-split", "", "TASK19 query/qrels split JSON")
	task19Output := flag.String("task19-output", "", "TASK19 JSON report path")
	flag.Parse()
	if *task19Corpus != "" || *task19Split != "" || *task19Output != "" {
		return task19Command(*task19Corpus, *task19Split, *task19Output)
	}
	input, err := loadFixture(*fixturePath)
	if err != nil {
		return err
	}
	root, err := os.MkdirTemp("", "ragy-tensor-comparison-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(root) }()
	ctx, cancel := context.WithTimeout(context.Background(), experimentTimeout)
	defer cancel()
	result, err := experiment(ctx, root, input)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if *outputPath != "" {
		return os.WriteFile(*outputPath, data, 0o600)
	}
	_, err = os.Stdout.Write(data)
	return err
}

func loadFixture(path string) (fixture, error) {
	file, err := os.Open(path)
	if err != nil {
		return fixture{}, err
	}
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(io.LimitReader(file, maxFixtureBytes+1))
	if err != nil {
		return fixture{}, err
	}
	if len(data) > maxFixtureBytes {
		return fixture{}, ragy.ErrInvalidArgument
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var input fixture
	if err = decoder.Decode(&input); err != nil {
		return fixture{}, err
	}
	var trailing any
	if err = decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return fixture{}, ragy.ErrInvalidArgument
	}
	if err = validateFixture(input); err != nil {
		return fixture{}, err
	}
	if input.TopK != 10 || input.CandidateBudget != 100 || input.Repetitions < 1 || input.Repetitions > 10 ||
		len(input.Documents) < 3 ||
		len(input.Documents) > 100 {
		return fixture{}, ragy.ErrInvalidArgument
	}
	return input, nil
}

func experiment(ctx context.Context, root string, input fixture) (report, error) {
	profiles, err := build(ctx, root, input)
	if err != nil {
		return report{}, err
	}
	configuration, identity, err := configuredExecution(input)
	if err != nil {
		return report{}, err
	}
	out := report{
		Configuration:     configuration,
		ConfigIdentity:    identity,
		DenseScope:        profiles.denseRead.Snapshot(),
		TensorScope:       profiles.tensorRead.Snapshot(),
		Fixture:           input,
		DenseSpace:        profiles.dense.QueryCapabilities().Space,
		TensorSpace:       profiles.tensor.QueryCapabilities().Space,
		DensePublication:  profiles.denseRead.Publication().Reference(),
		TensorPublication: profiles.tensorRead.Publication().Reference(),
		LatencyGate:       "unavailable-small-fixture",
		Limitation:        "Synthetic saved hand-defined embeddings; no production quality/model/tokenizer claim. A single query and at most ten repeats are insufficient for meaningful p50/p95; raw paired timings include dense candidates and tensor I/O. Embedding sizes are float32 payload bytes; index sizes are actual logical file bytes. No live embedding/model calls occur. Recommendation is withheld because the latency gate is unverified.",
	}
	for range input.Repetitions {
		sample, e := measure(ctx, profiles, input)
		if e != nil {
			return report{}, e
		}
		out.Samples = append(out.Samples, sample)
	}
	first := out.Samples[0]
	out.Baseline, out.Reranked = metrics(first.Dense, input), metrics(first.Reranked, input)
	out.CandidateRecall = candidateRecall(first.Candidates, input)
	if out.Reranked.NDCG == nil || out.Baseline.NDCG == nil {
		return report{}, ragy.ErrInvalidArgument
	}
	out.NDCGGain = *out.Reranked.NDCG - *out.Baseline.NDCG
	out.QualityGate = out.NDCGGain >= minimumNDCGGain && out.CandidateRecall >= minimumCandidateRecall
	negative := slices.DeleteFunc(slices.Clone(first.Candidates), func(id string) bool { return id == "t1" })
	ranked, err := rerank(ctx, profiles, input, negative)
	if err != nil {
		return report{}, err
	}
	out.NegativeRanking, out.Negative = ranked, metrics(ranked, input)
	out.NegativeCandidateRecall = candidateRecall(negative, input)
	for _, doc := range input.Documents {
		out.DenseEmbeddingBytes += len(doc.Dense) * float32Bytes
		for _, row := range doc.Tokens {
			out.TensorEmbeddingBytes += len(row) * float32Bytes
		}
	}
	out.DenseIndexBytes, err = directoryBytes(profiles.denseRoot)
	if err != nil {
		return report{}, err
	}
	out.TensorIndexBytes, err = directoryBytes(profiles.tensorRoot)
	return out, err
}

func measure(ctx context.Context, p profiles, input fixture) (observation, error) {
	start := time.Now()
	baseline, err := denseRanking(ctx, p, input, input.TopK)
	elapsed := time.Since(start).Nanoseconds()
	if err != nil {
		return observation{}, err
	}
	combinedStart := time.Now()
	candidates, err := denseRanking(ctx, p, input, input.CandidateBudget)
	if err != nil {
		return observation{}, err
	}
	ids := make([]string, 0, len(candidates))
	for _, item := range candidates {
		ids = append(ids, item.ID)
	}
	ranked, err := rerank(ctx, p, input, ids)
	if err != nil {
		return observation{}, err
	}
	return observation{
		Dense:                baseline,
		Reranked:             ranked,
		Candidates:           ids,
		DenseNanos:           elapsed,
		CandidateRerankNanos: time.Since(combinedStart).Nanoseconds(),
	}, nil
}

func denseRanking(ctx context.Context, p profiles, input fixture, topK int) ([]hit, error) {
	result, err := p.dense.Retrieve(
		ctx,
		retrieval.Query[densefs.Intent]{
			Read: p.denseRead,
			Intent: densefs.Intent{
				Embedding: dense.Embedding{Space: p.dense.QueryCapabilities().Space, Vector: input.Query.Dense},
			},
			Options: retrieval.RetrieveOptions{TopK: topK},
		},
	)
	if err != nil {
		return nil, err
	}
	return hits(result.Documents())
}

func rerank(ctx context.Context, p profiles, input fixture, ids []string) ([]hit, error) {
	refs := make([]source.Reference, 0, len(ids))
	for _, id := range ids {
		ref, ok := p.refs[id]
		if !ok {
			return nil, ragy.ErrProtocol
		}
		refs = append(refs, ref)
	}
	result, err := p.tensor.Query(
		ctx,
		retrieval.Query[tensorquery.Intent]{
			Read: p.tensorRead,
			Intent: tensorquery.Intent{
				Embedding: tensor.Embedding{
					Space:  p.tensor.QueryCapabilities().Space,
					Tokens: input.Query.Tokens,
				},
				Candidates:      refs,
				CandidateBudget: input.CandidateBudget,
			},
			Options: retrieval.RetrieveOptions{TopK: input.TopK},
		},
	)
	if err != nil {
		return nil, err
	}
	return hits(result.Documents.Documents())
}

func hits(docs []retrieval.Document[meta]) ([]hit, error) {
	out := make([]hit, 0, len(docs))
	for _, doc := range docs {
		locations := doc.SourceLocations()
		if len(locations) != 1 {
			return nil, ragy.ErrProtocol
		}
		out = append(
			out,
			hit{
				Reference: locations[0].Reference,
				ID:        locations[0].Reference.Artifact,
				Score:     doc.Score,
				Semantics: doc.ScoreSemantics,
			},
		)
	}
	return out, nil
}

func candidateRecall(ids []string, input fixture) float64 {
	relevant, found := 0, 0
	for _, doc := range input.Documents {
		if doc.Grade != nil && *doc.Grade > 0 {
			relevant++
			if slices.Contains(ids, doc.ID) {
				found++
			}
		}
	}
	if relevant == 0 {
		return 0
	}
	return float64(found) / float64(relevant)
}
func metrics(ranking []hit, input fixture) quality {
	grades := make(map[string]int, len(input.Documents))
	ideal := make([]int, 0, len(input.Documents))
	ids := make([]string, 0, len(ranking))
	for _, doc := range input.Documents {
		if doc.Grade == nil {
			return quality{}
		}
		grades[doc.ID] = *doc.Grade
		ideal = append(ideal, *doc.Grade)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(ideal)))
	actual := make([]int, 0, len(ranking))
	for _, item := range ranking {
		actual = append(actual, grades[item.ID])
		ids = append(ids, item.ID)
	}
	denominator := dcg(ideal, input.TopK)
	if denominator <= 0 {
		return quality{}
	}
	ndcg := dcg(actual, input.TopK) / denominator
	recall := candidateRecall(ids, input)
	return quality{Recall: &recall, NDCG: &ndcg}
}
func dcg(grades []int, k int) float64 {
	total := 0.0
	for i, grade := range grades {
		if i >= k {
			break
		}
		total += (math.Exp2(float64(grade)) - 1) / math.Log2(float64(i+2))
	}
	return total
}
func directoryBytes(root string) (int64, error) {
	var size int64
	err := filepath.WalkDir(root, func(_ string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		info, e := entry.Info()
		if e != nil {
			return e
		}
		size += info.Size()
		return nil
	})
	return size, err
}

func validateFixture(input fixture) error {
	if strings.TrimSpace(input.Query.ID) == "" || !utf8.ValidString(input.Query.ID) {
		return ragy.ErrInvalidArgument
	}
	seen := make(map[string]bool, len(input.Documents))
	for _, doc := range input.Documents {
		if strings.TrimSpace(doc.ID) == "" || !utf8.ValidString(doc.ID) || seen[doc.ID] || doc.Grade == nil ||
			*doc.Grade < 0 ||
			*doc.Grade > 3 {
			return ragy.ErrInvalidArgument
		}
		seen[doc.ID] = true
	}
	if !seen["t1"] {
		return ragy.ErrInvalidArgument
	}
	return nil
}
