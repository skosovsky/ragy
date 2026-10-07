//go:build darwin || linux

package task19

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorfs "github.com/skosovsky/ragy/tensor/persistent"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

const task19EmbeddingDimensions = 64
const task19SignWeight = 0.125
const task19BitsPerByte = 8
const task19Float32Bytes = 4
const task19ScoreBytes = 8
const task19OriginalTransformation = "original"
const task19TextRepresentation = "text"
const task19EmbeddingProfile = "sha256-token-64-sign-bits/v1"

type task19StoreDocument struct{ ID, Source, Revision, Scope, Text string }
type task19Stores struct {
	task19Profiles

	denseStore, tensorStore lifecycle.Store
	schema                  filter.Schema
	tenant                  filter.Field[string]
	tensorBytes             map[string]uint64
}

// This host simulator hashes Unicode tokens; it provides no model efficacy evidence.
func task19Embeddings(text string) ([]float32, tensor.Tensor) {
	words := strings.FieldsFunc(
		strings.ToLower(text),
		func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) },
	)
	if len(words) == 0 {
		words = []string{"empty-input"}
	}
	tokens := make(tensor.Tensor, 0, len(words))
	mean := make([]float32, task19EmbeddingDimensions)
	for _, word := range words {
		sum := sha256.Sum256([]byte(word))
		vector := make([]float32, task19EmbeddingDimensions)
		for i := range vector {
			vector[i] = -task19SignWeight
			if sum[i/task19BitsPerByte]&(1<<uint(i%task19BitsPerByte)) != 0 {
				vector[i] = task19SignWeight
			}
			mean[i] += vector[i]
		}
		tokens = append(tokens, vector)
	}
	norm := 0.0
	for _, value := range mean {
		norm += float64(value) * float64(value)
	}
	if norm == 0 {
		mean[0] = 1
	} else {
		norm = math.Sqrt(norm)
		for i := range mean {
			mean[i] = float32(float64(mean[i]) / norm)
		}
	}
	return mean, tokens
}

//nolint:gocognit,funlen // Explicit fixture indexing keeps both managed publications and their error paths together.
func task19BuildStores(ctx context.Context, root string, c Corpus) (task19Stores, error) {
	documents := []task19StoreDocument{}
	for _, d := range c.Documents {
		if d.Current {
			documents = append(documents, task19StoreDocument{d.ID, d.SourceID, d.Revision, d.Scope, d.Text})
		}
	}
	fields := filter.NewSchema()
	tenant, err := fields.String("scope")
	if err != nil {
		return task19Stores{}, err
	}
	schema, err := fields.Build()
	if err != nil {
		return task19Stores{}, err
	}
	out := task19Stores{
		schema:      schema,
		tenant:      tenant,
		tensorBytes: map[string]uint64{},
		refs:        map[string]source.Reference{},
		denseRoot:   filepath.Join(root, "dense"),
		tensorRoot:  filepath.Join(root, "tensor"),
	}
	out.denseStore, err = filestore.New(filepath.Join(root, "dense-manifests"), task19ManifestBytes)
	if err != nil {
		return out, err
	}
	out.tensorStore, err = filestore.New(filepath.Join(root, "tensor-manifests"), task19ManifestBytes)
	if err != nil {
		return out, err
	}
	ds := dense.Space{
		Metric:        "normalized-dot",
		Model:         task19EmbeddingProfile,
		ModelRevision: "v1",
		Configuration: "unicode-lowercase-sha256",
		VectorSpace:   "signed-token-mean",
		Dimension:     task19EmbeddingDimensions,
	}
	ts := tensor.Space{
		Metric:        "normalized-dot",
		Model:         task19EmbeddingProfile,
		ModelRevision: "v1",
		Configuration: "unicode-lowercase-sha256",
		VectorSpace:   "signed-token",
		Dimension:     task19EmbeddingDimensions,
	}
	limit := max(CandidateLimit, len(documents))
	dc := densefs.Config[Meta]{
		Root:            out.denseRoot,
		Namespace:       c.DatasetID,
		Target:          "dense",
		Store:           out.denseStore,
		Schema:          schema,
		Space:           ds,
		CloneMeta:       func(m Meta) (Meta, error) { return m, nil },
		MaxRecords:      limit,
		MaxScanRecords:  limit,
		MaxCatalogBytes: task19ManifestBytes,
		MaxPayloadBytes: task19ManifestBytes,
	}
	tc := tensorfs.Config[Meta]{
		Root:            out.tensorRoot,
		Namespace:       c.DatasetID,
		Target:          "tensor",
		Store:           out.tensorStore,
		Schema:          schema,
		Space:           ts,
		CloneMeta:       func(m Meta) (Meta, error) { return m, nil },
		MaxRecords:      limit,
		MaxCatalogBytes: task19ManifestBytes,
		MaxPayloadBytes: task19ManifestBytes,
	}
	out.dense, err = densefs.New(dc)
	if err != nil {
		return out, err
	}
	out.tensor, err = tensorfs.New(tc)
	if err != nil {
		return out, err
	}
	groups := map[string][]task19StoreDocument{}
	for _, doc := range documents {
		groups[doc.Source] = append(groups[doc.Source], doc)
	}
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for i, key := range keys {
		docs := groups[key]
		var dr []densefs.Record[Meta]
		var tr []tensorfs.Record[Meta]
		var drefs, trefs []source.Reference
		for _, doc := range docs {
			if doc.Revision != docs[0].Revision || doc.Scope != docs[0].Scope {
				return out, fmt.Errorf(
					"source %s has conflicting current revision/scope: %w",
					key,
					ragy.ErrInvalidArgument,
				)
			}
			dref, tref := task19Reference(c, doc), task19Reference(c, doc)
			vector, tokens := task19Embeddings(doc.Text)
			out.tensorBytes[doc.ID] = uint64(len(tokens) * task19EmbeddingDimensions * task19Float32Bytes)
			dr = append(
				dr,
				densefs.Record[Meta]{
					Reference: dref,
					Value: dense.Record[Meta]{
						Space:   ds,
						ID:      doc.ID,
						Content: doc.Text,
						Meta:    Meta{Scope: doc.Scope},
						Vector:  vector,
					},
				},
			)
			tr = append(
				tr,
				tensorfs.Record[Meta]{
					Reference: tref,
					Value: tensor.Record[Meta]{
						Space:   ts,
						ID:      doc.ID,
						Content: doc.Text,
						Meta:    Meta{Scope: doc.Scope},
						Tensor:  tokens,
					},
				},
			)
			drefs = append(drefs, dref)
			trefs = append(trefs, tref)
			out.refs[doc.ID] = tref
		}
		if err = task19Publish(ctx, out.denseStore, "dense", out.dense, dr, drefs, cloneDense, i); err != nil {
			return out, err
		}
		if err = task19Publish(ctx, out.tensorStore, "tensor", out.tensor, tr, trefs, cloneTensor, i); err != nil {
			return out, err
		}
	}
	// Fresh adapters read persisted publications, not staging state.
	out.dense, err = densefs.New(dc)
	if err != nil {
		return out, err
	}
	out.tensor, err = tensorfs.New(tc)
	return out, err
}

func task19Reference(c Corpus, doc task19StoreDocument) source.Reference {
	return source.Reference{
		Namespace:         c.DatasetID,
		Source:            doc.Source,
		Revision:          doc.Revision,
		Transformation:    task19OriginalTransformation,
		AccessFingerprint: doc.Scope,
		Artifact:          doc.ID,
		Representation:    task19TextRepresentation,
	}
}

func task19Publish[T any](
	ctx context.Context,
	store lifecycle.Store,
	target string,
	port lifecycle.StagePort[T],
	payload T,
	refs []source.Reference,
	clone func(T) (T, error),
	sequence int,
) error {
	fingerprint := func(value T) (string, error) {
		data, err := json.Marshal(value)
		if err != nil {
			return "", err
		}
		sum := sha256.Sum256(data)
		return hex.EncodeToString(sum[:]), nil
	}
	hash, err := fingerprint(payload)
	if err != nil {
		return err
	}
	first := refs[0]
	artifacts := make([]lifecycle.Artifact, 0, len(refs))
	for _, ref := range refs {
		support := ref
		support.Transformation = task19OriginalTransformation
		support.Representation = task19TextRepresentation
		artifacts = append(artifacts, lifecycle.Artifact{Reference: ref, Supports: []source.Reference{support}})
	}
	plan := lifecycle.Manifest{
		ID:      fmt.Sprintf("source-%d", sequence),
		Key:     first.Source,
		Payload: hash,
		State:   lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      first.Namespace,
			Source:         first.Source,
			Revision:       first.Revision,
			Content:        hash,
			Transformation: first.Transformation,
			Access:         first.AccessFingerprint,
		},
		Targets: []lifecycle.Target{
			{Name: target, Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
		},
	}
	executor, err := lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[T]{
			Store:        store,
			Targets:      []lifecycle.Registration[T]{{Name: target, Port: port}},
			ClonePayload: clone,
			ValidatePayload: func(m lifecycle.Manifest, value T) error {
				actual, e := fingerprint(value)
				if e != nil {
					return e
				}
				if actual != m.Payload {
					return ragy.ErrInvalidArgument
				}
				return nil
			},
		},
	)
	if err != nil {
		return err
	}
	if _, err = executor.Prepare(ctx, plan); err != nil {
		return err
	}
	if _, err = executor.Stage(ctx, first.Namespace, plan.ID, target, payload); err != nil {
		return err
	}
	_, err = executor.Publish(ctx, first.Namespace, plan.ID)
	return err
}

func (p task19Stores) binding(
	ctx context.Context,
	store lifecycle.Store,
	namespace, target, scope string,
) (access.Binding, error) {
	publication, err := lifecycle.CapturePublication(ctx, store, namespace, []string{target})
	if err != nil {
		return access.Binding{}, err
	}
	builder, err := filter.NewBuilder(p.schema)
	if err != nil {
		return access.Binding{}, err
	}
	mandatory, err := filter.In(builder, p.tenant, "public", scope).Build()
	if err != nil {
		return access.Binding{}, err
	}
	now := time.Now()
	return access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "task19/" + scope,
				PolicyEpoch: 1,
				IssuedAt:    now,
				ExpiresAt:   now.Add(time.Minute),
			},
			Mandatory:   mandatory,
			Schema:      p.schema,
			Publication: publication,
			Now:         time.Now,
			Authority:   access.AuthorityFunc(func(context.Context, access.Snapshot) error { return nil }),
		},
	)
}

const task19ManifestBytes = 8 << 20

type task19Profiles struct {
	dense                 *densefs.Adapter[Meta]
	tensor                *tensorfs.Adapter[Meta]
	refs                  map[string]source.Reference
	denseRoot, tensorRoot string
}

func cloneDense(input []densefs.Record[Meta]) ([]densefs.Record[Meta], error) {
	out := slices.Clone(input)
	for i := range out {
		out[i].Value.Vector = slices.Clone(out[i].Value.Vector)
	}
	return out, nil
}
func cloneTensor(input []tensorfs.Record[Meta]) ([]tensorfs.Record[Meta], error) {
	out := slices.Clone(input)
	for i := range out {
		out[i].Value.Tensor = slices.Clone(out[i].Value.Tensor)
		for j := range out[i].Value.Tensor {
			out[i].Value.Tensor[j] = slices.Clone(out[i].Value.Tensor[j])
		}
	}
	return out, nil
}

// DenseRetrieve supplies actual scoped persistent mean-vector candidates.
func DenseRetrieve(ctx context.Context, c Corpus, q Query) (access.Binding, []retrieval.Document[Meta], error) {
	return task19ManagedRetrieve(ctx, c, q, false, nil)
}

// TensorRetrieve performs candidate-only native MaxSim on dense candidates.
func TensorRetrieve(ctx context.Context, c Corpus, q Query) (access.Binding, []retrieval.Document[Meta], error) {
	return task19ManagedRetrieve(ctx, c, q, true, nil)
}

func DenseRetrieveObserved(
	ctx context.Context,
	c Corpus,
	q Query,
	row *Row,
) (access.Binding, []retrieval.Document[Meta], error) {
	return task19ManagedRetrieve(ctx, c, q, false, row)
}

func TensorRetrieveObserved(
	ctx context.Context,
	c Corpus,
	q Query,
	row *Row,
) (access.Binding, []retrieval.Document[Meta], error) {
	return task19ManagedRetrieve(ctx, c, q, true, row)
}

//nolint:gocognit,funlen // The harness records each native dispatch and enforces its resource budget at the boundary.
func task19ManagedRetrieve(
	ctx context.Context,
	c Corpus,
	q Query,
	rerank bool,
	row *Row,
) (access.Binding, []retrieval.Document[Meta], error) {
	p, release, err := task19AcquireStores(ctx, c)
	defer release()
	if err != nil {
		return access.Binding{}, nil, err
	}
	dRead, err := p.binding(ctx, p.denseStore, c.DatasetID, "dense", q.Scope)
	if err != nil {
		return dRead, nil, err
	}
	queryTokenCount := len(
		strings.FieldsFunc(
			strings.ToLower(q.Text),
			func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) },
		),
	)
	queryTokenCount = max(1, queryTokenCount)
	encoderOutput := uint64((1 + queryTokenCount) * task19EmbeddingDimensions * task19Float32Bytes)
	if err = task19ReserveOperation(row, uint64(len(q.Text)), encoderOutput, true); err != nil {
		return dRead, nil, err
	}
	if row != nil {
		row.EncoderCalls++
		row.LocalInputUnits += uint64(len(q.Text))
	}
	vector, tokens := task19Embeddings(q.Text)
	if row != nil {
		row.LocalOutputUnits += encoderOutput
	}
	if row != nil {
		if row.RetrievalCalls >= RetrievalSlots {
			return dRead, nil, errors.New("retrieval slots exhausted")
		}
		row.RetrievalCalls++
		row.DispatchCandidates = append(row.DispatchCandidates, 0)
	}
	candidates, err := p.dense.Retrieve(
		ctx,
		retrieval.Query[densefs.Intent]{
			Read: dRead,
			Intent: densefs.Intent{
				Embedding: dense.Embedding{Space: p.dense.QueryCapabilities().Space, Vector: vector},
			},
			Options: retrieval.RetrieveOptions{TopK: CandidateLimit},
		},
	)
	if err != nil {
		return dRead, nil, err
	}
	docs := task19OriginalDocuments(c, candidates.Documents())
	if row != nil {
		row.DispatchCandidates[len(row.DispatchCandidates)-1] = len(docs)
		for _, doc := range docs {
			row.CandidateIDs = append(row.CandidateIDs, doc.ID)
		}
	}
	if !rerank {
		return dRead, docs, nil
	}
	tRead, err := p.binding(ctx, p.tensorStore, c.DatasetID, "tensor", q.Scope)
	if err != nil {
		return tRead, nil, err
	}
	refs := make([]source.Reference, 0, len(docs))
	for _, d := range docs {
		refs = append(refs, p.refs[d.ID])
	}
	if len(refs) == 0 {
		return tRead, nil, nil
	}
	rerankerInput := uint64(len(tokens) * task19EmbeddingDimensions * task19Float32Bytes)
	outputBound := uint64(0)
	for _, doc := range docs {
		rerankerInput += p.tensorBytes[doc.ID]
		loc := doc.SourceLocations()[0]
		nativeID, e := loc.Identity()
		if e != nil {
			return tRead, nil, e
		}
		outputBound += uint64(len(nativeID) + task19ScoreBytes)
	}
	if err = task19ReserveOperation(row, rerankerInput, outputBound, true); err != nil {
		return tRead, nil, err
	}
	if row != nil {
		if row.RetrievalCalls >= RetrievalSlots {
			return tRead, nil, errors.New("retrieval slots exhausted")
		}
		row.RetrievalCalls++
		row.RerankerCalls++
		row.LocalInputUnits += rerankerInput
		row.DispatchCandidates = append(row.DispatchCandidates, len(refs))
	}
	result, err := p.tensor.Query(
		ctx,
		retrieval.Query[tensorquery.Intent]{
			Read: tRead,
			Intent: tensorquery.Intent{
				Embedding:       tensor.Embedding{Space: p.tensor.QueryCapabilities().Space, Tokens: tokens},
				Candidates:      refs,
				CandidateBudget: CandidateLimit,
			},
			Options: retrieval.RetrieveOptions{TopK: TopK},
		},
	)
	if err != nil {
		return tRead, nil, err
	}
	ranked := result.Documents.Documents()
	if row != nil {
		for _, doc := range ranked {
			row.LocalOutputUnits += uint64(len(doc.ID) + task19ScoreBytes)
		}
	}
	return tRead, task19OriginalDocuments(c, ranked), nil
}

func task19OriginalDocuments(c Corpus, docs []retrieval.Document[Meta]) []retrieval.Document[Meta] {
	for i := range docs {
		locations := docs[i].SourceLocations()
		if len(locations) == 1 {
			for _, original := range c.Documents {
				if locations[0].Reference == Reference(c, original) {
					docs[i].ID = original.ID
					docs[i].SourceSupports = []source.Locator{Locator(c, original)}
					break
				}
			}
		}
	}
	return docs
}

// ManagedSetup records indexing separately from query resource policy.
type ManagedSetup struct {
	Nanos              int64  `json:"elapsed_nanos"`
	CorpusDigest       string `json:"corpus_digest"`
	Documents          int    `json:"documents"`
	EncoderCalls       int    `json:"local_encoder_calls"`
	EncoderInputBytes  int    `json:"local_encoder_input_bytes"`
	EncoderOutputBytes int    `json:"local_encoder_output_bytes"`
	DeadlineNanos      int64  `json:"deadline_nanos"`
}

const ManagedSetupDeadline = 60 * time.Second

//nolint:gochecknoglobals // Mutex-protected process registry owns only explicitly prepared fixture indexes.
var task19Prepared = struct {
	mu     sync.RWMutex
	stores map[string]task19Stores
	setups map[string]ManagedSetup
}{stores: map[string]task19Stores{}, setups: map[string]ManagedSetup{}}

func task19CorpusKey(c Corpus) string { data, _ := json.Marshal(c); return Digest(data) }

// PrepareDenseTensor creates an explicitly owned index once; no silent warm cache.
func PrepareDenseTensor(ctx context.Context, c Corpus) (func(), ManagedSetup, error) {
	key := task19CorpusKey(c)
	setup := ManagedSetup{CorpusDigest: key, DeadlineNanos: int64(ManagedSetupDeadline)}
	for _, d := range c.Documents {
		if d.Current {
			setup.Documents++
			setup.EncoderCalls++
			setup.EncoderInputBytes += len(d.Text)
			count := max(
				1,
				len(
					strings.FieldsFunc(
						strings.ToLower(d.Text),
						func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) },
					),
				),
			)
			setup.EncoderOutputBytes += (1 + count) * task19EmbeddingDimensions * task19Float32Bytes
		}
	}
	root, err := os.MkdirTemp("", "ragy-task19-prepared-")
	if err != nil {
		return nil, setup, err
	}
	started := time.Now()
	bounded, cancel := context.WithTimeout(ctx, ManagedSetupDeadline)
	defer cancel()
	p, err := task19BuildStores(bounded, root, c)
	setup.Nanos = time.Since(started).Nanoseconds()
	if err != nil {
		_ = os.RemoveAll(root)
		return nil, setup, err
	}
	task19Prepared.mu.Lock()
	task19Prepared.stores[key] = p
	task19Prepared.setups[key] = setup
	task19Prepared.mu.Unlock()
	cleanup := func() {
		task19Prepared.mu.Lock()
		delete(task19Prepared.stores, key)
		delete(task19Prepared.setups, key)
		task19Prepared.mu.Unlock()
		_ = os.RemoveAll(root)
	}
	return cleanup, setup, nil
}
func task19AcquireStores(ctx context.Context, c Corpus) (task19Stores, func(), error) {
	key := task19CorpusKey(c)
	task19Prepared.mu.RLock()
	p, ok := task19Prepared.stores[key]
	task19Prepared.mu.RUnlock()
	if ok {
		return p, func() {}, nil
	}
	root, err := os.MkdirTemp("", "ragy-task19-dense-tensor-")
	if err != nil {
		return p, func() {}, err
	}
	p, err = task19BuildStores(ctx, root, c)
	return p, func() { _ = os.RemoveAll(root) }, err
}

func PreparedSetup(c Corpus) (ManagedSetup, bool) {
	task19Prepared.mu.RLock()
	defer task19Prepared.mu.RUnlock()
	setup, ok := task19Prepared.setups[task19CorpusKey(c)]
	return setup, ok
}

const LocalInputByteLimit uint64 = InputBytes
const LocalOutputByteLimit uint64 = OutputBytes

func task19ReserveOperation(row *Row, input, output uint64, operation bool) error {
	if row == nil {
		return nil
	}
	if row.LocalInputUnits > LocalInputByteLimit || input > LocalInputByteLimit-row.LocalInputUnits ||
		row.LocalOutputUnits > LocalOutputByteLimit ||
		output > LocalOutputByteLimit-row.LocalOutputUnits {
		return errors.New("local serialized-byte budget exhausted")
	}
	if operation && row.ModelCalls+row.EncoderCalls+row.RerankerCalls >= ModelSlots {
		return errors.New("local operation slots exhausted")
	}
	return nil
}
