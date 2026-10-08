package bridge_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/skosovsky/contexty"
	"github.com/skosovsky/memy"
	"github.com/skosovsky/memy/reference"
	"github.com/skosovsky/memy/store/memory"

	bridge "github.com/skosovsky/ragy/examples/context-bridge"

	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type metadata struct {
	Canonical bridge.Reference
	Private   []string
}
type knowledge struct {
	Text    string            `json:"text"`
	Mapping source.MappedText `json:"mapping"`
	Private []string          `json:"private"`
}
type uncertainty struct {
	State   string   `json:"state"`
	Reasons []string `json:"reasons"`
}
type sink struct {
	mu      sync.Mutex
	data    map[string][]byte
	fail    bool
	calls   int
	cleanup func(context.Context, memy.PurgeBatch) error
}

func (s *sink) Name() string { return "context" }
func (s *sink) Purge(ctx context.Context, b memy.PurgeBatch) (memy.PurgeAck, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	if err := ctx.Err(); err != nil {
		return memy.PurgeAck{}, err
	}
	if s.fail {
		return memy.PurgeAck{}, memy.ErrUnavailable
	}
	if s.cleanup != nil {
		if err := s.cleanup(ctx, b); err != nil {
			return memy.PurgeAck{}, err
		}
	}
	for id, raw := range s.data {
		out, err := bridge.Decode[uncertainty](ctx, raw, bridge.Registry[uncertainty]("host.uncertainty/1"))
		if err != nil {
			return memy.PurgeAck{}, err
		}
		if out.Evidence.Scope != b.Scope {
			continue
		}
		for _, ref := range out.Evidence.References {
			for _, record := range b.Records {
				if ref.RecordID == record {
					delete(s.data, id)
				}
			}
		}
	}
	return memy.PurgeAck{Sink: s.Name(), OperationID: b.OperationID, Epoch: b.Epoch, Chunk: b.Chunk}, nil
}
func (s *sink) publish(ctx context.Context, out bridge.Published[uncertainty]) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[out.Message.ID] = append([]byte(nil), out.Durable...)
	return nil
}

type observedStore struct {
	memy.Store

	updateAttempt chan struct{}
	updateOnce    sync.Once
}

func (s *observedStore) Update(ctx context.Context, scope memy.Scope, fn func(memy.Bucket) error) error {
	if s.updateAttempt != nil {
		s.updateOnce.Do(func() { close(s.updateAttempt) })
	}
	return s.Store.Update(ctx, scope, fn)
}

type fixture struct {
	store    *observedStore
	b        bridge.Bridge[knowledge, string, string, metadata, uncertainty]
	clock    *reference.Clock
	sources  *reference.Registry[string]
	policy   *reference.Policy[string]
	sink     *sink
	batch    bridge.Batch[metadata]
	receipts []memy.CommitReceipt
	u        *uncertainty
}

func check(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func newFixture(t testing.TB) *fixture {
	t.Helper()
	ctx := t.Context()
	scope := memy.Scope{Tenant: "tenant", Namespace: "knowledge", Subject: "reader"}
	clock := reference.NewClock(time.Date(2026, 10, 7, 0, 0, 0, 0, time.UTC))
	policy := reference.NewPolicy(func(a string) string { return a })
	policy.Grant(
		"reader",
		scope,
		"authority/1",
		memy.ActionPropose,
		memy.ActionAccept,
		memy.ActionCommit,
		memy.ActionRead,
		memy.ActionForget,
	)
	sources := reference.NewRegistry[string](memy.JSONCodec[string]{})
	store := &observedStore{Store: memory.New()}
	t.Cleanup(func() { check(t, store.Close()) })
	sink := &sink{data: map[string][]byte{}}
	engine, err := memy.New(
		memy.Config[knowledge, string, string]{
			Store:     store,
			Authority: policy,
			Sources:   sources,
			Clock:     clock,
			Retention: reference.Retain[knowledge]{
				Version: "retain/1",
			},
			PayloadCodec:   memy.JSONCodec[knowledge]{},
			ReferenceCodec: memy.JSONCodec[string]{},
			Sinks:          []memy.Sink{sink},
		},
	)
	check(t, err)
	f := &fixture{store: store,
		clock:   clock,
		sources: sources,
		policy:  policy,
		sink:    sink,
		u:       &uncertainty{State: "unknown", Reasons: []string{"extractor did not attest"}},
	}
	f.b = bridge.Bridge[knowledge, string, string, metadata, uncertainty]{
		Engine:        engine,
		Authority:     "reader",
		Scope:         scope,
		Purpose:       "assist",
		Read:          retrieval.UnrestrictedRead(),
		MaxCandidates: 16,
		Map: func(ctx context.Context, d retrieval.Document[metadata]) (bridge.Reference, error) {
			return d.Meta.Canonical, ctx.Err()
		},
		Project: func(ctx context.Context, r memy.Record[knowledge, string]) (retrieval.Document[metadata], error) {
			return retrieval.Document[metadata]{
				ID:            r.ID,
				Content:       r.Payload.Text,
				Meta:          metadata{Private: r.Payload.Private},
				SourceMapping: r.Payload.Mapping,
			}, ctx.Err()
		},
		Uncertainty: func(ctx context.Context, _ memy.Record[knowledge, string]) (*uncertainty, error) {
			return f.u, ctx.Err()
		},
		Options: bridge.Options[metadata]{
			UncertaintyType: "host.uncertainty/1",
			MessageID:       "message",
			Role:            contexty.RoleUser,
			Prefix:          "<context>\n",
			Suffix:          "\n</context>",
			Render: retrieval.ArtifactRenderOptions[metadata]{
				Resource:  retrieval.RuneResource(10000),
				CloneMeta: func(m metadata) (metadata, error) { m.Private = append([]string(nil), m.Private...); return m, nil },
			},
			Limits: bridge.Limits{
				Bytes:     100000,
				Runes:     100000,
				JSONBytes: 1000000,
				Tokens:    100000,
				Tokenizer: "fixture-code-points",
				MeasureTokens: func(ctx context.Context, text string) (int64, error) {
					return int64(utf8.RuneCountInString(text)), ctx.Err()
				},
			},
		},
		Publish: sink.publish,
	}
	for i := range 2 {
		id := fmt.Sprintf("record-%d", i)
		src := memy.Source[string]{
			ID:        fmt.Sprintf("source-%d", i),
			Revision:  "revision-1",
			Reference: "host-retained-text",
		}
		check(t, sources.Put(scope, src))
		text := "Привет 😀 \"quote\" \\ повтор повтор"
		loc := source.Locator{
			Reference: source.Reference{
				Namespace:         scope.Key(),
				Source:            src.ID,
				Revision:          src.Revision,
				Transformation:    "identity",
				AccessFingerprint: "authority/1",
				Artifact:          id,
				Representation:    "utf8",
			},
			Kind: source.TextLocation,
			Span: source.ByteSpan{Start: 0, End: len(text)},
		}
		mapping, err := source.OriginalText(loc, text)
		check(t, err)
		payload := knowledge{Text: text, Mapping: mapping, Private: []string{"PRIVATE-METADATA-MARKER"}}
		proposal, err := engine.Remember(
			ctx,
			"reader",
			scope,
			"propose-"+id,
			"assist",
			memy.Suggestion[knowledge, string]{
				Payload:       payload,
				Sources:       []memy.Source[string]{src},
				Evidence:      "retained source",
				Extractor:     "host/1",
				Losses:        []string{"canonical-loss"},
				Uncertainties: []string{"canonical-uncertainty"},
				ObservedAt:    clock.Now(),
				Valid:         memy.Interval{Known: true, From: clock.Now()},
			},
		)
		check(t, err)
		acceptance, err := engine.Accept(
			ctx,
			"reader",
			scope,
			proposal.ID,
			proposal.Digest,
			proposal.Revision,
			"assist",
		)
		check(t, err)
		receipt, err := engine.Commit(
			ctx,
			"reader",
			scope,
			"assist",
			memy.CommitRequest{
				OperationID: "commit-" + id,
				RecordID:    id,
				ProposalID:  proposal.ID,
				Acceptance:  acceptance,
				Reconcile: memy.Reconciliation{
					Mode:          memy.Append,
					PolicyVersion: "append/1",
					Basis:         "explicit fixture",
				},
			},
		)
		check(t, err)
		f.receipts = append(f.receipts, receipt)
		f.batch.Documents = append(
			f.batch.Documents,
			retrieval.Document[metadata]{
				ID:      "index-" + id,
				Content: "HOSTILE INDEX PAYLOAD",
				Rank:    2,
				Meta: metadata{
					Canonical: bridge.Reference{Scope: scope, RecordID: id, Revision: receipt.Revision},
					Private:   []string{"INDEX-PRIVATE"},
				},
			},
		)
	}
	f.batch.Coverage = []memy.Coverage{{Backend: "retrieval", Status: "ready"}}
	f.b.Retrieve = func(ctx context.Context) (bridge.Batch[metadata], error) { return f.batch, ctx.Err() }
	return f
}
func (f *fixture) forget(ctx context.Context) (memy.PurgeReceipt, error) {
	return f.b.Engine.Forget(
		ctx,
		f.b.Authority,
		f.b.Scope,
		f.b.Purpose,
		memy.ForgetRequest{
			OperationID: "forget-record",
			Selector:    memy.Selector{Kind: memy.SelectRecord, ID: "record-0"},
			Expected: []memy.RevisionRef{
				{RecordID: "record-0", Revision: f.receipts[0].Revision},
			},
			Reason:        "host deletion",
			PolicyVersion: "delete/1",
			Limit:         100,
			MaxBytes:      1 << 20,
		},
	)
}

func TestCanonicalPublicationAndDurableOwnership(t *testing.T) {
	// Arrange: hostile index payload and host-owned mutable uncertainty.
	f := newFixture(t)
	// Act: real canonical Recall, renderer and message extension codec.
	out, err := f.b.Run(t.Context())
	check(t, err)
	decoded, err := bridge.Decode[uncertainty](
		t.Context(),
		out.Durable,
		bridge.Registry[uncertainty]("host.uncertainty/1"),
	)
	check(t, err)
	f.u.Reasons[0] = "mutated"
	f.batch.Documents[0].Meta.Private[0] = "mutated"
	// Assert: canonical payload and independent evidence snapshots, no public private data.
	if !reflect.DeepEqual(decoded.Evidence, out.Evidence) || len(f.sink.data) != 1 ||
		strings.Contains(string(out.Public), "PRIVATE") || strings.Contains(string(out.Durable), "PRIVATE") ||
		strings.Contains(out.Evidence.Text, "HOSTILE") || out.Evidence.Inputs[0].Uncertainty.Reasons[0] == "mutated" {
		t.Fatalf("bad snapshot/publication: %+v", out.Evidence)
	}
	for _, s := range out.Evidence.Snippets {
		if s.Span == nil || out.Evidence.Text[s.Span.Start:s.Span.End] != s.Content ||
			s.Mapping.Fragments()[0].Precision != source.ExactPrecision {
			t.Fatal("inexact citation")
		}
	}
}

func TestIdentityAndAuthorityFailures(t *testing.T) {
	for _, fault := range []string{"scope", "namespace", "missing", "stale", "duplicate", "source", "denied", "protocol"} {
		t.Run(fault, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			switch fault {
			case "scope":
				f.batch.Documents[0].Meta.Canonical.Scope.Tenant = "other"
			case "namespace":
				f.batch.Documents[0].Meta.Canonical.Scope.Namespace = "other"
			case "missing":
				f.batch.Documents[0].Meta.Canonical.RecordID = "missing"
			case "stale":
				f.batch.Documents[0].Meta.Canonical.Revision++
			case "duplicate":
				f.batch.Documents[1].Meta.Canonical = f.batch.Documents[0].Meta.Canonical
			case "source":
				check(
					t,
					f.sources.Put(
						f.b.Scope,
						memy.Source[string]{ID: "source-0", Revision: "revision-2", Reference: "host-retained-text"},
					),
				)
			case "denied":
				f.b.Authority = "intruder"
			case "protocol":
				f.batch.Coverage[0].Status = "invalid"
			}
			// Act.
			_, err := f.b.Run(t.Context())
			// Assert: no success or writes, including missing/stale canonical references.
			if err == nil || len(f.sink.data) != 0 {
				t.Fatalf("err=%v writes=%d", err, len(f.sink.data))
			}
		})
	}
}

func TestForgetBetweenRecallAndPublicationAndLateRetry(t *testing.T) {
	// Arrange: deterministic barrier after recall in renderer measurement.
	f := newFixture(t)
	entered := make(chan struct{})
	resume := make(chan struct{})
	measure := f.b.Options.Render.Resource.Measure
	var once sync.Once
	f.b.Options.Render.Resource.Measure = func(ctx context.Context, text string) (int64, error) {
		once.Do(func() { close(entered); <-resume })
		return measure(ctx, text)
	}
	result := make(chan error, 1)
	// Act: delete while publication is pending.
	go func() { _, err := f.b.Run(t.Context()); result <- err }()
	<-entered
	receipt, err := f.forget(t.Context())
	check(t, err)
	close(resume)
	runErr := <-result
	_, lateErr := f.b.Run(t.Context())
	// Assert: real canonical deletion/epoch prevents both queued and retried publication.
	if !receipt.CanonicalComplete || !errors.Is(runErr, memy.ErrStaleInput) || lateErr == nil || len(f.sink.data) != 0 {
		t.Fatalf("receipt=%+v run=%v late=%v", receipt, runErr, lateErr)
	}
}

func TestPublicationExcludedAgainstConcurrentForget(t *testing.T) {
	// Arrange: block inside the synchronous derived-write callback.
	f := newFixture(t)
	f.store.updateAttempt = make(chan struct{})
	entered := make(chan struct{})
	resume := make(chan struct{})
	f.b.Publish = func(ctx context.Context, p bridge.Published[uncertainty]) error {
		close(entered)
		<-resume
		return f.sink.publish(ctx, p)
	}
	published := make(chan error, 1)
	forgot := make(chan error, 1)
	// Act: Forget competes with an admitted publication, then purges its managed sink.
	go func() { _, err := f.b.Run(t.Context()); published <- err }()
	<-entered
	go func() { _, err := f.forget(t.Context()); forgot <- err }()
	<-f.store.updateAttempt // Forget reached the real Store.Update call while publication owns the scope fence.
	close(resume)
	check(t, <-published)
	check(t, <-forgot)
	// Assert.
	if len(f.sink.data) != 0 {
		t.Fatal("publication survived managed purge")
	}
}

func TestForgetSinkFailureAndIdempotentResume(t *testing.T) {
	// Arrange: a published context and unavailable managed sink.
	f := newFixture(t)
	_, err := f.b.Run(t.Context())
	check(t, err)
	f.sink.fail = true
	// Act.
	pending, err := f.forget(t.Context())
	check(t, err)
	_, staleErr := f.b.Run(t.Context())
	f.sink.fail = false
	complete, err := f.forget(t.Context())
	check(t, err)
	calls := f.sink.calls
	again, err := f.forget(t.Context())
	check(t, err)
	// Assert: canonical gate stays closed even when physical cleanup is pending.
	if pending.State != memy.PurgePending || staleErr == nil || complete.State != memy.PurgeComplete ||
		again.State != memy.PurgeComplete ||
		len(f.sink.data) != 0 ||
		f.sink.calls != calls {
		t.Fatalf("pending=%+v complete=%+v stale=%v", pending, complete, staleErr)
	}
}

func TestNativeScoreEvidenceAndRankPolicy(t *testing.T) {
	for _, state := range []string{"absent", "zero", "negative", "incompatible"} {
		t.Run(state, func(t *testing.T) {
			// Arrange: equal rank ties with reversed candidate order.
			f := newFixture(t)
			configureScores(f, state)
			f.batch.Documents[0], f.batch.Documents[1] = f.batch.Documents[1], f.batch.Documents[0]
			// Act.
			out, err := f.b.Run(t.Context())
			check(t, err)
			// Assert: original native states and observations are independent of derived order.
			if out.Evidence.RankPolicy != bridge.RankPolicy || out.Evidence.Inputs[0].Reference.RecordID != "record-0" {
				t.Fatal("unstable rank policy")
			}
			assertScores(t, f, out)
		})
	}
}

func TestRewriteAndTruncationCannotClaimExactSpans(t *testing.T) {
	for _, mode := range []string{"rewrite", "truncate"} {
		t.Run(mode, func(t *testing.T) {
			// Arrange: repeated UTF-8 source and JSON escaping already supplied by fixture.
			f := newFixture(t)
			if mode == "rewrite" {
				f.b.Options.Rewrite = func(ctx context.Context, s string) (string, error) {
					return strings.ReplaceAll(s, "повтор", "изменён"), ctx.Err()
				}
			} else {
				f.b.Options.TruncateRunes = 50
			}
			// Act.
			out, err := f.b.Run(t.Context())
			check(t, err)
			// Assert: source mappings remain source-only; final coordinates are absent.
			if out.Evidence.Precision != "unavailable" || !utf8.ValidString(out.Evidence.Text) {
				t.Fatal("bad precision")
			}
			for _, s := range out.Evidence.Snippets {
				if s.Span != nil || !s.DeliveryUncertain || len(s.Mapping.Supports()) == 0 {
					t.Fatal("lost support or stale exact span")
				}
			}
		})
	}
}

func TestCodecRejectsLossCorruptionAndUnknownSchema(t *testing.T) {
	for _, fault := range []string{"missing-codec", "missing-sidecar", "wrong-text", "span", "schema", "duplicate", "unknown", "nil-sidecar"} {
		t.Run(fault, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			out, err := f.b.Run(t.Context())
			check(t, err)
			raw := append([]byte(nil), out.Durable...)
			reg := bridge.Registry[uncertainty]("host.uncertainty/1")
			switch fault {
			case "missing-codec":
				reg = contexty.NewExtensionRegistry()
			case "duplicate":
				raw = append([]byte(`{"role":"user",`), raw[1:]...)
			default:
				var wire map[string]json.RawMessage
				check(t, json.Unmarshal(raw, &wire))
				switch fault {
				case "missing-sidecar":
					delete(wire, "extensions")
				case "nil-sidecar":
					wire["extensions"] = json.RawMessage(`[null]`)
				case "wrong-text":
					wire["parts"] = json.RawMessage(`[{"type":"text","text":"changed"}]`)
				default:
					var exts []map[string]json.RawMessage
					check(t, json.Unmarshal(wire["extensions"], &exts))
					var side map[string]json.RawMessage
					check(t, json.Unmarshal(exts[0]["payload"], &side))
					switch fault {
					case "span":
						side["snippets"] = json.RawMessage(
							`[{"document_id":"record-0","span":{"start":0,"end":999999}}]`,
						)
					case "schema":
						side["schema"] = json.RawMessage(`"future"`)
					case "unknown":
						side["unknown"] = json.RawMessage(`true`)
					}
					exts[0]["payload"], err = json.Marshal(side)
					check(t, err)
					wire["extensions"], err = json.Marshal(exts)
					check(t, err)
				}
				raw, err = json.Marshal(wire)
				check(t, err)
			}
			// Act.
			_, err = bridge.Decode[uncertainty](t.Context(), raw, reg)
			// Assert.
			if err == nil {
				t.Fatal("accepted corrupt/lossy context")
			}
		})
	}
}

func TestFinalResourceBudgetsAreIndependent(t *testing.T) {
	for _, unit := range []string{"bytes", "runes", "json", "tokens", "wrapped"} {
		t.Run(unit, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			switch unit {
			case "bytes":
				f.b.Options.Limits.Bytes = 60
			case "runes":
				f.b.Options.Limits.Runes = 30
			case "json":
				f.b.Options.Limits.JSONBytes = 1000
			case "tokens":
				f.b.Options.Limits.Tokens = 30
			case "wrapped":
				f.b.Options.Prefix = strings.Repeat("wrap", 100)
				f.b.Options.Limits.Tokens = 200
			}
			// Act.
			_, err := f.b.Run(t.Context())
			// Assert.
			if !errors.Is(err, memy.ErrBudget) || len(f.sink.data) != 0 {
				t.Fatalf("err=%v", err)
			}
		})
	}
}

func TestPartialOmissionsRendererPackingAndEmpty(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	f.batch.Partial = true
	f.batch.Omissions = 3
	f.batch.Coverage[0].Status = "degraded"
	f.b.Options.Render.Resource = retrieval.RuneResource(85)
	// Act.
	out, err := f.b.Run(t.Context())
	check(t, err)
	// Assert: no success claim hides omitted input, renderer packing or partial retrieval.
	if !out.Evidence.Partial || out.Evidence.Omissions != 3 ||
		out.Evidence.Resource.Packing == retrieval.ArtifactPackingComplete ||
		len(out.Evidence.Inputs) != 2 {
		t.Fatalf("%+v", out.Evidence)
	}
	// Arrange: true empty retrieval has coverage and unknown uncertainty, no writes.
	empty := newFixture(t)
	empty.batch.Documents = nil
	// Act.
	result, err := empty.b.Run(t.Context())
	check(t, err)
	// Assert.
	if len(result.Evidence.References) != 0 || len(empty.sink.data) != 0 {
		t.Fatal("empty publication")
	}
}

func TestCancellationAndDeadlineAtCooperativeBoundaries(t *testing.T) {
	for _, class := range []string{"cancel", "deadline"} {
		for _, stage := range []string{"retrieval", "mapping", "projection", "rewrite", "serialization", "publication"} {
			t.Run(class+"/"+stage, func(t *testing.T) {
				// Arrange: callback surfaces a cause with private error text.
				f := newFixture(t)
				sentinel := context.Canceled
				if class == "deadline" {
					sentinel = context.DeadlineExceeded
				}
				private := fmt.Errorf("PRIVATE-ERROR: %w", sentinel)
				switch stage {
				case "retrieval":
					f.b.Retrieve = func(context.Context) (bridge.Batch[metadata], error) { return f.batch, private }
				case "mapping":
					f.b.Map = func(context.Context, retrieval.Document[metadata]) (bridge.Reference, error) {
						return bridge.Reference{}, private
					}
				case "projection":
					f.b.Project = func(context.Context, memy.Record[knowledge, string]) (retrieval.Document[metadata], error) {
						return retrieval.Document[metadata]{}, private
					}
				case "rewrite":
					f.b.Options.Rewrite = func(context.Context, string) (string, error) { return "", private }
				case "serialization":
					f.b.Options.Limits.MeasureTokens = func(context.Context, string) (int64, error) { return 0, private }
				case "publication":
					f.b.Publish = func(context.Context, bridge.Published[uncertainty]) error { return private }
				}
				// Act.
				_, err := f.b.Run(t.Context())
				// Assert.
				if !errors.Is(err, sentinel) || strings.Contains(err.Error(), "PRIVATE") || len(f.sink.data) != 0 {
					t.Fatalf("%v", err)
				}
			})
		}
	}
}

func configureScores(f *fixture, state string) {
	switch state {
	case "absent":
	case "zero":
		for i := range f.batch.Documents {
			f.batch.Documents[i].ScoreState = retrieval.ScorePresent
			f.batch.Documents[i].ScoreSemantics = "native"
		}
	case "negative":
		for i := range f.batch.Documents {
			f.batch.Documents[i].ScoreState = retrieval.ScorePresent
			f.batch.Documents[i].ScoreSemantics = "native"
			f.batch.Documents[i].Score = -2
		}
	case "incompatible":
		for i := range f.batch.Documents {
			f.batch.Documents[i].ScoreState = retrieval.ScorePresent
			f.batch.Documents[i].ScoreSemantics = retrieval.ScoreSemantics(fmt.Sprintf("scale-%d", i))
			f.batch.Documents[i].Score = float64(i)
		}
	}
	for i := range f.batch.Documents {
		if f.batch.Documents[i].ScoreState.IsScored() {
			d := f.batch.Documents[i]
			f.batch.Documents[i].ScoreHistory = []retrieval.ScoreObservation{
				{DocumentID: d.ID, Value: d.Score, State: d.ScoreState, Semantics: d.ScoreSemantics},
			}
		}
	}
}

func assertScores(t *testing.T, f *fixture, out bridge.Published[uncertainty]) {
	t.Helper()
	for _, input := range out.Evidence.Inputs {
		for _, d := range f.batch.Documents {
			if d.ID == input.ArtifactID &&
				(input.ScoreState != d.ScoreState || input.Score != d.Score || input.ScoreSemantics != d.ScoreSemantics || !reflect.DeepEqual(input.ScoreHistory, d.ScoreHistory)) {
				t.Fatal("lost native evidence")
			}
		}
	}
}

func TestDurableFileAndPlainMarshalLoss(t *testing.T) {
	// Arrange: an actual encoded context saved and reloaded independently of the original object.
	f := newFixture(t)
	out, err := f.b.Run(t.Context())
	check(t, err)
	path := filepath.Join(t.TempDir(), "context.json")
	check(t, os.WriteFile(path, out.Durable, 0o600))
	// Act.
	raw, err := os.ReadFile(path)
	check(t, err)
	decoded, err := bridge.Decode[uncertainty](t.Context(), raw, bridge.Registry[uncertainty]("host.uncertainty/1"))
	check(t, err)
	lossy, err := json.Marshal(out.Message)
	check(t, err)
	_, lostErr := bridge.Decode[uncertainty](t.Context(), lossy, bridge.Registry[uncertainty]("host.uncertainty/1"))
	// Assert: codec roundtrip is lossless; plain JSON encoding cannot claim success.
	if !reflect.DeepEqual(decoded.Evidence, out.Evidence) || lostErr == nil {
		t.Fatal("durable/loss contract violated")
	}
}

func TestHostRoleAndUncertaintyIdentityFailClosed(t *testing.T) {
	for _, role := range []contexty.Role{contexty.RoleSystem, contexty.RoleAssistant, contexty.Role("developer"), contexty.Role("unknown")} {
		t.Run(string(role), func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			f.b.Options.Role = role
			// Act.
			_, err := f.b.Run(t.Context())
			// Assert.
			if err == nil || len(f.sink.data) != 0 {
				t.Fatal("retrieval promoted role")
			}
		})
	}
	// Arrange.
	f := newFixture(t)
	out, err := f.b.Run(t.Context())
	check(t, err)
	// Act.
	_, err = bridge.Decode[uncertainty](t.Context(), out.Durable, bridge.Registry[uncertainty]("different-contract"))
	// Assert.
	if err == nil {
		t.Fatal("uncertainty codec identity ignored")
	}
}

func TestSchemaRequiredNestedFields(t *testing.T) {
	for _, field := range []string{"score_state", "uncertainty", "reference"} {
		t.Run(field, func(t *testing.T) {
			// Arrange: retain valid relations except a missing required input field.
			f := newFixture(t)
			out, err := f.b.Run(t.Context())
			check(t, err)
			var wire map[string]json.RawMessage
			check(t, json.Unmarshal(out.Durable, &wire))
			var exts []map[string]json.RawMessage
			check(t, json.Unmarshal(wire["extensions"], &exts))
			var side map[string]json.RawMessage
			check(t, json.Unmarshal(exts[0]["payload"], &side))
			var inputs []map[string]json.RawMessage
			check(t, json.Unmarshal(side["inputs"], &inputs))
			delete(inputs[0], field)
			side["inputs"], err = json.Marshal(inputs)
			check(t, err)
			exts[0]["payload"], err = json.Marshal(side)
			check(t, err)
			wire["extensions"], err = json.Marshal(exts)
			check(t, err)
			raw, err := json.Marshal(wire)
			check(t, err)
			// Act.
			_, err = bridge.Decode[uncertainty](t.Context(), raw, bridge.Registry[uncertainty]("host.uncertainty/1"))
			// Assert: missing fields never silently receive Go zero defaults.
			if err == nil {
				t.Fatal("accepted missing nested field")
			}
		})
	}
}

func TestSnapshotSinkOwnershipAndScopedIdentity(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	out, err := f.b.Run(t.Context())
	check(t, err)
	sink, err := bridge.NewSnapshotSink[uncertainty]("managed", "host.uncertainty/1")
	check(t, err)
	// Act.
	check(t, sink.Publish(t.Context(), out))
	snapshots := sink.Snapshots()
	snapshots[0][0] = 'x'
	first := sink.Snapshots()[0]
	_, err = bridge.Decode[uncertainty](t.Context(), first, bridge.Registry[uncertainty]("host.uncertainty/1"))
	check(t, err)
	// Assert: returned snapshots cannot mutate stored bytes.
	if first[0] == 'x' {
		t.Fatal("sink alias")
	}
	// Act: canonical deletion acknowledgement and its replay.
	batch := memy.PurgeBatch{Scope: f.b.Scope, OperationID: "delete", Records: []string{"record-0"}, Epoch: 1}
	ack, err := sink.Purge(t.Context(), batch)
	check(t, err)
	again, err := sink.Purge(t.Context(), batch)
	check(t, err)
	// Assert.
	if ack != again || len(sink.Snapshots()) != 0 {
		t.Fatal("sink purge was not idempotent")
	}
}

func TestAuthorityRevokedAfterRecallBeforePublication(t *testing.T) {
	// Arrange: the renderer measurement is a deterministic post-Recall barrier.
	f := newFixture(t)
	entered := make(chan struct{})
	resume := make(chan struct{})
	measure := f.b.Options.Render.Resource.Measure
	var once sync.Once
	f.b.Options.Render.Resource.Measure = func(ctx context.Context, text string) (int64, error) {
		once.Do(func() { close(entered); <-resume })
		return measure(ctx, text)
	}
	result := make(chan error, 1)
	// Act: revoke the exact authority while materialization is pending.
	go func() { _, err := f.b.Run(t.Context()); result <- err }()
	<-entered
	f.policy.Grant("reader", f.b.Scope, "revoked-authority/2")
	close(resume)
	err := <-result
	// Assert: the real canonical publication gate rechecks authority and writes nothing.
	if !errors.Is(err, memy.ErrUnauthorized) || len(f.sink.data) != 0 {
		t.Fatalf("err=%v writes=%d", err, len(f.sink.data))
	}
}
