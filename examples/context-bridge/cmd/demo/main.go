// Command demo runs an offline canonical context composition and managed Forget.
package main

import (
	"context"
	"fmt"
	"os"
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

type metadata struct{ Ref bridge.Reference }
type uncertainty struct {
	State string `json:"state"`
}

const uncertaintyType = "demo.extractor-uncertainty/1"
const canonicalID = "canonical"
const candidateLimit = 8
const renderLimit = 1000
const textLimit = 10000
const jsonLimit = 100000
const cleanupLimit = 100
const cleanupBytes = 1 << 20
const measurementProfile = "demo-codepoint-counter"

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(ctx context.Context) error {
	scope := memy.Scope{Tenant: "demo", Namespace: "knowledge", Subject: "reader"}
	clock := reference.NewClock(time.Date(2026, 10, 7, 0, 0, 0, 0, time.UTC))
	authority := reference.NewPolicy(func(actor string) string { return actor })
	authority.Grant(
		"reader",
		scope,
		"host-policy/1",
		memy.ActionRead,
		memy.ActionPropose,
		memy.ActionAccept,
		memy.ActionCommit,
		memy.ActionForget,
	)
	sources := reference.NewRegistry[string](memy.JSONCodec[string]{})
	src := memy.Source[string]{ID: "source", Revision: "retained-1", Reference: "host://source"}
	if err := sources.Put(scope, src); err != nil {
		return err
	}
	sink, err := bridge.NewSnapshotSink[uncertainty]("context", uncertaintyType)
	if err != nil {
		return err
	}
	store := memory.New()
	defer func() { _ = store.Close() }()
	engine, err := memy.New(
		memy.Config[string, string, string]{
			Store:     store,
			Authority: authority,
			Sources:   sources,
			Clock:     clock,
			Retention: reference.Retain[string]{
				Version: "retain/1",
			},
			PayloadCodec:   memy.JSONCodec[string]{},
			ReferenceCodec: memy.JSONCodec[string]{},
			Sinks:          []memy.Sink{sink},
		},
	)
	if err != nil {
		return err
	}
	receipt, err := commitDemo(ctx, engine, scope, src, clock)
	if err != nil {
		return err
	}
	mapper := newMapper(engine, scope, src, receipt, sink)
	output, err := mapper.Run(ctx)
	if err != nil {
		return err
	}
	fmt.Println(string(output.Public))
	decoded, err := bridge.Decode[uncertainty](ctx, output.Durable, bridge.Registry[uncertainty](uncertaintyType))
	if err != nil {
		return err
	}
	fmt.Printf(
		"roundtrip references=%d precision=%s score-state=%d\n",
		len(decoded.Evidence.References),
		decoded.Evidence.Precision,
		decoded.Evidence.Inputs[0].ScoreState,
	)
	purge, err := engine.Forget(
		ctx,
		"reader",
		scope,
		"assist",
		memy.ForgetRequest{
			OperationID:   "forget",
			Selector:      memy.Selector{Kind: memy.SelectRecord, ID: canonicalID},
			Expected:      []memy.RevisionRef{{RecordID: canonicalID, Revision: receipt.Revision}},
			Reason:        "host deletion",
			PolicyVersion: "delete/1",
			Limit:         cleanupLimit,
			MaxBytes:      cleanupBytes,
		},
	)
	if err != nil {
		return err
	}
	fmt.Printf("forget=%s managed-contexts=%d\n", purge.State, len(sink.Snapshots()))
	if _, err := mapper.Run(ctx); err == nil {
		return memy.ErrInvalid
	}
	return nil
}

func commitDemo(
	ctx context.Context,
	engine *memy.Engine[string, string, string],
	scope memy.Scope,
	src memy.Source[string],
	clock *reference.Clock,
) (memy.CommitReceipt, error) {
	proposal, err := engine.Remember(
		ctx,
		"reader",
		scope,
		"proposal",
		"assist",
		memy.Suggestion[string, string]{
			Payload:    "Канонический текст 😀",
			Sources:    []memy.Source[string]{src},
			Evidence:   "host retained source",
			Extractor:  "manual/1",
			ObservedAt: clock.Now(),
			Valid:      memy.Interval{Known: true, From: clock.Now()},
		},
	)
	if err != nil {
		return memy.CommitReceipt{}, err
	}
	accepted, err := engine.Accept(ctx, "reader", scope, proposal.ID, proposal.Digest, proposal.Revision, "assist")
	if err != nil {
		return memy.CommitReceipt{}, err
	}
	receipt, err := engine.Commit(
		ctx,
		"reader",
		scope,
		"assist",
		memy.CommitRequest{
			OperationID: "commit",
			ProposalID:  proposal.ID,
			Acceptance:  accepted,
			RecordID:    canonicalID,
			Reconcile:   memy.Reconciliation{Mode: memy.Append, PolicyVersion: "append/1", Basis: "host approval"},
		},
	)
	if err != nil {
		return memy.CommitReceipt{}, err
	}

	return receipt, nil
}

func newMapper(
	engine *memy.Engine[string, string, string],
	scope memy.Scope,
	src memy.Source[string],
	receipt memy.CommitReceipt,
	sink *bridge.SnapshotSink[uncertainty],
) bridge.Bridge[string, string, string, metadata, uncertainty] {
	return bridge.Bridge[string, string, string, metadata, uncertainty]{
		Engine:        engine,
		Authority:     "reader",
		Scope:         scope,
		Purpose:       "assist",
		Read:          retrieval.UnrestrictedRead(),
		MaxCandidates: candidateLimit,
		Retrieve: func(ctx context.Context) (bridge.Batch[metadata], error) {
			return bridge.Batch[metadata]{
				Documents: []retrieval.Document[metadata]{
					{
						ID:      "index-artifact",
						Content: "untrusted stale index content",
						Rank:    1,
						Meta: metadata{
							Ref: bridge.Reference{Scope: scope, RecordID: receipt.RecordID, Revision: receipt.Revision},
						},
					},
				},
				Coverage: []memy.Coverage{{Backend: "retrieval", Status: "ready"}},
			}, ctx.Err()
		},
		Map: func(ctx context.Context, d retrieval.Document[metadata]) (bridge.Reference, error) {
			return d.Meta.Ref, ctx.Err()
		},
		Project: func(ctx context.Context, r memy.Record[string, string]) (retrieval.Document[metadata], error) {
			loc := source.Locator{
				Reference: source.Reference{
					Namespace:         r.Scope.Key(),
					Source:            src.ID,
					Revision:          src.Revision,
					Transformation:    "identity",
					AccessFingerprint: "host-policy/1",
					Artifact:          r.ID,
					Representation:    "utf8",
				},
				Kind: source.TextLocation,
				Span: source.ByteSpan{Start: 0, End: len(r.Payload)},
			}
			mapping, err := source.OriginalText(loc, r.Payload)
			if err != nil {
				return retrieval.Document[metadata]{}, err
			}
			return retrieval.Document[metadata]{ID: r.ID, Content: r.Payload, SourceMapping: mapping}, ctx.Err()
		},
		Options: bridge.Options[metadata]{
			UncertaintyType: uncertaintyType,
			MessageID:       "context-message",
			Role:            contexty.RoleUser,
			Prefix:          "<data>\n",
			Suffix:          "\n</data>",
			Render: retrieval.ArtifactRenderOptions[metadata]{
				Resource:  retrieval.RuneResource(renderLimit),
				CloneMeta: func(m metadata) (metadata, error) { return m, nil },
			},
			Limits: bridge.Limits{
				Bytes:     textLimit,
				Runes:     textLimit,
				JSONBytes: jsonLimit,
				Tokens:    textLimit,
				Tokenizer: measurementProfile,
				MeasureTokens: func(ctx context.Context, text string) (int64, error) {
					return int64(utf8.RuneCountInString(text)), ctx.Err()
				},
			},
		},
		Publish: sink.Publish,
	}
}
