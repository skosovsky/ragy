package neo4j

import (
	"context"
	"errors"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/retrieval"
)

type deliveryRunner struct {
	snapshot graph.Snapshot[contracttest.StructMeta]
	err      error
	cancel   context.CancelFunc
	calls    int
}

func (r *deliveryRunner) Traverse(context.Context, Query) (graph.Snapshot[contracttest.StructMeta], error) {
	r.calls++
	if r.cancel != nil {
		r.cancel()
	}
	return r.snapshot, r.err
}
func (*deliveryRunner) Upsert(context.Context, graph.Snapshot[contracttest.StructMeta]) error {
	return nil
}

type privateRunnerError struct{ secret string }

func (e *privateRunnerError) Error() string { return e.secret }
func (*privateRunnerError) Unwrap() error   { return ragy.ErrProtocol }

func TestRetrieveFinalDelivery(t *testing.T) {
	valid := graph.Node[contracttest.StructMeta]{ID: "ok", Content: "good"}
	private := &privateRunnerError{secret: "private runner payload"}
	for _, test := range []deliveryCase{
		{name: "success", nodes: []graph.Node[contracttest.StructMeta]{valid}, wantLen: 1},
		{name: "empty"},
		{name: "partial", nodes: []graph.Node[contracttest.StructMeta]{valid, {ID: "", Content: "bad"}}, wantLen: 1, wantErr: ragy.ErrProtocol},
		{name: "runner_error", runnerErr: private, wantErr: ragy.ErrProtocol},
	} {
		for _, canceled := range []bool{false, true} {
			name := test.name + "/active"
			if canceled {
				name = test.name + "/canceled"
			}
			t.Run(name, func(t *testing.T) { checkDeliveryCase(t, test, canceled, private) })
		}
	}
}

func TestRetrievePreCanceledNoTraversal(t *testing.T) {
	// Arrange.
	runner := &deliveryRunner{}
	store, err := New(runner, graph.EmptySchema(), Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	// Act.
	result, err := retrieveStore(ctx, store, "", retrieval.RetrieveOptions{})
	// Assert.
	if result.Len() != 0 || !errors.Is(err, context.Canceled) || !access.IsProtectionFailure(err) || runner.calls != 0 {
		t.Fatal(result, err, runner.calls)
	}
}

type deliveryCase struct {
	name      string
	nodes     []graph.Node[contracttest.StructMeta]
	runnerErr error
	wantLen   int
	wantErr   error
}

func checkDeliveryCase(t *testing.T, test deliveryCase, canceled bool, private *privateRunnerError) {
	t.Helper()
	// Arrange.
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	runner := &deliveryRunner{snapshot: graph.Snapshot[contracttest.StructMeta]{Nodes: test.nodes}, err: test.runnerErr}
	if canceled {
		runner.cancel = cancel
	}
	store, err := New(runner, graph.EmptySchema(), Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := retrieveStore(
		ctx,
		store,
		"",
		retrieval.RetrieveOptions{
			TopK:  5,
			Graph: &retrieval.GraphOptions{Seeds: []string{"ok"}, Direction: graph.DirectionOutbound, Depth: 1},
		},
	)
	// Assert.
	if runner.calls != 1 {
		t.Fatal("traversal count", runner.calls)
	}
	if test.wantErr != nil && !errors.Is(err, test.wantErr) {
		t.Fatal("lost observed cause", err)
	}
	if canceled {
		assertCanceledDelivery(t, result, err, test.runnerErr, private)
		return
	}

	if result.Len() != test.wantLen || (test.wantErr == nil && err != nil) {
		t.Fatal(result, err)
	}
	if result.Len() > 0 &&
		(result.Documents()[0].Rank != 1 || result.Documents()[0].ScoreState != retrieval.ScoreAbsent) {
		t.Fatal("projection control", result)
	}
}

func assertCanceledDelivery(
	t *testing.T,
	result retrieval.ResultSet[contracttest.StructMeta],
	err, runnerErr error,
	private *privateRunnerError,
) {
	t.Helper()

	if result.Len() != 0 || !access.IsProtectionFailure(err) || !errors.Is(err, context.Canceled) {
		t.Fatal("late delivery", result, err)
	}
	var payload *privateRunnerError
	if errors.As(err, &payload) || strings.Contains(err.Error(), private.secret) {
		t.Fatal("private error leaked", err)
	}
	if runnerErr != nil && !errors.Is(err, runnerErr) {
		t.Fatal("lost runner identity", err)
	}
}
