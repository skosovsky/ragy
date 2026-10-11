package retrieval_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type callbackRequestBackend struct {
	schema filter.Schema
	run    func() (retrieval.ResultSet[accessMeta], error)
}

func (b callbackRequestBackend) Schema() filter.Schema { return b.schema }

func (b callbackRequestBackend) ReadCapabilities() access.Capabilities {
	return access.Capabilities{ScopeProfile: true}
}

func (b callbackRequestBackend) Retrieve(
	context.Context,
	retrieval.Request[struct{}, struct{}],
) (retrieval.ResultSet[accessMeta], error) {
	return b.run()
}

type callbackExecutionBackend struct{ callbackRequestBackend }

func (b callbackExecutionBackend) Retrieve(
	_ context.Context,
	_ retrieval.Request[struct{}, struct{}],
	exec string,
) (retrieval.RetrievalResult[accessMeta, string], error) {
	rs, err := b.run()
	return retrieval.RetrievalResult[accessMeta, string]{
		ResultSet:   rs,
		Executed:    exec,
		Diagnostics: []retrieval.ExecutionDiagnostic{{Stage: "backend", Key: "private", Value: "private"}},
	}, err
}

func callbackNode(
	kind string,
	schema filter.Schema,
	run func() (retrieval.ResultSet[accessMeta], error),
) retrieval.RequestExecutionNode[struct{}, struct{}, accessMeta, string] {
	b := callbackRequestBackend{schema: schema, run: run}
	if kind == "backend" {
		return retrieval.RequestBackendNode[struct{}, struct{}, accessMeta, string]{Backend: b}
	}
	return retrieval.RequestExecutionRetrieverNode[struct{}, struct{}, accessMeta, string]{
		Backend: callbackExecutionBackend{b},
	}
}
func callbackDocuments() retrieval.ResultSet[accessMeta] {
	return retrieval.NewResultSet(
		[]retrieval.Document[accessMeta]{
			{ID: "private-document", Content: "private-payload", Meta: accessMeta{Tenant: "a", Visibility: "public"}},
		},
		nil,
	)
}
func callbackRequest(read access.Binding) retrieval.Request[struct{}, struct{}] {
	return retrieval.Request[struct{}, struct{}]{
		Read:    read,
		Text:    "policy",
		Options: retrieval.RetrieveOptions{TopK: 1},
	}
}

//nolint:gocognit // Keep the public-node/gate matrix and its AAA assertions together.
func TestCallbackErrorGateContracts(t *testing.T) {
	for _, kind := range []string{"backend", "execution"} {
		for _, gate := range []string{"none", "cancel", "revoke", "cancel-before"} {
			for _, failure := range []string{"success", "ordinary", "partial", "fusion"} {
				t.Run(kind+"/"+gate+"/"+failure, func(t *testing.T) {
					// Arrange.
					ctx, cancel := context.WithCancel(t.Context())
					defer cancel()
					cause, denied := errors.New("backend sentinel"), errors.New("authority revoked")
					var revoked atomic.Bool
					schema := newScopeFixture(t, allowAuthority()).schema
					read := access.Unrestricted()
					if gate == "revoke" {
						read = newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
							if revoked.Load() {
								return denied
							}
							return nil
						})).binding
					}
					docs := callbackDocuments()
					var callbackErr error
					switch failure {
					case "success":
					case "ordinary":
						callbackErr = cause
					case "partial":
						callbackErr = &retrieval.PartialFailureError[accessMeta]{Errors: []error{cause}, Result: docs}
					case "fusion":
						callbackErr = callbackFusionError(t, schema, docs, cause)
					}
					calls := 0
					node := callbackNode(kind, schema, func() (retrieval.ResultSet[accessMeta], error) {
						calls++
						if gate == "cancel" {
							cancel()
						}
						if gate == "revoke" {
							revoked.Store(true)
						}
						return docs, callbackErr
					})
					if gate == "cancel-before" {
						cancel()
					}
					// Act: public SDK only; no host context capture.
					result, err := node.Execute(ctx, callbackRequest(read), "private-exec")
					// Assert.
					wantCalls := 1
					if gate == "cancel-before" {
						wantCalls = 0
					}
					wantCause := failure != "success" && wantCalls > 0
					if calls != wantCalls || errors.Is(err, cause) != wantCause {
						t.Fatalf("calls=%d want=%d, callback cause want=%v: %v", calls, wantCalls, wantCause, err)
					}
					if gate == "none" {
						if (err == nil) != (failure == "success") || access.IsProtectionFailure(err) ||
							result.Len() != 1 ||
							result.Executed != "private-exec" {
							t.Fatalf("ordinary contract: result=%+v, err=%v", result, err)
						}
					} else {
						gateCause := context.Canceled
						if gate == "revoke" {
							gateCause = denied
						}
						assertCallbackProtection(t, result, err, gateCause)
					}
					assertCallbackDiagnostics(
						t,
						err,
						callbackErr,
						cause,
						gate != "none",
						failure == "fusion" && wantCalls > 0,
					)
				})
			}
		}
	}
}

func assertCallbackProtection(
	t *testing.T,
	result retrieval.RetrievalResult[accessMeta, string],
	err, gateCause error,
) {
	t.Helper()
	if !errors.Is(err, gateCause) || !access.IsProtectionFailure(err) {
		t.Fatalf("gate cause/protection lost: %v", err)
	}
	if result.ResultSet == nil || !result.IsEmpty() || result.Executed != "" || len(result.Diagnostics) != 0 ||
		len(result.BranchTrace) != 0 {
		t.Fatalf("protected payload leaked: %+v", result)
	}
	if strings.Contains(err.Error(), "private-payload") || strings.Contains(err.Error(), "private-document") {
		t.Fatalf("error exposed document: %v", err)
	}
}

//nolint:gocognit // Keep the public-node/gate matrix and its AAA assertions together.
func TestCallbackCauseConcurrentGate(t *testing.T) {
	for _, kind := range []string{"backend", "execution"} {
		for _, gate := range []string{"cancel", "revoke"} {
			t.Run(kind+"/"+gate, func(t *testing.T) {
				// Arrange.
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				cause, denied := errors.New("concurrent callback sentinel"), errors.New("authority revoked")
				var revoked atomic.Bool
				var calls atomic.Int64
				schema := newScopeFixture(t, allowAuthority()).schema
				read := access.Unrestricted()
				if gate == "revoke" {
					read = newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
						if revoked.Load() {
							return denied
						}
						return nil
					})).binding
				}
				entered, release := make(chan struct{}), make(chan struct{})
				node := callbackNode(kind, schema, func() (retrieval.ResultSet[accessMeta], error) {
					calls.Add(1)
					close(entered)
					<-release
					return callbackDocuments(), cause
				})
				type outcome struct {
					result retrieval.RetrievalResult[accessMeta, string]
					err    error
				}
				done := make(chan outcome, 1)
				// Act.
				go func() {
					result, err := node.Execute(ctx, callbackRequest(read), "private-exec")
					done <- outcome{result, err}
				}()
				select {
				case <-entered:
				case out := <-done:
					t.Fatalf("callback not dispatched: %v", out.err)
				}
				gateCause := context.Canceled
				if gate == "cancel" {
					cancel()
				} else {
					revoked.Store(true)
					gateCause = denied
				}
				close(release)
				out := <-done
				// Assert.
				assertCallbackProtection(t, out.result, out.err, gateCause)
				if !errors.Is(out.err, cause) || calls.Load() != 1 {
					t.Fatalf("cause/call contract: calls=%d err=%v", calls.Load(), out.err)
				}
			})
		}
	}
}

// TestCallbackCauseRescueLateGateParity records the consumer cleanup boundary.
// Successful rescue may leave a primary cause only in diagnostics. This bugfix
// does not promise an error history through later SDK or host delivery gates.
//
//nolint:gocognit // Keep the public-node/gate matrix and its AAA assertions together.
func TestCallbackCauseRescueLateGateParity(t *testing.T) {
	for _, kind := range []string{"backend", "execution"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange: replay each authority position of a successful native rescue,
			// including an additional host delivery gate, without host cause capture.
			cause, denied := errors.New("recovered primary cause"), errors.New("late revocation")
			run := func(revokeAt int) (int, bool, retrieval.ResultSet[accessMeta], error) {
				var checks atomic.Int64
				called := false
				fixture := newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
					n := checks.Add(1)
					if revokeAt > 0 && n >= int64(revokeAt) {
						return denied
					}
					return nil
				}))
				read, schema := fixture.binding, fixture.schema
				primary := callbackNode(kind, schema, func() (retrieval.ResultSet[accessMeta], error) {
					called = true
					return retrieval.NewResultSet[accessMeta](nil, nil), cause
				})
				secondary := callbackNode(
					kind,
					schema,
					func() (retrieval.ResultSet[accessMeta], error) { return callbackDocuments(), nil },
				)
				root := retrieval.RequestRescueNode[struct{}, struct{}, accessMeta, string]{
					Primary:   primary,
					Secondary: secondary,
				}
				pipeline, err := retrieval.NewRequestExecutionPipelineBuilder[struct{}, struct{}, accessMeta, string]().WithRoot(root).
					Build()
				if err != nil {
					t.Fatal(err)
				}
				result, err := pipeline.Execute(t.Context(), callbackRequest(read))
				rs, err := retrieval.DeliverRead(t.Context(), read, result.ResultSet, err, nil)
				return int(checks.Load()), called, rs, err
			}
			total, called, rs, err := run(0)
			if err != nil || !called || rs.IsEmpty() || total == 0 {
				t.Fatalf("rescue baseline: checks=%d called=%v err=%v", total, called, err)
			}
			preserved, lost := 0, 0
			// Act and Assert: every gate must fail closed, including host delivery.
			for check := 1; check <= total; check++ {
				_, called, rs, err = run(check)
				if !errors.Is(err, denied) || !access.IsProtectionFailure(err) || !rs.IsEmpty() {
					t.Fatalf("gate %d/%d: result=%v err=%v", check, total, rs, err)
				}
				if called {
					if errors.Is(err, cause) {
						preserved++
					} else {
						lost++
						t.Logf("recovered cause unavailable at gate %d/%d", check, total)
					}
				}
			}
			if preserved == 0 || lost == 0 || errors.Is(err, cause) {
				t.Fatalf(
					"expected immediate parity and late/host gap: preserved=%d lost=%d final=%v",
					preserved,
					lost,
					err,
				)
			}
			t.Logf(
				"cleanup blocked: checked %d gates, preserved=%d unavailable=%d after callback",
				total,
				preserved,
				lost,
			)
		})
	}
}

func assertCallbackDiagnostics(t *testing.T, err, callbackErr, cause error, protected, fusionCalled bool) {
	t.Helper()
	if partial, ok := retrieval.AsPartialFailure[accessMeta](err); ok {
		if partial.Result.IsEmpty() != protected {
			t.Fatal("incorrect partial diagnostic payload")
		}
		if original, ok := retrieval.AsPartialFailure[accessMeta](
			callbackErr,
		); !ok ||
			original.Result.IsEmpty() {
			t.Fatal("provider error mutated")
		}
	}
	if fusionCalled {
		view, ok := errors.AsType[*retrieval.FusionFailureError[accessMeta]](err)
		wantObservations := 1
		if protected {
			wantObservations = 0
		}
		if !ok || !errors.Is(view, cause) || len(view.Observations()) != wantObservations {
			t.Fatal("fusion diagnosis lost or exposed")
		}
		original, ok := errors.AsType[*retrieval.FusionFailureError[accessMeta]](callbackErr)
		if !ok || len(original.Observations()) != 1 {
			t.Fatal("provider fusion error mutated")
		}
	}
}

type callbackFailingMerger struct{ cause error }

func (m callbackFailingMerger) Merge(
	context.Context,
	...retrieval.ResultSet[accessMeta],
) (retrieval.ResultSet[accessMeta], error) {
	return nil, m.cause
}

func callbackFusionError(t *testing.T, schema filter.Schema, docs retrieval.ResultSet[accessMeta], cause error) error {
	t.Helper()
	node := retrieval.RequestAggregateNode[struct{}, struct{}, accessMeta, string]{
		Nodes: []retrieval.RequestExecutionNode[struct{}, struct{}, accessMeta, string]{
			callbackNode("backend", schema, func() (retrieval.ResultSet[accessMeta], error) { return docs, nil }),
		},
		Merger: callbackFailingMerger{cause: cause},
	}
	_, err := node.Execute(t.Context(), callbackRequest(access.Unrestricted()), "")
	view, ok := errors.AsType[*retrieval.FusionFailureError[accessMeta]](err)
	if !ok || !errors.Is(err, cause) || len(view.Observations()) != 1 {
		t.Fatalf("fusion fixture failed: %v", err)
	}
	return err
}
