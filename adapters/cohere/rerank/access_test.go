package rerank

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

func modelRead(t *testing.T, revoked *atomic.Bool) access.Binding {
	t.Helper()
	builder := filter.NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	predicates, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(predicates, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	binding, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "host-policy-7",
				PolicyEpoch: 7,
				IssuedAt:    now,
				ExpiresAt:   now.Add(30 * time.Second),
			},
			Schema:      schema,
			Mandatory:   mandatory,
			Publication: access.CurrentPublication(),
			Now:         func() time.Time { return now },
			Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
				if revoked.Load() {
					return ragy.ErrUnavailable
				}
				return nil
			}),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return binding
}

func TestModelRerankerGateBeforeDispatchAndAfterIO(t *testing.T) {
	for _, alreadyRevoked := range []bool{true, false} {
		t.Run(map[bool]string{true: "before-dispatch", false: "during-http"}[alreadyRevoked], func(t *testing.T) {
			// Arrange.
			var revoked atomic.Bool
			revoked.Store(alreadyRevoked)
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				revoked.Store(true)
				_, _ = w.Write([]byte(`{"results":[{"index":0,"relevance_score":2}]}`))
			}))
			defer server.Close()
			client, err := New[struct{}](Config{APIKey: "fixture-key", Model: "fixture-model", BaseURL: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			read := modelRead(t, &revoked)
			rs := retrieval.NewResultSet([]retrieval.Document[struct{}]{{ID: "allowed", Content: "policy"}}, nil)
			// Act.
			out, err := client.Rerank(context.Background(), read, "policy", rs)
			// Assert: a model may run only while authorized; revoked output is never delivered.
			wantCalls := int64(1)
			if alreadyRevoked {
				wantCalls = 0
			}
			if !access.IsProtectionFailure(err) || !errors.Is(err, ragy.ErrUnavailable) || !out.IsEmpty() ||
				calls.Load() != wantCalls {
				t.Fatalf("model gate bypassed: err=%v, calls=%d, len=%d", err, calls.Load(), out.Len())
			}
		})
	}
}
