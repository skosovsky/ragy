package consumer_test

import (
	"context"
	"testing"
	"time"

	"go.opentelemetry.io/otel/trace/noop"

	"github.com/skosovsky/ragy/access"
	ragyotel "github.com/skosovsky/ragy/adapters/observability/otel"
	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

func TestExternalTypedDecoratorConformance(t *testing.T) {
	// Arrange: host metadata and actual scoped BM25/payload boundary from newFixture.
	var compositions []contracttest.ReadDecoratorFixture[searchIntent, requestMeta, sourceMeta]
	for _, order := range []string{"plain", "ctp", "cpt", "tcp", "tpc", "pct", "ptc"} {
		compositions = append(
			compositions,
			contracttest.ReadDecoratorFixture[searchIntent, requestMeta, sourceMeta]{
				Name: order,
				Decorate: func(t *testing.T, next retrieval.RequestBackend[searchIntent, requestMeta, sourceMeta]) retrieval.RequestBackend[searchIntent, requestMeta, sourceMeta] {
					return decorateTyped(t, next, order)
				},
			},
		)
	}
	// Act/Assert: public suite independently invokes raw composed backends for all cases.
	contracttest.RunScopedReadDecoratorSuite(t, newFixture, compositions)
}

func decorateTyped(
	t *testing.T,
	next retrieval.RequestBackend[searchIntent, requestMeta, sourceMeta],
	order string,
) retrieval.RequestBackend[searchIntent, requestMeta, sourceMeta] {
	t.Helper()
	if order == "plain" {
		return next
	}
	for _, kind := range order {
		switch kind {
		case 'c':
			store, err := retrieval.NewMemoryCache(
				16,
				time.Now,
				func(m sourceMeta) (sourceMeta, error) { return m, nil },
			)
			if err != nil {
				t.Fatal(err)
			}
			cached, err := retrieval.NewCachedBackend(retrieval.CacheConfig[searchIntent, requestMeta, sourceMeta]{
				Next:      next,
				Store:     store,
				TTL:       time.Minute,
				Now:       time.Now,
				CloneMeta: func(m sourceMeta) (sourceMeta, error) { return m, nil },
				SnapshotRequest: func(r retrieval.Request[searchIntent, requestMeta]) (retrieval.Request[searchIntent, requestMeta], error) {
					return retrieval.CopyRequestOptions(r), nil
				},
				Identity: func(context.Context, retrieval.Request[searchIntent, requestMeta]) (retrieval.CacheIdentity, error) {
					return retrieval.CacheIdentity{
						Index:         "fixture",
						IndexRevision: "r1",
						Recipe:        "direct",
						Configuration: "qa",
						Capabilities:  []string{"scope"},
					}, nil
				},
				HostIdentity: func(r retrieval.Request[searchIntent, requestMeta]) ([]byte, error) {
					return []byte(r.Intent.Collection + "/" + r.Meta.Correlation), nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			next = cached
		case 't':
			traced, err := ragyotel.WrapRequestBackend(next, noop.NewTracerProvider().Tracer("qa"))
			if err != nil {
				t.Fatal(err)
			}
			next = traced
		case 'p':
			next = retrieval.ProjectedBackend[searchIntent, requestMeta, searchIntent, requestMeta, sourceMeta]{
				Next:    next,
				Project: retrieval.CopyRequestOptions[searchIntent, requestMeta],
			}
		}
	}
	return next
}

type richIntent struct {
	Collection string
	Extra      string
}

func TestCrossTypeProjectionDecoratorPermutations(t *testing.T) {
	for _, order := range []string{"plain", "ctp", "cpt", "tcp", "tpc", "pct", "ptc"} {
		t.Run(order, func(t *testing.T) {
			// Arrange: actual scoped host backend requires its typed intent/meta.
			f := newFixture(t)
			project := func(r retrieval.Request[richIntent, requestMeta]) retrieval.Request[searchIntent, requestMeta] {
				return retrieval.Request[searchIntent, requestMeta]{
					Read:    r.Read,
					Intent:  searchIntent{Collection: r.Intent.Collection},
					Meta:    r.Meta,
					Text:    r.Text,
					Options: r.Options,
				}
			}
			b := retrieval.ProjectedBackend[richIntent, requestMeta, searchIntent, requestMeta, sourceMeta]{
				Next:             decorateTyped(t, f.Backend, order),
				Project:          project,
				AdmissionProject: project,
			}
			req := retrieval.Request[richIntent, requestMeta]{
				Read:    f.Request.Read,
				Intent:  richIntent{Collection: f.Request.Intent.Collection},
				Meta:    f.Request.Meta,
				Text:    f.Request.Text,
				Options: f.Request.Options,
			}
			node := retrieval.RequestBackendNode[richIntent, requestMeta, sourceMeta, retrieval.NoExecutionMeta]{
				Backend: b,
			}
			// Act.
			_, err := retrieval.InspectRead(t.Context(), req, node)
			result, dispatchErr := node.Execute(t.Context(), req, retrieval.NoExecutionMeta{})
			// Assert: successful admission/dispatch is independent of decorator order.
			if err != nil || dispatchErr != nil || result.Len() != 1 {
				t.Fatalf("cross-type composition: %v %v", err, dispatchErr)
			}
			b.AdmissionProject = nil
			before := f.IOCount()
			_, err = retrieval.InspectRead(
				t.Context(),
				req,
				retrieval.RequestBackendNode[richIntent, requestMeta, sourceMeta, retrieval.NoExecutionMeta]{
					Backend: b,
				},
			)
			if !access.IsUnsupportedCapability(err) || f.IOCount() != before {
				t.Fatalf("implicit cross-type admission: %v", err)
			}
		})
	}
}
