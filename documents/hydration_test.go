package documents_test

import (
	"context"
	"errors"
	"maps"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type permissionMeta struct{ tenant, visibility string }
type payloadMeta map[string]string
type retainedHost struct {
	descriptors                   map[source.Reference]permissionMeta
	payloads                      map[source.Reference]retrieval.Document[payloadMeta]
	describeCalls, loadCalls      int
	afterDescribe, afterLoad      func()
	wrongDescriptor, wrongPayload bool
	mutateInput                   bool
	loaded                        []source.Reference
}

func (h *retainedHost) Describe(
	_ context.Context,
	req source.LookupRequest,
) ([]source.Descriptor[permissionMeta], error) {
	h.describeCalls++
	var out []source.Descriptor[permissionMeta]
	for _, ref := range req.References {
		if meta, exists := h.descriptors[ref]; exists {
			actual := ref
			if h.wrongDescriptor {
				actual.Revision = "r2"
			}
			out = append(out, source.Descriptor[permissionMeta]{Reference: actual, Access: meta})
		}
	}
	if h.mutateInput && len(req.References) > 0 {
		req.References[0].Revision = "r2"
		req.Read = retrieval.UnrestrictedRead()
		if req.Read.IsScoped() {
			return nil, ragy.ErrProtocol
		}
	}
	if h.afterDescribe != nil {
		h.afterDescribe()
	}
	return out, nil
}

func (h *retainedHost) Load(
	_ context.Context,
	req source.LookupRequest,
) ([]source.Materialized[retrieval.Document[payloadMeta]], error) {
	if h.mutateInput && !req.Read.IsScoped() {
		return nil, ragy.ErrProtocol
	}
	h.loadCalls++
	h.loaded = append(h.loaded, req.References...)
	var out []source.Materialized[retrieval.Document[payloadMeta]]
	for _, ref := range req.References {
		if doc, exists := h.payloads[ref]; exists {
			actual := ref
			if h.wrongPayload {
				actual.Revision = "r2"
				doc = h.payloads[actual]
			}
			out = append(out, source.Materialized[retrieval.Document[payloadMeta]]{Reference: actual, Payload: doc})
		}
	}
	if h.afterLoad != nil {
		h.afterLoad()
	}
	return out, nil
}

type hydrationFixture struct {
	host    *retainedHost
	reader  *documents.Hydrator[permissionMeta, payloadMeta]
	read    access.Binding
	r1, r2  source.Reference
	revoked *bool
}

func newHydrationFixture(t *testing.T) hydrationFixture {
	t.Helper()
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	visibility, err := fields.String("visibility")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.In(filter.Eq(builder, tenant, "a"), visibility, "public").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	revoked := new(bool)
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "policy",
			PolicyEpoch: 7,
			IssuedAt:    now,
			ExpiresAt:   now.Add(30 * time.Second),
		},
		Mandatory:   mandatory,
		Schema:      schema,
		Publication: access.CurrentPublication(),
		Now:         func() time.Time { return now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			if *revoked {
				return ragy.ErrUnavailable
			}
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	r1 := source.Reference{
		Namespace:         "ns",
		Source:            "manual",
		Revision:          "r1",
		Transformation:    "text-transform",
		AccessFingerprint: "acl",
		Artifact:          "c1",
		Representation:    "page-text",
	}
	r2 := r1
	r2.Revision = "r2"
	host := &retainedHost{
		descriptors: map[source.Reference]permissionMeta{
			r1: {tenant: "a", visibility: "public"},
			r2: {tenant: "a", visibility: "public"},
		},
		payloads: map[source.Reference]retrieval.Document[payloadMeta]{
			r1: {ID: "c1", Content: "old revision", Meta: payloadMeta{"title": "old"}},
			r2: {ID: "c1", Content: "new revision", Meta: payloadMeta{"title": "new"}},
		},
	}
	reader, err := documents.NewHydrator(documents.HydrationConfig[permissionMeta, payloadMeta]{
		Target: "lexical", Schema: schema, Catalog: host, Loader: host,
		Attributes: func(meta permissionMeta) (filter.RawAttributes, error) {
			return filter.RawAttributes{"tenant": meta.tenant, "visibility": meta.visibility}, nil
		},
		CloneMeta: func(meta payloadMeta) (payloadMeta, error) { return maps.Clone(meta), nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	return hydrationFixture{host: host, reader: reader, read: read, r1: r1, r2: r2, revoked: revoked}
}
func (f hydrationFixture) request() source.LookupRequest {
	return source.LookupRequest{Read: f.read, References: []source.Reference{f.r1}, Filters: filter.Condition{}}
}

func TestHydrationPinsRetainedRevisionAndOwnsMetadata(t *testing.T) {
	// Arrange: r2 exists but the request and citation identity remain r1.
	fixture := newHydrationFixture(t)
	fixture.host.mutateInput = true
	// Act.
	result, err := fixture.reader.Lookup(t.Context(), fixture.request())
	// Assert.
	if err != nil || len(result) != 1 || result[0].Reference != fixture.r1 ||
		result[0].Payload.Content != "old revision" ||
		fixture.host.loaded[0] != fixture.r1 {
		t.Fatalf("revision substituted: %v, %v", result, err)
	}
	result[0].Payload.Meta["title"] = "changed"
	if fixture.host.payloads[fixture.r1].Meta["title"] != "old" {
		t.Fatal("hydration metadata aliases retained source")
	}
}
func TestHydrationDeletedOrDeniedRevisionNeverLoadsPayload(t *testing.T) {
	for _, kind := range []string{"deleted", "denied", "wrong-descriptor"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange.
			fixture := newHydrationFixture(t)
			switch kind {
			case "deleted":
				delete(fixture.host.descriptors, fixture.r1)
				delete(fixture.host.payloads, fixture.r1)
			case "denied":
				fixture.host.descriptors[fixture.r1] = permissionMeta{tenant: "b", visibility: "private"}
			case "wrong-descriptor":
				fixture.host.wrongDescriptor = true
			}
			// Act.
			result, err := fixture.reader.Lookup(t.Context(), fixture.request())
			// Assert: neither fallback to r2 nor loading denied payload is permitted.
			expected := ragy.ErrUnavailable
			if kind == "wrong-descriptor" {
				expected = ragy.ErrProtocol
			}
			if !errors.Is(err, expected) || !access.IsProtectionFailure(err) || len(result) != 0 ||
				fixture.host.loadCalls != 0 {
				t.Fatalf("unavailable/denied source loaded: %v, calls=%d", err, fixture.host.loadCalls)
			}
		})
	}
}
func TestHydrationRevocationAndCancellationStopDelivery(t *testing.T) {
	for _, phase := range []string{"catalog", "payload", "canceled"} {
		t.Run(phase, func(t *testing.T) {
			// Arrange.
			fixture := newHydrationFixture(t)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			switch phase {
			case "catalog":
				fixture.host.afterDescribe = func() { *fixture.revoked = true }
			case "payload":
				fixture.host.afterLoad = func() { *fixture.revoked = true }
			case "canceled":
				cancel()
			}
			// Act.
			result, err := fixture.reader.Lookup(ctx, fixture.request())
			// Assert.
			if !access.IsProtectionFailure(err) || len(result) != 0 {
				t.Fatalf("revoked payload delivered: %v, %v", result, err)
			}
			if phase != "payload" && fixture.host.loadCalls != 0 {
				t.Fatal("payload loaded after pre-load denial")
			}
			if phase == "canceled" && fixture.host.describeCalls != 0 {
				t.Fatal("catalog called after cancellation")
			}
		})
	}
}
func TestHydrationRejectsLatestSubstitutionBeforePayloadConsumer(t *testing.T) {
	// Arrange: the host loader violates exact-revision identity after admission.
	fixture := newHydrationFixture(t)
	fixture.host.wrongPayload = true
	// Act.
	result, err := fixture.reader.Lookup(t.Context(), fixture.request())
	// Assert.
	if !errors.Is(err, ragy.ErrProtocol) || !access.IsProtectionFailure(err) || len(result) != 0 ||
		fixture.host.loadCalls != 1 {
		t.Fatalf("wrong revision delivered: %v, %v", result, err)
	}
}
func TestHydrationRejectsMismatchedPublicationBeforeCatalog(t *testing.T) {
	// Arrange: pinned inventory r2 cannot hydrate a requested r1 artifact.
	fixture := newHydrationFixture(t)
	publication, err := access.PinPublication(
		"publication",
		[]access.TargetRevision{
			{
				Target:            "lexical",
				Namespace:         "ns",
				Source:            "manual",
				Revision:          "r2",
				Transformation:    "text-transform",
				AccessFingerprint: "acl",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	request := fixture.request()
	request.Read, err = access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := fixture.reader.Lookup(t.Context(), request)
	// Assert.
	if !errors.Is(err, ragy.ErrUnavailable) || len(result) != 0 || fixture.host.describeCalls != 0 ||
		fixture.host.loadCalls != 0 {
		t.Fatalf("mixed publication hydrated: %v", err)
	}
}
func TestHydrationAllOrNothingAdmission(t *testing.T) {
	// Arrange: one requested artifact is public and the second is denied.
	fixture := newHydrationFixture(t)
	fixture.host.descriptors[fixture.r2] = permissionMeta{tenant: "a", visibility: "private"}
	request := fixture.request()
	request.References = append(request.References, fixture.r2)
	// Act.
	result, err := fixture.reader.Lookup(t.Context(), request)
	// Assert: no payload port calls before the complete batch has been admitted.
	if err == nil || len(result) != 0 || fixture.host.loadCalls != 0 {
		t.Fatalf("partial batch loaded before admission: %v", err)
	}
}
