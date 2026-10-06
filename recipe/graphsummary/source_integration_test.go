package graphsummary_test

import (
	"context"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/source"
)

type sourceHost struct {
	rows    map[source.Reference]string
	deleted bool
	private bool
	loads   int
}

func (h *sourceHost) Describe(_ context.Context, request source.LookupRequest) ([]source.Descriptor[acl], error) {
	var result []source.Descriptor[acl]
	for _, reference := range request.References {
		if h.deleted {
			return nil, nil
		}
		if _, ok := h.rows[reference]; !ok {
			return nil, nil
		}
		tenant := "a"
		if h.private && reference.Source == "s2" {
			tenant = "b"
		}
		result = append(result, source.Descriptor[acl]{Reference: reference, Access: acl{Tenant: tenant, Secret: nil}})
	}
	return result, nil
}
func (h *sourceHost) Load(_ context.Context, request source.LookupRequest) ([]source.Materialized[string], error) {
	h.loads++
	var result []source.Materialized[string]
	for _, reference := range request.References {
		result = append(result, source.Materialized[string]{Reference: reference, Payload: h.rows[reference]})
	}
	return result, nil
}

func reader(t *testing.T, f *fixture, h *sourceHost) *source.Reader[acl, string] {
	t.Helper()
	result, err := source.NewReader(
		source.ReadConfig[acl, string]{
			Target:     "source",
			Schema:     f.config.Schema,
			Catalog:    h,
			Loader:     h,
			Attributes: func(a acl) (filter.RawAttributes, error) { return filter.RawAttributes{"tenant": a.Tenant}, nil },
			ValidatePayload: func(_ source.Reference, text string) error {
				if text == "" {
					return ragy.ErrUnavailable
				}
				return nil
			},
			ClonePayload: func(text string) (string, error) { return text, nil },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestRetainedSourceReaderSummaryAndDeletionInvalidation(t *testing.T) {
	// Arrange: actual scoped Reader admits both retained original representations as one batch.
	f := newFixture(t)
	h := &sourceHost{rows: make(map[source.Reference]string), deleted: false, private: false, loads: 0}
	var references []source.Reference
	for _, community := range f.request.Communities {
		snippet := community.Snippets[0]
		reference := snippet.Mapping.Supports()[0].Reference
		h.rows[reference] = snippet.Mapping.Text()
		references = append(references, reference)
	}
	retained := reader(t, f, h)
	loaded, err := retained.Lookup(
		context.Background(),
		source.LookupRequest{Read: f.request.Read, References: references, Filters: filter.Condition{}},
	)
	if err != nil || len(loaded) != 2 || h.loads != 1 {
		t.Fatal(loaded, err, h.loads)
	}
	for i, payload := range loaded {
		location := f.request.Communities[i].Snippets[0].Mapping.Supports()[0]
		mapping, mapErr := source.OriginalText(location, payload.Payload)
		if mapErr != nil {
			t.Fatal(mapErr)
		}
		f.request.Communities[i].Snippets[0].Mapping = mapping
	}
	f.config.AdmitSource = func(ctx context.Context, read access.Binding, location source.Locator) error {
		rows, lookupErr := retained.Lookup(
			ctx,
			source.LookupRequest{
				Read:       read,
				References: []source.Reference{location.Reference},
				Filters:    filter.Condition{},
			},
		)
		if lookupErr != nil {
			return lookupErr
		}
		if len(rows) != 1 {
			return ragy.ErrUnavailable
		}
		return location.Span.ValidateText(rows[0].Payload)
	}
	// Act.
	result, _, err := run(context.Background(), t, f, true)
	// Assert: generated artifact cites both admitted original source revisions.
	if err != nil || result.Global == nil || len(result.Global.Supports()) != 2 {
		t.Fatal(result, err)
	}
	h.deleted = true
	before := h.loads
	mapping, err := result.Global.Resolve(context.Background(), f.request.Read, f.config.AdmitSource)
	if !access.IsProtectionFailure(err) || mapping.Text() != "" || h.loads != before {
		t.Fatal(mapping.Text(), err, h.loads, before)
	}
}

func TestRetainedSummaryInputDeniesPrivateLastDescriptorBeforePayloadLoad(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	h := &sourceHost{rows: make(map[source.Reference]string), deleted: false, private: true, loads: 0}
	var references []source.Reference
	for _, community := range f.request.Communities {
		snippet := community.Snippets[0]
		reference := snippet.Mapping.Supports()[0].Reference
		h.rows[reference] = snippet.Mapping.Text()
		references = append(references, reference)
	}
	retained := reader(t, f, h)
	// Act.
	loaded, err := retained.Lookup(
		context.Background(),
		source.LookupRequest{Read: f.request.Read, References: references, Filters: filter.Condition{}},
	)
	// Assert.
	if !access.IsProtectionFailure(err) || len(loaded) != 0 || h.loads != 0 {
		t.Fatal(loaded, err, h.loads)
	}
}
