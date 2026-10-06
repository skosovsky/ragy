package lexical

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

func TestBM25FailedRebuildPreservesPublishedState(t *testing.T) {
	t.Parallel()
	// Arrange.
	index, err := NewBM25Index[struct{}](
		filter.EmptySchema(), Config[struct{}]{SearchFields: []string{"content"}}, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	original := []retrieval.Document[struct{}]{{ID: "old", Content: "secret secret"}}
	if indexErr := index.Index(original); indexErr != nil {
		t.Fatal(indexErr)
	}
	request := retrieval.Query[struct{}]{
		Read:    retrieval.UnrestrictedRead(),
		Text:    "secret",
		Options: retrieval.RetrieveOptions{TopK: 10},
	}
	before, err := index.Retrieve(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	for _, replacement := range [][]retrieval.Document[struct{}]{
		{{ID: "bad", Content: ""}},
		{{ID: "new", Content: "replacement"}, {ID: "bad", Content: ""}},
		{{ID: "new", Content: "replacement"}, {Content: "no identity"}},
	} {
		// Act.
		rebuildErr := index.Index(replacement)
		after, retrieveErr := index.Retrieve(context.Background(), request)
		// Assert.
		if rebuildErr == nil || retrieveErr != nil || !reflect.DeepEqual(before.Documents(), after.Documents()) {
			t.Fatalf(
				"failed rebuild changed results: rebuild=%v retrieve=%v before=%v after=%v",
				rebuildErr,
				retrieveErr,
				before.Documents(),
				after.Documents(),
			)
		}
		if index.docCount != 1 || index.docLengths["old"] != 2 || index.avgLength != 2 || len(index.postings) != 1 {
			t.Fatalf(
				"failed rebuild changed statistics: count=%d lengths=%v average=%v postings=%v",
				index.docCount,
				index.docLengths,
				index.avgLength,
				index.postings,
			)
		}
	}
	// Act: intentional successful clearing is distinct from a failed rebuild.
	clearErr := index.Index(nil)
	empty, retrieveErr := index.Retrieve(context.Background(), request)
	// Assert.
	if clearErr != nil || retrieveErr != nil || !empty.IsEmpty() {
		t.Fatalf("clear=%v retrieve=%v docs=%v", clearErr, retrieveErr, empty.Documents())
	}
}

func TestBM25IntegerTenantFiltersDoNotCollapseIDs(t *testing.T) {
	t.Parallel()
	// Arrange.
	const tenantA int64 = 9007199254740992
	const tenantB int64 = 9007199254740993
	type meta struct {
		Tenant int64 `json:"tenant"`
	}
	builder := filter.NewSchema()
	tenant, err := builder.Int("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	index, err := NewBM25Index[meta](schema, Config[meta]{SearchFields: []string{"content"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := index.Index([]retrieval.Document[meta]{
		{ID: "a", Content: "secret", Meta: meta{Tenant: tenantA}},
		{ID: "b", Content: "secret", Meta: meta{Tenant: tenantB}},
	}); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name  string
		build func(*filter.Builder) *filter.Builder
		want  int64
	}{
		{"eq-a", func(b *filter.Builder) *filter.Builder { return filter.Eq(b, tenant, tenantA) }, tenantA},
		{"eq-b", func(b *filter.Builder) *filter.Builder { return filter.Eq(b, tenant, tenantB) }, tenantB},
		{"in-a", func(b *filter.Builder) *filter.Builder { return filter.In(b, tenant, tenantA) }, tenantA},
		{"in-b", func(b *filter.Builder) *filter.Builder { return filter.In(b, tenant, tenantB) }, tenantB},
		{"neq-a", func(b *filter.Builder) *filter.Builder { return filter.NotEq(b, tenant, tenantA) }, tenantB},
		{"neq-b", func(b *filter.Builder) *filter.Builder { return filter.NotEq(b, tenant, tenantB) }, tenantA},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			// Arrange.
			filterBuilder, buildErr := filter.NewBuilder(schema)
			if buildErr != nil {
				t.Fatal(buildErr)
			}
			condition, buildErr := tc.build(filterBuilder).Build()
			if buildErr != nil {
				t.Fatal(buildErr)
			}
			request := retrieval.Query[struct{}]{Read: retrieval.UnrestrictedRead(),
				Text:    "secret",
				Options: retrieval.RetrieveOptions{TopK: 10, Filters: condition},
			}
			// Act.
			result, retrieveErr := index.Retrieve(context.Background(), request)
			// Assert.
			if retrieveErr != nil || result.Len() != 1 || result.Documents()[0].Meta.Tenant != tc.want {
				t.Fatalf("docs=%v error=%v wantTenant=%d", result.Documents(), retrieveErr, tc.want)
			}
		})
	}
}

func TestBM25FailedCodecRebuildPreservesState(t *testing.T) {
	t.Parallel()
	// Arrange.
	type meta struct {
		Tenant string `json:"tenant"`
	}
	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	codec := &failingAfterFirstEncodeCodec[meta]{inner: retrieval.NewJSONCodec[meta](schema)}
	index, err := NewBM25Index[meta](schema, Config[meta]{SearchFields: []string{"tenant"}, Codec: codec}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := index.Index([]retrieval.Document[meta]{{ID: "old", Meta: meta{Tenant: "secret"}}}); err != nil {
		t.Fatal(err)
	}
	// Act.
	rebuildErr := index.Index([]retrieval.Document[meta]{{ID: "new", Meta: meta{Tenant: "replacement"}}})
	// Assert: inspect published state without invoking the deliberately failed codec again.
	if !errors.Is(rebuildErr, ragy.ErrInvalidArgument) || index.docCount != 1 ||
		index.docs["old"].Meta.Tenant != "secret" ||
		len(index.postings["secret"]) != 1 {
		t.Fatalf("rebuild=%v count=%d docs=%v postings=%v", rebuildErr, index.docCount, index.docs, index.postings)
	}
}

func TestBM25ReadersNeverObservePartialRebuild(t *testing.T) {
	// Arrange.
	index, err := NewBM25Index[struct{}](
		filter.EmptySchema(), Config[struct{}]{SearchFields: []string{"content"}}, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	old := []retrieval.Document[struct{}]{{ID: "old-a", Content: "common"}, {ID: "old-b", Content: "common"}}
	next := []retrieval.Document[struct{}]{{ID: "new-a", Content: "common"}, {ID: "new-b", Content: "common"}}
	if indexErr := index.Index(old); indexErr != nil {
		t.Fatal(indexErr)
	}
	request := retrieval.Query[struct{}]{
		Read:    retrieval.UnrestrictedRead(),
		Text:    "common",
		Options: retrieval.RetrieveOptions{TopK: 10},
	}
	// Act.
	var workers sync.WaitGroup
	for range 4 {
		workers.Go(func() {
			for range 32 {
				result, retrieveErr := index.Retrieve(context.Background(), request)
				// Assert: each query snapshot has the entire old or new publication.
				if retrieveErr != nil || result.Len() != 2 {
					t.Errorf("error=%v docs=%v", retrieveErr, result.Documents())
					return
				}
				docs := result.Documents()
				if strings.HasPrefix(docs[0].ID, "old") != strings.HasPrefix(docs[1].ID, "old") {
					t.Errorf("mixed snapshot: %v", docs)
				}
			}
		})
	}
	for range 16 {
		if indexErr := index.Index(next); indexErr != nil {
			t.Fatal(indexErr)
		}
		if indexErr := index.Index(old); indexErr != nil {
			t.Fatal(indexErr)
		}
	}
	workers.Wait()
}

type rebuildGateTokenizer struct {
	entered chan struct{}
	proceed chan struct{}
}

func (g rebuildGateTokenizer) Tokenize(text string) []string {
	if text == "replacement" {
		close(g.entered)
		<-g.proceed
	}
	return DefaultTokenizer{}.Tokenize(text)
}

func TestBM25ConcurrentUpsertSurvivesRebuildPublication(t *testing.T) {
	// Arrange.
	gate := rebuildGateTokenizer{entered: make(chan struct{}), proceed: make(chan struct{})}
	index, err := NewBM25Index[struct{}](
		filter.EmptySchema(), Config[struct{}]{SearchFields: []string{"content"}}, gate, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	rebuilt := make(chan error, 1)
	upserted := make(chan error, 1)
	// Act: staging holds the same serialization boundary used by Upsert.
	go func() { rebuilt <- index.Index([]retrieval.Document[struct{}]{{ID: "new", Content: "replacement"}}) }()
	<-gate.entered
	go func() { upserted <- index.Upsert(retrieval.Document[struct{}]{ID: "concurrent", Content: "added"}) }()
	close(gate.proceed)
	rebuildErr := <-rebuilt
	upsertErr := <-upserted
	// Assert.
	if rebuildErr != nil || upsertErr != nil || index.docCount != 2 || index.docs["concurrent"].Content != "added" {
		t.Fatalf("rebuild=%v upsert=%v count=%d docs=%v", rebuildErr, upsertErr, index.docCount, index.docs)
	}
}
