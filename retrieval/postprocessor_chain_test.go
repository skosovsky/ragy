package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func retrieveWithPostProcessors[TMeta any](
	t *testing.T,
	backend Backend[struct{}, TMeta],
	resolver IdentityResolver[TMeta],
	query string,
	opts RetrieveOptions,
	processors ...PostProcessor[TMeta],
) (ResultSet[TMeta], error) {
	t.Helper()

	pipeline, err := NewPipelineBuilder[struct{}, TMeta]().
		WithRoot(RetrieverNode[struct{}, TMeta]{Backend: backend, Resolver: resolver}).
		WithPostProcessors(processors...).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	return pipeline.Retrieve(context.Background(), Query[struct{}]{Text: query, Options: opts})
}

func TestPostProcessorChainRejectsInvalidBackendDocuments(t *testing.T) {
	t.Parallel()

	backend := invalidBackend{}
	out, err := retrieveWithPostProcessors(
		t,
		backend,
		DocumentIDResolver[struct{}]{},
		"q",
		RetrieveOptions{TopK: 1},
	)
	requireNonNilResultSetOnError(t, out, err)
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Retrieve() error = %v, want missing id", err)
	}
}

func TestPostProcessorChainRejectsInvalidBackendScore(t *testing.T) {
	t.Parallel()

	backend := invalidScoreBackend{}
	out, err := retrieveWithPostProcessors(
		t,
		backend,
		DocumentIDResolver[struct{}]{},
		"q",
		RetrieveOptions{TopK: 1},
	)
	requireNonNilResultSetOnError(t, out, err)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol", err)
	}
}

func TestPostProcessorChainRejectsInvalidProcessorOutput(t *testing.T) {
	t.Parallel()

	backend := stubBackend[struct{}]{
		docs: []Document[struct{}]{{ID: "doc-1", Content: "ok", Score: 0.5}},
	}
	out, err := retrieveWithPostProcessors(
		t,
		backend,
		DocumentIDResolver[struct{}]{},
		"q",
		RetrieveOptions{TopK: 1},
		brokenProcessor[struct{}]{},
	)
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Retrieve() error = %v, want missing id", err)
	}
	if out == nil || out.IsEmpty() {
		t.Fatalf("Documents() = %#v, want preserved invalid processor output", out.Documents())
	}
}

func TestPostProcessorChainRejectsInvalidProcessorScore(t *testing.T) {
	t.Parallel()

	backend := stubBackend[struct{}]{
		docs: []Document[struct{}]{{ID: "doc-1", Content: "ok", Score: 0.5}},
	}
	out, err := retrieveWithPostProcessors(
		t,
		backend,
		DocumentIDResolver[struct{}]{},
		"q",
		RetrieveOptions{TopK: 1},
		brokenScoreProcessor[struct{}]{},
	)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol", err)
	}
	if out == nil || out.IsEmpty() {
		t.Fatalf("Documents() = %#v, want preserved invalid processor output", out.Documents())
	}
}

type invalidBackend struct{}

func (invalidBackend) Retrieve(_ context.Context, _ Query[struct{}]) (ResultSet[struct{}], error) {
	return NewResultSet([]Document[struct{}]{{Content: "broken", Score: 0.5}}, DocumentIDResolver[struct{}]{}), nil
}

type invalidScoreBackend struct{}

func (invalidScoreBackend) Retrieve(_ context.Context, _ Query[struct{}]) (ResultSet[struct{}], error) {
	return NewResultSet(
		[]Document[struct{}]{{ID: "doc-1", Content: "broken", Score: 1.5}},
		DocumentIDResolver[struct{}]{},
	), nil
}

type brokenProcessor[TMeta any] struct{}

func (brokenProcessor[TMeta]) Process(_ ResultSet[TMeta]) (ResultSet[TMeta], error) {
	return NewResultSet([]Document[TMeta]{{Content: "broken", Score: 0.5}}, DocumentIDResolver[TMeta]{}), nil
}

type brokenScoreProcessor[TMeta any] struct{}

func (brokenScoreProcessor[TMeta]) Process(_ ResultSet[TMeta]) (ResultSet[TMeta], error) {
	return NewResultSet(
		[]Document[TMeta]{{ID: "doc-1", Content: "broken", Score: 1.5}},
		DocumentIDResolver[TMeta]{},
	), nil
}

type passthroughProcessor[TMeta any] struct {
	resolver IdentityResolver[TMeta]
}

func (p passthroughProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	return NewResultSet(rs.Documents(), p.resolver), nil
}

func TestCustomPostProcessorMustUseConstructorResolver(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	backend := orchestratorStubBackend[struct{}, struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "grp", Score: 1},
	}}
	rs, err := retrieveWithPostProcessors(
		t,
		backend,
		resolver,
		"q",
		RetrieveOptions{TopK: 1},
		passthroughProcessor[struct{}]{resolver: resolver},
	)
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want passthrough with constructor resolver", rs.Documents())
	}
}

type resolverProbeProcessor[TMeta any] struct{}

func (resolverProbeProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	return rs, nil
}

func TestPostProcessorChainDoesNotInjectResolverIntoCustomProcessor(t *testing.T) {
	t.Parallel()

	mergeResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	backend := orchestratorStubBackend[struct{}, struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "same-key", Score: 0.9},
		{ID: "b", Content: "same-key", Score: 0.5},
	}}
	pipeline, err := NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(RetrieverNode[struct{}, struct{}]{Backend: backend}).
		WithPostProcessors(resolverProbeProcessor[struct{}]{}).
		WithResolver(mergeResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), Query[struct{}]{Text: "q", Options: RetrieveOptions{TopK: 5}})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (custom processor not merged by pipeline WithResolver)", rs.Len())
	}
}

type failingProcessor[TMeta any] struct {
	err error
}

func (p failingProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	return rs, p.err
}

func TestPostProcessorChainPreservesBackendResultOnError(t *testing.T) {
	t.Parallel()

	out, err := retrieveWithPostProcessors(
		t,
		partialBackend[struct{}, struct{}]{
			docs: []Document[struct{}]{{ID: "a", Content: "hit", Score: 1}},
			err:  ragy.ErrUnavailable,
		},
		DocumentIDResolver[struct{}]{},
		"q",
		RetrieveOptions{TopK: 1},
	)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved backend docs", out.Documents())
	}
}

func TestPostProcessorChainProcessPreservesResultOnInvalidOptions(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{{ID: "a", Content: "hit", Score: 1}}, DocumentIDResolver[struct{}]{})
	chain := NewPostProcessorChain[struct{}]()

	out, err := chain.Process(context.Background(), RetrieveOptions{FetchLimit: 1, TopK: 3}, rs)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Process() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved input docs", out.Documents())
	}
}

func TestPostProcessorChainInitialValidationPreservesResult(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{{Content: "broken", Score: 0.5}}, DocumentIDResolver[struct{}]{})
	chain := NewPostProcessorChain[struct{}]()

	out, err := chain.Process(context.Background(), RetrieveOptions{TopK: 1}, rs)
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Process() error = %v, want missing id", err)
	}
	if out.Len() != 1 {
		t.Fatalf("Len() = %d, want preserved invalid doc", out.Len())
	}
}

func TestPostProcessorChainRetrievePreservesInvalidBackendDocs(t *testing.T) {
	t.Parallel()

	out, err := retrieveWithPostProcessors(
		t,
		invalidBackend{},
		DocumentIDResolver[struct{}]{},
		"q",
		RetrieveOptions{TopK: 1},
	)
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Retrieve() error = %v, want missing id", err)
	}
	if out == nil || out.IsEmpty() {
		t.Fatalf("Documents() = %#v, want preserved invalid backend doc", out.Documents())
	}
}

func TestPostProcessorChainPreservesResultOnProcessorError(t *testing.T) {
	t.Parallel()

	backend := stubBackend[struct{}]{
		docs: []Document[struct{}]{{ID: "doc-1", Content: "ok", Score: 0.5}},
	}
	out, err := retrieveWithPostProcessors(
		t,
		backend,
		DocumentIDResolver[struct{}]{},
		"q",
		RetrieveOptions{TopK: 1},
		failingProcessor[struct{}]{err: ragy.ErrUnavailable},
	)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "doc-1" {
		t.Fatalf("Documents() = %#v, want preserved input docs", out.Documents())
	}
}

func TestPostProcessorChainRewrapsOnNoOpMinSimilarity(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	rs := NewResultSet([]Document[struct{}]{{ID: "a", Content: "grp", Score: 0.5}}, DocumentIDResolver[struct{}]{})
	chain := NewPostProcessorChainWithResolver[struct{}](resolver)

	out, err := chain.Process(context.Background(), RetrieveOptions{TopK: 1, MinSimilarity: 0}, rs)
	if err != nil {
		t.Fatalf("Process(): %v", err)
	}
	other, mergeErr := out.Merge(NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "grp", Score: 0.1},
	}, resolver))
	if mergeErr != nil {
		t.Fatalf("Merge() error = %v", mergeErr)
	}
	if other.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 by content merge key", other.Len())
	}
}

func TestPostProcessorChainRewrapsOnNoOpTopK(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	rs := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "grp", Score: 0.9},
		{ID: "b", Content: "other", Score: 0.5},
	}, DocumentIDResolver[struct{}]{})
	chain := NewPostProcessorChainWithResolver[struct{}](resolver)

	out, err := chain.Process(context.Background(), RetrieveOptions{TopK: 0, FetchLimit: 10}, rs)
	if err != nil {
		t.Fatalf("Process(): %v", err)
	}
	if out.Len() != 2 {
		t.Fatalf("Len() = %d, want all docs when top_k is zero", out.Len())
	}
	other, mergeErr := out.Merge(NewResultSet([]Document[struct{}]{
		{ID: "c", Content: "grp", Score: 0.1},
	}, resolver))
	if mergeErr != nil {
		t.Fatalf("Merge() error = %v", mergeErr)
	}
	if other.Len() != 2 {
		t.Fatalf("merged Len() = %d, want dedup by content merge key", other.Len())
	}
}

func TestApplyTopKSelectsHighestScore(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{
		{ID: "low", Score: 0.1},
		{ID: "high", Score: 0.9},
	}, DocumentIDResolver[struct{}]{})
	out := applyTopK(rs, 1, DocumentIDResolver[struct{}]{})
	if out.Len() != 1 || out.Documents()[0].ID != "high" {
		t.Fatalf("Documents() = %#v, want highest score doc", out.Documents())
	}
}

func TestApplyTopKSortsWhenLenEqualsTopK(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{
		{ID: "low", Score: 0.1},
		{ID: "high", Score: 0.9},
	}, DocumentIDResolver[struct{}]{})
	out := applyTopK(rs, 2, DocumentIDResolver[struct{}]{})
	if out.Len() != 2 || out.Documents()[0].ID != "high" {
		t.Fatalf("Documents() = %#v, want highest score first", out.Documents())
	}
}

func TestApplyTopKPreservesTieOrder(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{
		{ID: "b", Score: 0.5},
		{ID: "a", Score: 0.5},
	}, DocumentIDResolver[struct{}]{})
	out := applyTopK(rs, 2, DocumentIDResolver[struct{}]{})
	docs := out.Documents()
	if len(docs) != 2 || docs[0].ID != "b" || docs[1].ID != "a" {
		t.Fatalf("Documents() = %#v, want stable input order on score tie", docs)
	}
}

func TestPostProcessorChainRewrapsOnActiveMinSimilarity(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	rs := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "grp", Score: 0.9},
		{ID: "b", Content: "other", Score: 0.2},
	}, DocumentIDResolver[struct{}]{})
	chain := NewPostProcessorChainWithResolver[struct{}](resolver)

	out, err := chain.Process(context.Background(), RetrieveOptions{TopK: 5, MinSimilarity: 0.5}, rs)
	if err != nil {
		t.Fatalf("Process(): %v", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want doc above min similarity", out.Documents())
	}
	other, mergeErr := out.Merge(NewResultSet([]Document[struct{}]{
		{ID: "c", Content: "grp", Score: 0.1},
	}, resolver))
	if mergeErr != nil {
		t.Fatalf("Merge() error = %v", mergeErr)
	}
	if other.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 by content merge key", other.Len())
	}
}

func TestPostProcessorChainRewrapsOnActiveTopK(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	rs := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "grp", Score: 0.9},
		{ID: "b", Content: "other", Score: 0.5},
	}, DocumentIDResolver[struct{}]{})
	chain := NewPostProcessorChainWithResolver[struct{}](resolver)

	out, err := chain.Process(context.Background(), RetrieveOptions{TopK: 1}, rs)
	if err != nil {
		t.Fatalf("Process(): %v", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want top scored doc", out.Documents())
	}
	other, mergeErr := out.Merge(NewResultSet([]Document[struct{}]{
		{ID: "c", Content: "grp", Score: 0.1},
	}, resolver))
	if mergeErr != nil {
		t.Fatalf("Merge() error = %v", mergeErr)
	}
	if other.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 by content merge key", other.Len())
	}
}
