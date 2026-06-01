package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestPipelineRejectsInvalidBackendDocuments(t *testing.T) {
	t.Parallel()

	backend := invalidBackend{}
	pipeline := NewPipeline(backend)

	_, err := pipeline.Retrieve(context.Background(), "q", RetrieveOptions{TopK: 1})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Retrieve() error = %v, want missing id", err)
	}
}

func TestPipelineRejectsInvalidBackendScore(t *testing.T) {
	t.Parallel()

	backend := invalidScoreBackend{}
	pipeline := NewPipeline(backend)

	_, err := pipeline.Retrieve(context.Background(), "q", RetrieveOptions{TopK: 1})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

func TestPipelineRejectsInvalidProcessorOutput(t *testing.T) {
	t.Parallel()

	backend := stubBackend[struct{}]{
		docs: []Document[struct{}]{{ID: "doc-1", Content: "ok", Score: 0.5}},
	}
	pipeline := NewPipeline(backend, brokenProcessor[struct{}]{})

	_, err := pipeline.Retrieve(context.Background(), "q", RetrieveOptions{TopK: 1})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Retrieve() error = %v, want missing id", err)
	}
}

func TestPipelineRejectsInvalidProcessorScore(t *testing.T) {
	t.Parallel()

	backend := stubBackend[struct{}]{
		docs: []Document[struct{}]{{ID: "doc-1", Content: "ok", Score: 0.5}},
	}
	pipeline := NewPipeline(backend, brokenScoreProcessor[struct{}]{})

	_, err := pipeline.Retrieve(context.Background(), "q", RetrieveOptions{TopK: 1})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

type invalidBackend struct{}

func (invalidBackend) Retrieve(_ context.Context, _ string, _ RetrieveOptions) ([]Document[struct{}], error) {
	return []Document[struct{}]{{Content: "broken", Score: 0.5}}, nil
}

type invalidScoreBackend struct{}

func (invalidScoreBackend) Retrieve(_ context.Context, _ string, _ RetrieveOptions) ([]Document[struct{}], error) {
	return []Document[struct{}]{{ID: "doc-1", Content: "broken", Score: 1.5}}, nil
}

type brokenProcessor[TMeta any] struct{}

func (brokenProcessor[TMeta]) Process(_ []Document[TMeta]) ([]Document[TMeta], error) {
	return []Document[TMeta]{{Content: "broken", Score: 0.5}}, nil
}

type brokenScoreProcessor[TMeta any] struct{}

func (brokenScoreProcessor[TMeta]) Process(_ []Document[TMeta]) ([]Document[TMeta], error) {
	return []Document[TMeta]{{ID: "doc-1", Content: "broken", Score: 1.5}}, nil
}
