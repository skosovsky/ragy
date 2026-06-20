package retrieval

import (
	"strings"
	"testing"
)

func TestDefaultArtifactRendererBuildsBudgetedArtifact(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{ Source string }]{
		{
			ID:         "doc-1",
			Content:    "alpha beta",
			ScoreState: ScoreAbsent,
			Rank:       3,
			Meta:       struct{ Source string }{Source: "kb"},
		},
	}, DocumentIDResolver[struct{ Source string }]{})

	artifact, err := DefaultArtifactRenderer[struct{ Source string }]{}.Render(
		rs,
		ArtifactRenderOptions[struct{ Source string }]{
			Budget: 5,
			Provenance: func(doc Document[struct{ Source string }]) Provenance {
				return Provenance{SourceID: doc.ID, URI: "file://doc-1", Label: doc.Meta.Source}
			},
			Diagnostics: []PlannerDiagnostic{{Key: "planner", Value: "ok"}},
		},
	)
	if err != nil {
		t.Fatalf("Render(): %v", err)
	}
	if artifact.UntrustedDataBoundary == "" {
		t.Fatal("UntrustedDataBoundary is empty")
	}
	if artifact.Budget.Used != 5 {
		t.Fatalf("Budget.Used = %d, want 5", artifact.Budget.Used)
	}
	if len(artifact.Snippets) != 1 {
		t.Fatalf("len(Snippets) = %d, want 1", len(artifact.Snippets))
	}
	snippet := artifact.Snippets[0]
	if snippet.Content != "alpha" {
		t.Fatalf("Content = %q, want budget-trimmed alpha", snippet.Content)
	}
	if snippet.Provenance.URI != "file://doc-1" {
		t.Fatalf("Provenance = %#v, want URI", snippet.Provenance)
	}
	if snippet.ScoreState != ScoreAbsent || snippet.Rank != 3 {
		t.Fatalf("score/rank = (%v,%d), want absent rank 3", snippet.ScoreState, snippet.Rank)
	}
	if len(artifact.Diagnostics) != 1 || artifact.Diagnostics[0].Key != "planner" {
		t.Fatalf("Diagnostics = %#v, want planner diagnostic", artifact.Diagnostics)
	}
	if !strings.Contains(artifact.RenderedText, artifact.UntrustedDataBoundary) {
		t.Fatalf("RenderedText = %q, want untrusted boundary", artifact.RenderedText)
	}
}

func TestDefaultArtifactRendererRejectsNegativeBudget(t *testing.T) {
	t.Parallel()

	_, err := DefaultArtifactRenderer[struct{}]{}.Render(nil, ArtifactRenderOptions[struct{}]{Budget: -1})
	if err == nil {
		t.Fatal("Render() error = nil, want error")
	}
}

func TestDefaultArtifactRendererDedupsAndFormatsRenderedText(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{
		{ID: "doc-1", Content: "alpha", Score: 1},
		{ID: "doc-2", Content: "alpha duplicate", Score: 0.9},
	}, DocumentIDResolver[struct{}]{})

	artifact, err := DefaultArtifactRenderer[struct{}]{}.Render(rs, ArtifactRenderOptions[struct{}]{
		UntrustedDataBoundary: "UNTRUSTED",
		DedupKey: func(_ Document[struct{}]) string {
			return "same-source"
		},
		FormatSnippet: func(snippet ContextSnippet[struct{}]) string {
			return snippet.DocumentID + ": " + snippet.Content
		},
	})
	if err != nil {
		t.Fatalf("Render(): %v", err)
	}
	if len(artifact.Snippets) != 1 {
		t.Fatalf("len(Snippets) = %d, want deduped 1", len(artifact.Snippets))
	}
	if artifact.RenderedText != "UNTRUSTED\n\ndoc-1: alpha" {
		t.Fatalf("RenderedText = %q, want custom formatted text", artifact.RenderedText)
	}
}
