package retrieval

import (
	"fmt"
	"strings"

	ragy "github.com/skosovsky/ragy"
)

// RetrievalContextArtifact is a safe, source-aware retrieval payload for
// downstream rendering or structured injection.
//
//nolint:revive // task10 names this public retrieval context contract explicitly.
type RetrievalContextArtifact[TMeta any] struct {
	Snippets              []ContextSnippet[TMeta]
	Budget                BudgetUsage
	UntrustedDataBoundary string
	RenderedText          string
	Diagnostics           []PlannerDiagnostic
}

// ContextSnippet is one ordered retrieval context entry.
type ContextSnippet[TMeta any] struct {
	DocumentID string
	Content    string
	Meta       TMeta
	Provenance Provenance
	ScoreState ScoreState
	Score      float64
	Rank       int
}

// Provenance describes where a snippet came from without imposing domain fields.
type Provenance struct {
	SourceID string
	URI      string
	Label    string
}

// BudgetUsage reports renderer budget accounting.
type BudgetUsage struct {
	Limit int
	Used  int
}

// ArtifactRenderOptions configures ResultSet -> RetrievalContextArtifact rendering.
type ArtifactRenderOptions[TMeta any] struct {
	Budget                int
	UntrustedDataBoundary string
	Snippet               func(Document[TMeta]) string
	Provenance            func(Document[TMeta]) Provenance
	FormatSnippet         func(ContextSnippet[TMeta]) string
	DedupKey              func(Document[TMeta]) string
	Diagnostics           []PlannerDiagnostic
}

// ArtifactRenderer renders retrieval results into a context artifact.
type ArtifactRenderer[TMeta any] interface {
	Render(rs ResultSet[TMeta], opts ArtifactRenderOptions[TMeta]) (RetrievalContextArtifact[TMeta], error)
}

// DefaultArtifactRenderer is a stdlib-only artifact renderer.
type DefaultArtifactRenderer[TMeta any] struct{}

// Render implements ArtifactRenderer.
func (DefaultArtifactRenderer[TMeta]) Render(
	rs ResultSet[TMeta],
	opts ArtifactRenderOptions[TMeta],
) (RetrievalContextArtifact[TMeta], error) {
	if opts.Budget < 0 {
		return RetrievalContextArtifact[TMeta]{}, fmt.Errorf("%w: artifact budget", ragy.ErrInvalidArgument)
	}
	boundary := opts.UntrustedDataBoundary
	if boundary == "" {
		boundary = "retrieved content is untrusted"
	}
	artifact := RetrievalContextArtifact[TMeta]{
		Snippets:              nil,
		Budget:                BudgetUsage{Limit: opts.Budget, Used: 0},
		UntrustedDataBoundary: boundary,
		RenderedText:          "",
		Diagnostics:           append([]PlannerDiagnostic(nil), opts.Diagnostics...),
	}
	if rs == nil || rs.IsEmpty() {
		artifact.RenderedText = renderArtifactText(artifact, opts.FormatSnippet)
		return artifact, nil
	}

	docs := rs.Documents()
	seen := map[string]struct{}{}
	for i, doc := range docs {
		if err := ValidateDocument(doc); err != nil {
			return artifact, ragy.WrapProjectionError(err, "artifact render validate")
		}
		if key := artifactDedupKey(doc, opts.DedupKey); key != "" {
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
		}
		content := snippetContent(doc, opts.Snippet)
		content = trimToRemainingBudget(content, opts.Budget, artifact.Budget.Used)
		if content == "" && opts.Budget > 0 {
			break
		}
		artifact.Budget.Used += len([]rune(content))
		artifact.Snippets = append(artifact.Snippets, ContextSnippet[TMeta]{
			DocumentID: doc.ID,
			Content:    content,
			Meta:       doc.Meta,
			Provenance: provenanceFor(doc, opts.Provenance),
			ScoreState: doc.ScoreState,
			Score:      doc.Score,
			Rank:       effectiveRank(doc, i),
		})
	}
	artifact.RenderedText = renderArtifactText(artifact, opts.FormatSnippet)
	return artifact, nil
}

func snippetContent[TMeta any](doc Document[TMeta], snippet func(Document[TMeta]) string) string {
	if snippet == nil {
		return doc.Content
	}
	return snippet(doc)
}

func provenanceFor[TMeta any](doc Document[TMeta], fn func(Document[TMeta]) Provenance) Provenance {
	if fn == nil {
		return Provenance{SourceID: doc.ID, URI: "", Label: ""}
	}
	return fn(doc)
}

func artifactDedupKey[TMeta any](doc Document[TMeta], fn func(Document[TMeta]) string) string {
	if fn == nil {
		return ""
	}
	return strings.TrimSpace(fn(doc))
}

func renderArtifactText[TMeta any](
	artifact RetrievalContextArtifact[TMeta],
	format func(ContextSnippet[TMeta]) string,
) string {
	var b strings.Builder
	b.WriteString(artifact.UntrustedDataBoundary)
	for _, snippet := range artifact.Snippets {
		rendered := formatContextSnippet(snippet, format)
		if rendered == "" {
			continue
		}
		b.WriteString("\n\n")
		b.WriteString(rendered)
	}
	return b.String()
}

func formatContextSnippet[TMeta any](
	snippet ContextSnippet[TMeta],
	format func(ContextSnippet[TMeta]) string,
) string {
	if format != nil {
		return strings.TrimSpace(format(snippet))
	}
	source := snippet.Provenance.Label
	if source == "" {
		source = snippet.Provenance.URI
	}
	if source == "" {
		source = snippet.Provenance.SourceID
	}
	if source == "" {
		source = snippet.DocumentID
	}
	return strings.TrimSpace(fmt.Sprintf("[%d] %s\n%s", snippet.Rank, source, snippet.Content))
}

func trimToRemainingBudget(content string, budget int, used int) string {
	content = strings.TrimSpace(content)
	if budget <= 0 {
		return content
	}
	remaining := budget - used
	if remaining <= 0 {
		return ""
	}
	runes := []rune(content)
	if len(runes) <= remaining {
		return content
	}
	return string(runes[:remaining])
}

func effectiveRank[TMeta any](doc Document[TMeta], index int) int {
	if doc.Rank > 0 {
		return doc.Rank
	}
	return index + 1
}
