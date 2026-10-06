package pdf_test

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/access"
	pdfadapter "github.com/skosovsky/ragy/adapters/pdf"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func TestActualParserOCRSimulationAndModalityIndexRender(t *testing.T) {
	// Arrange: actual parsing and explicit simulated unreadable OCR are separate.
	ctx := context.Background()
	parser, err := pdfadapter.New(parserConfig(integrationPython(t)))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parser.Parse(ctx, layout.Input{Reference: fixtureReference(), Data: loadFixture(t)})
	if err != nil {
		t.Fatal(err)
	}
	simulated := layout.OCRObservation{
		Source:         parsed.Pages[1].Images[0].Location,
		State:          layout.OCRUnreadable,
		Transformation: "fixture-ocr-simulation",
	}
	observed, err := layout.ApplyOCR(ctx, parsed, []layout.OCRObservation{simulated})
	if err != nil {
		t.Fatal(err)
	}
	schema, read := parserScope(t)
	projected, err := layout.Project(ctx, observed, layout.ProjectionOptions{Read: read,
		ImageText: func(_ context.Context, _ access.Binding, image layout.Image) (source.MappedText, error) {
			if image.Location.Page.PhysicalIndex != 0 {
				return image.OCR.Mapping()
			}
			return source.DerivedText("fixture diagram", []source.Locator{image.Location})
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	byID := make(map[string]layout.Projected, len(projected))
	var docs []retrieval.Document[sourceMeta]
	for _, evidence := range projected {
		byID[evidence.ID] = evidence
		docs = append(
			docs,
			retrieval.Document[sourceMeta]{
				ID:      evidence.ID,
				Content: evidence.Text.Text(),
				Meta: sourceMeta{
					Organization: "a",
					Visibility:   "public",
					Artifact:     evidence.Location.Reference.Artifact,
					Revision:     evidence.Location.Reference.Revision,
					Coverage:     string(evidence.Coverage),
				},
			},
		)
	}
	index, err := lexical.NewBM25Index[sourceMeta](
		schema,
		lexical.Config[sourceMeta]{SearchFields: []string{"content"}},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if indexErr := index.Index(docs); indexErr != nil {
		t.Fatal(indexErr)
	}
	retention := newModalityRetention(t, observed, projected)
	// Act/Assert: retrieved supports resolve through scoped original retention.
	for _, query := range []string{"Revenue", "diagram"} {
		t.Run(query, func(t *testing.T) {
			checkModalityQuery(ctx, t, index, read, byID, query, retention)
		})
	}
	if observed.Pages[1].Diagnostics[0].Code != layout.DiagnosticOCRUnreadable ||
		observed.Pages[1].Coverage != layout.Partial {
		t.Fatal("simulated partial OCR was discarded")
	}
}

func checkModalityQuery(
	ctx context.Context,
	t *testing.T,
	index *lexical.BM25Index[sourceMeta],
	read access.Binding,
	byID map[string]layout.Projected,
	query string,
	retention *layoutPayloadHost,
) {
	t.Helper()
	results, retrieveErr := index.Retrieve(
		ctx,
		retrieval.Query[struct{}]{Read: read, Text: query, Options: retrieval.RetrieveOptions{TopK: 10}},
	)
	if retrieveErr != nil {
		t.Fatal(retrieveErr)
	}
	artifact, renderErr := (retrieval.DefaultArtifactRenderer[sourceMeta]{}).Render(
		ctx,
		read,
		results,
		retrieval.ArtifactRenderOptions[sourceMeta]{
			CloneMeta: func(meta sourceMeta) (sourceMeta, error) { return meta, nil },
			Mapping:   func(doc retrieval.Document[sourceMeta]) (source.MappedText, error) { return byID[doc.ID].Text, nil },
		},
	)
	if renderErr != nil {
		t.Fatal(renderErr)
	}
	if len(artifact.Snippets) == 0 {
		t.Fatal("modality evidence not retrieved")
	}

	checkModalitySnippets(t, artifact, byID, query)
	checkRetrievedSupports(ctx, t, read, artifact, retention)
}

func checkModalitySnippets(
	t *testing.T,
	artifact retrieval.RetrievalContextArtifact[sourceMeta],
	byID map[string]layout.Projected,
	query string,
) {
	t.Helper()
	cells, images := 0, 0
	for _, snippet := range artifact.Snippets {
		evidence := byID[snippet.DocumentID]
		if snippet.Meta.Coverage != "partial" || evidence.Coverage != layout.Partial {
			t.Fatal("projection/index/render promoted source coverage")
		}
		switch evidence.Location.Kind {
		case source.CellLocation:
			cells++
			if snippet.Content != "Revenue" || snippet.Mapping.Fragments()[0].Origin != source.OriginalContent {
				t.Fatal("cell text lost original support")
			}
		case source.ImageLocation:
			images++
			if snippet.Content != "fixture diagram" ||
				snippet.Mapping.Fragments()[0].Origin != source.DerivedContent {
				t.Fatal("description presented as original image text")
			}
		case source.DocumentLocation, source.TextLocation, source.PageLocation, source.RegionLocation:
		}
	}
	if query == "Revenue" && cells != 1 {
		t.Fatal("merged cell duplicated as independent retrieval sources")
	}
	if query == "diagram" && images != 1 {
		t.Fatal("unreadable OCR fabricated another image description")
	}
}
